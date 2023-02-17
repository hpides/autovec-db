#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"
#include "simd.hpp"

using RowId = uint32_t;
using DictEntry = uint32_t;

using DictColumn = AlignedData<DictEntry, 64>;
using MatchingRows = AlignedData<RowId, 64>;

static constexpr size_t NUM_BASE_ROWS = 16;
static constexpr size_t SCALE_FACTOR = 1024ull * 64;
static constexpr size_t NUM_ROWS = NUM_BASE_ROWS * SCALE_FACTOR;
static constexpr size_t NUM_UNIQUE_VALUES = 16;

/*
 * Builds a lookup table that, given a comparison-result bitmask, returns the indices of the matching elements
 * compressed to the front. Can be used as a shuffle mask for source-selecting shuffles. Examples:
 * [0 0 0 1] -> [0, unused_index, unused_index, unused_index]
 * [1 0 1 0] -> [1, 3, unused_index, unused_index]
 */
template <size_t ComparisonResultBits, typename IndexT, IndexT unused_index>
static constexpr auto lookup_table_for_compressed_offsets_by_comparison_result() {
  std::array<std::array<IndexT, ComparisonResultBits>, 1 << ComparisonResultBits> lookup_table{};

  for (size_t index = 0; index < lookup_table.size(); ++index) {
    auto& shuffle_mask = lookup_table[index];
    std::fill(shuffle_mask.begin(), shuffle_mask.end(), unused_index);

    size_t first_empty_output_slot = 0;
    for (size_t comparison_result_rest = index; comparison_result_rest != 0;
         comparison_result_rest &= comparison_result_rest - 1) {
      shuffle_mask[first_empty_output_slot++] = std::countr_zero(comparison_result_rest);
    }
  }

  return lookup_table;
}

/*
 * SSE and NEON do not allow shuffling elements with run-time masks, so we have to create byte shuffle masks.
 * This transforms a shuffle mask like [1, 3, unused_index, unused_index] for `uint32_t`s to the byte mask
 * [4, 5, 6, 7,   12, 13, 14, 15,   U, U, U, U,   U, U, U, U ]
 */
template <typename VectorElementT, typename IndexT, IndexT unused_index>
static constexpr auto element_shuffle_table_to_byte_shuffle_table(auto element_shuffle_table) {
  static_assert(std::endian::native == std::endian::little, "Probably doesn't work for big-endian systems.");
  constexpr size_t OUTPUT_ELEMENTS_PER_MASK = sizeof(VectorElementT) * element_shuffle_table[0].size();
  std::array<std::array<IndexT, OUTPUT_ELEMENTS_PER_MASK>, element_shuffle_table.size()> byte_shuffle_table{};

  for (size_t row = 0; row < element_shuffle_table.size(); ++row) {
    const auto& element_shuffle_mask = element_shuffle_table[row];
    auto& byte_shuffle_mask = byte_shuffle_table[row];

    for (size_t element_index = 0; element_index < element_shuffle_mask.size(); ++element_index) {
      const auto element_mask_value = element_shuffle_mask[element_index];
      IndexT* byte_mask_group_begin = byte_shuffle_mask.data() + element_index * sizeof(VectorElementT);
      IndexT* byte_mask_group_end = byte_mask_group_begin + sizeof(VectorElementT);

      if (element_mask_value == unused_index) {
        std::fill(byte_mask_group_begin, byte_mask_group_end, unused_index);
      } else {
        std::iota(byte_mask_group_begin, byte_mask_group_end, element_mask_value * sizeof(VectorElementT));
      }
    }
  }

  return byte_shuffle_table;
}

struct naive_scalar_scan {
  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* column_data = column.aligned_data();
    RowId* output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    for (RowId row = 0; row < NUM_ROWS; ++row) {
      if (column_data[row] < filter_val) {
        output[num_matching_rows++] = row;
      }
    }
    return num_matching_rows;
  }
};

struct autovec_scalar_scan {
  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    // The naive version should be autovectorizable with clang, but they currently don't do this
    // see https://github.com/llvm/llvm-project/issues/42210
    // According to the issue, ICC can autovectorize this.
    // Godbolt playground: https://godbolt.org/z/aahTPczdr

    const DictEntry* __restrict column_data = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    for (RowId row = 0; row < NUM_ROWS; ++row) {
      output[num_matching_rows] = row;
      num_matching_rows += static_cast<int>(column_data[row] < filter_val);
    }
    return num_matching_rows;
  }
};

struct vector_128_scan_predication {
  using DictVec = simd::GccVec<DictEntry, 16>::T;
  static constexpr size_t NUM_MATCHES_PER_VECTOR = sizeof(DictVec) / sizeof(DictEntry);

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    RowId num_matching_rows = 0;

    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    static_assert(NUM_ROWS % (NUM_MATCHES_PER_VECTOR) == 0);
    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      const auto rows_to_match = simd::load<DictVec>(rows + chunk_start_row);
      const DictVec matches = rows_to_match < filter_val;

      for (RowId row = 0; row < NUM_MATCHES_PER_VECTOR; ++row) {
        output[num_matching_rows] = chunk_start_row + row;
        num_matching_rows += matches[row] & 1u;
      }
    }

    return num_matching_rows;
  }
};

struct vector_128_scan_shuffle {
  using DictVec = simd::GccVec<DictEntry, 16>::T;
  using RowVec = simd::GccVec<RowId, 16>::T;

  static constexpr size_t NUM_MATCHES_PER_VECTOR = sizeof(DictVec) / sizeof(DictEntry);

  // TODO: remove the __aarch64__ here with better Neon shuffle
#if GCC_COMPILER || defined(__aarch64__)
  // We need 4 values here, and they could be of uint8_t to save memory. However, this has a conversion cost before the
  // shuffle in GCC and Neon, so we use the larger uint32_t to avoid that runtime cost (~60% in one experiment!)
  using ShuffleIndexT = RowId;
#else
  // The entire lookup table fits into a single cache line (4B x 16 = 64B).
  using ShuffleIndexT = uint8_t;
#endif

  using ShuffleVec = simd::GccVec<ShuffleIndexT, 4 * sizeof(ShuffleIndexT)>::T;

  static_assert(NUM_MATCHES_PER_VECTOR == 4);
  alignas(64) static constexpr std::array<std::array<ShuffleIndexT, 4>, 16> MATCHES_TO_SHUFFLE_MASK =
      lookup_table_for_compressed_offsets_by_comparison_result<4, ShuffleIndexT, static_cast<ShuffleIndexT>(-1)>();

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);
    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      const auto rows_to_match = simd::load<DictVec>(rows + chunk_start_row);
      const DictVec matches = rows_to_match < filter_val;

      static_assert(NUM_MATCHES_PER_VECTOR == 4);
      constexpr RowVec ROW_OFFSETS{0, 1, 2, 3};
      const RowVec row_ids = chunk_start_row + ROW_OFFSETS;
      const uint8_t mask = simd::comparison_to_bitmask<DictVec, 4>(matches);
      assert(mask < 16 && "Mask cannot have more than 4 bits set.");

      const auto shuffle_mask = simd::load<ShuffleVec>(MATCHES_TO_SHUFFLE_MASK[mask].data());
      const RowVec compressed_rows = simd::shuffle_vector(row_ids, shuffle_mask);
      simd::store_unaligned(output + num_matching_rows, compressed_rows);
      num_matching_rows += std::popcount(mask);
    }

    return num_matching_rows;
  }
};

struct vector_128_scan_add {
  using VecT = simd::GccVec<DictEntry, 16>::T;
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(VecT) / sizeof(DictEntry);

  static_assert(NUM_MATCHES_PER_VECTOR == 4);
  alignas(16) static constexpr std::array<std::array<uint32_t, 4>, 16> MATCHES_TO_ROW_OFFSETS =
      lookup_table_for_compressed_offsets_by_comparison_result<4, uint32_t, 0>();

  // Haswell is a bit slower here because LLVM generates the comparison and move-to-mask inefficiently, see
  // https://godbolt.org/z/bzrxb57Kh
  // This does not occur on more modern architectures.
  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict column_data = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);

    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      const VecT table_values = simd::load<VecT>(column_data + chunk_start_row);
      const VecT compare_result = table_values < filter_val;
      const unsigned int packed_compare_result = simd::comparison_to_bitmask<VecT, 4>(compare_result);

      const VecT matching_row_offsets = simd::load<VecT>(MATCHES_TO_ROW_OFFSETS[packed_compare_result].data());
      const VecT compressed_matching_rows = chunk_start_row + matching_row_offsets;

      simd::store_unaligned(output + num_matching_rows, compressed_matching_rows);
      num_matching_rows += std::popcount(packed_compare_result);
    }
    return num_matching_rows;
  }
};

enum class Vector512ScanStrategy { SHUFFLE_MASK_16_BIT, SHUFFLE_MASK_8_BIT, SHUFFLE_MASK_4_BIT };

template <Vector512ScanStrategy STRATEGY>
struct vector_512_scan {
  using DictVec = simd::GccVec<uint32_t, 64>::T;
  using RowVec = simd::GccVec<uint32_t, 64>::T;

  static constexpr size_t NUM_MATCHES_PER_VECTOR = sizeof(DictVec) / sizeof(DictEntry);

  // TODO: remove the __aarch64__ here with better Neon shuffle
#if GCC_COMPILER || defined(__aarch64__)
  // We need 16 values here, and they could be of uint8_t to save memory. However, this has a conversion cost before the
  // shuffle, so we use the larger uint32_t to avoid that runtime cost.
  using ShuffleVecElementT = RowId;
#else
  using ShuffleVecElementT = uint8_t;
#endif
  using ShuffleMask16Elements = simd::GccVec<ShuffleVecElementT, 16 * sizeof(ShuffleVecElementT)>::T;
  static constexpr ShuffleVecElementT SDC = -1;  // SDC == SHUFFLE_DONT_CARE

  ShuffleMask16Elements get_shuffle_mask_from_16bit(uint16_t mask) {
    alignas(64) constexpr auto MATCHES_TO_SHUFFLE_MASK_16_BIT =
        lookup_table_for_compressed_offsets_by_comparison_result<16, ShuffleVecElementT, SDC>();

    return simd::load<ShuffleMask16Elements>(MATCHES_TO_SHUFFLE_MASK_16_BIT[mask].data());
  }

  alignas(64) static constexpr auto MATCHES_TO_SHUFFLE_MASK_8_BIT =
      lookup_table_for_compressed_offsets_by_comparison_result<8, ShuffleVecElementT, SDC>();
  ShuffleMask16Elements get_shuffle_mask_from_8bit(uint16_t mask) {
    const uint8_t lo_mask = mask;
    const uint8_t hi_mask = mask >> 8;
    const size_t num_lo_matches = std::popcount(lo_mask);

    const auto& lo_shuffle_mask = MATCHES_TO_SHUFFLE_MASK_8_BIT[lo_mask];
    const auto& hi_shuffle_mask = MATCHES_TO_SHUFFLE_MASK_8_BIT[hi_mask];

    alignas(64) std::array<ShuffleVecElementT, 16> combined_mask{0};
    std::memcpy(combined_mask.data(), lo_shuffle_mask.data(), sizeof(lo_shuffle_mask));

    using ShuffleMask8Elements = simd::GccVec<ShuffleVecElementT, 8 * sizeof(ShuffleVecElementT)>::T;
    auto upper_mask = simd::load<ShuffleMask8Elements>(hi_shuffle_mask.data());
    upper_mask += 8;
    simd::store_unaligned(combined_mask.data() + num_lo_matches, upper_mask);

    return simd::load<ShuffleMask16Elements>(combined_mask.data());
  }

  alignas(64) static constexpr auto MATCHES_TO_SHUFFLE_MASK_4_BIT =
      lookup_table_for_compressed_offsets_by_comparison_result<4, ShuffleVecElementT, SDC>();
  ShuffleMask16Elements get_shuffle_mask_from_4bit(uint16_t mask) {
    alignas(64) std::array<ShuffleVecElementT, 16> combined_mask{};
    size_t first_empty_slot = 0;

    using ShuffleMask4Elements = simd::GccVec<ShuffleVecElementT, 4 * sizeof(ShuffleVecElementT)>::T;

    for (uint8_t i = 0; i < 16; i += 4) {
      const uint8_t current_mask = (mask >> i) & 0b1111u;
      auto current_shuffle_mask = simd::load<ShuffleMask4Elements>(MATCHES_TO_SHUFFLE_MASK_4_BIT[current_mask].data());
      current_shuffle_mask += i;
      std::memcpy(combined_mask.data() + first_empty_slot, &current_shuffle_mask, sizeof(current_shuffle_mask));
      first_empty_slot += std::popcount(current_mask);
    }

    return simd::load<ShuffleMask16Elements>(combined_mask.data());
  }

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % (NUM_MATCHES_PER_VECTOR) == 0);
    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      static_assert(NUM_MATCHES_PER_VECTOR == 16);
      constexpr RowVec ROW_OFFSETS{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
      const RowVec row_ids = chunk_start_row + ROW_OFFSETS;

      const auto rows_to_match = simd::load<DictVec>(rows + chunk_start_row);
      const DictVec matches = rows_to_match < filter_val;

      const uint16_t mask = simd::comparison_to_bitmask(matches);

      // On haswell, this doesn't generate proper code. For the shuffle, llvm spills everything to memory 20 times
      // and then reloads from that: https://godbolt.org/z/b17zaofcq
      // Looks like something similar happens on ARM: https://godbolt.org/z/TGezGG8q1
      static_assert(NUM_MATCHES_PER_VECTOR == 16);
      ShuffleMask16Elements shuffle_mask;
      if constexpr (STRATEGY == Vector512ScanStrategy::SHUFFLE_MASK_16_BIT) {
        shuffle_mask = get_shuffle_mask_from_16bit(mask);
      } else if constexpr (STRATEGY == Vector512ScanStrategy::SHUFFLE_MASK_8_BIT) {
        shuffle_mask = get_shuffle_mask_from_8bit(mask);
      } else {
        static_assert(STRATEGY == Vector512ScanStrategy::SHUFFLE_MASK_4_BIT);
        shuffle_mask = get_shuffle_mask_from_4bit(mask);
      }

      const RowVec compressed_rows = simd::shuffle_vector(row_ids, shuffle_mask);
      simd::store_unaligned(output + num_matching_rows, compressed_rows);
      num_matching_rows += std::popcount(mask);
    }

    return num_matching_rows;
  }
};

#if defined(__aarch64__)
struct neon_scan {
  using DictVec = simd::NeonVecT<sizeof(DictEntry)>::T;
  using RowVec = simd::NeonVecT<sizeof(RowId)>::T;

  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(DictVec) / sizeof(DictEntry);
  static_assert(NUM_MATCHES_PER_VECTOR == 4);
  static constexpr std::array<std::array<uint8_t, 16>, 16> MATCHES_TO_SHUFFLE_MASK =
      element_shuffle_table_to_byte_shuffle_table<uint32_t, uint8_t, static_cast<uint8_t>(-1)>(
          lookup_table_for_compressed_offsets_by_comparison_result<4, uint8_t, static_cast<uint8_t>(-1)>());

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();
    const DictVec filter_vec = vmovq_n_u32(filter_val);

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);
    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      static_assert(NUM_MATCHES_PER_VECTOR == 4);
      constexpr RowVec ROW_OFFSETS = {0, 1, 2, 3};
      const RowVec row_ids = vmovq_n_u32(chunk_start_row) + ROW_OFFSETS;

      const DictVec rows_to_match = vld1q_u32(rows + chunk_start_row);
      const DictVec matches = vcltq_u32(rows_to_match, filter_vec);

      constexpr DictVec BIT_MASK = {1, 2, 4, 8};
      const uint8_t mask = vaddvq_u32(vandq_u32(matches, BIT_MASK));
      assert(mask >> 4 == 0 && "High 4 bits must be 0");

      const auto* shuffle_mask = reinterpret_cast<const uint8x16_t*>(MATCHES_TO_SHUFFLE_MASK[mask].data());
      const RowVec compressed_rows = vqtbl1q_u8(row_ids, *shuffle_mask);
      vst1q_u32(output + num_matching_rows, compressed_rows);
      num_matching_rows += std::popcount(mask);
    }

    return num_matching_rows;
  }
};
#endif

#if defined(__x86_64__)
struct x86_128_scan_manual_inner_loop {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m128i) / sizeof(DictEntry);

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict column_data = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);

    for (RowId row = 0; row < NUM_ROWS; row += NUM_MATCHES_PER_VECTOR) {
      const auto* table_values = reinterpret_cast<const __m128i*>(column_data + row);
      const __m128i compare_result = _mm_cmplt_epi32(*table_values, _mm_set1_epi32(static_cast<int>(filter_val)));
      unsigned int matches_bits = _mm_movemask_ps(reinterpret_cast<__m128>(compare_result));

      while (matches_bits != 0) {
        const int matching_element = std::countr_zero(matches_bits);
        output[num_matching_rows++] = row + matching_element;
        matches_bits &= matches_bits - 1;
      }
    }
    return num_matching_rows;
  }
};

struct x86_128_scan_predication {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m128i) / sizeof(DictEntry);

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict column_data = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);

    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      const auto* table_values = reinterpret_cast<const __m128i*>(column_data + chunk_start_row);
      const __m128i compare_result_vector =
          _mm_cmplt_epi32(*table_values, _mm_set1_epi32(static_cast<int>(filter_val)));

      alignas(16) std::array<uint32_t, 4> compare_result{};
      std::memcpy(compare_result.data(), &compare_result_vector, sizeof(compare_result));

      for (RowId row_offset = 0; row_offset < NUM_MATCHES_PER_VECTOR; ++row_offset) {
        output[num_matching_rows] = chunk_start_row + row_offset;
        num_matching_rows += compare_result[row_offset] & 1u;
      }
    }
    return num_matching_rows;
  }
};

struct x86_128_scan_pext {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m128i) / sizeof(DictEntry);

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict column_data = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);

    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      const auto* table_values = reinterpret_cast<const __m128i*>(column_data + chunk_start_row);
      const __m128i compare_result = _mm_cmplt_epi32(*table_values, _mm_set1_epi32(static_cast<int>(filter_val)));

      const __m128i row_indices =
          _mm_add_epi32(_mm_set1_epi32(static_cast<int>(chunk_start_row)), _mm_set_epi32(3, 2, 1, 0));

      const uint64_t lower_mask = _mm_extract_epi64(compare_result, 0);
      const uint64_t lower_indices = _mm_extract_epi64(row_indices, 0);
      const uint64_t lower_compressed_indices = _pext_u64(lower_indices, lower_mask);
      const uint64_t lower_match_bits = _mm_popcnt_u64(lower_mask);

      const uint64_t upper_mask = _mm_extract_epi64(compare_result, 1);
      const uint64_t upper_indices = _mm_extract_epi64(row_indices, 1);
      const uint64_t upper_compressed_indices = _pext_u64(upper_indices, upper_mask);
      const uint64_t upper_match_bits = _mm_popcnt_u64(upper_mask);

      std::memcpy(output + num_matching_rows, &lower_compressed_indices, sizeof(lower_compressed_indices));
      num_matching_rows += lower_match_bits / 32;

      std::memcpy(output + num_matching_rows, &upper_compressed_indices, sizeof(upper_compressed_indices));
      num_matching_rows += upper_match_bits / 32;
    }
    return num_matching_rows;
  }
};

struct x86_128_scan_shuffle {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m128i) / sizeof(DictEntry);
  static_assert(NUM_MATCHES_PER_VECTOR == 4);
  static constexpr std::array<std::array<uint8_t, 16>, 16> MATCHES_TO_SHUFFLE_MASK =
      element_shuffle_table_to_byte_shuffle_table<uint32_t, uint8_t, static_cast<uint8_t>(-1)>(
          lookup_table_for_compressed_offsets_by_comparison_result<4, uint8_t, static_cast<uint8_t>(-1)>());

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict column_data = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);

    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      const auto* table_values = reinterpret_cast<const __m128i*>(column_data + chunk_start_row);
      const __m128i compare_result = _mm_cmplt_epi32(*table_values, _mm_set1_epi32(static_cast<int>(filter_val)));
      const unsigned int packed_compare_result = _mm_movemask_ps(reinterpret_cast<const __m128&>(compare_result));

      const __m128i row_indices =
          _mm_add_epi32(_mm_set1_epi32(static_cast<int>(chunk_start_row)), _mm_set_epi32(3, 2, 1, 0));
      const auto* shuffle_mask =
          reinterpret_cast<const __m128i*>(MATCHES_TO_SHUFFLE_MASK.data() + packed_compare_result);
      const __m128i compressed_indices = _mm_shuffle_epi8(row_indices, *shuffle_mask);

      _mm_storeu_si128(reinterpret_cast<__m128i_u*>(output + num_matching_rows), compressed_indices);

      num_matching_rows += _mm_popcnt_u32(packed_compare_result);
    }
    return num_matching_rows;
  }
};

struct x86_128_scan_add {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m128i) / sizeof(DictEntry);

  static_assert(NUM_MATCHES_PER_VECTOR == 4);
  alignas(16) static constexpr std::array<std::array<uint32_t, 4>, 16> MATCHES_TO_ROW_OFFSETS =
      lookup_table_for_compressed_offsets_by_comparison_result<4, uint32_t, 0>();

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict column_data = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);

    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      const auto* table_values = reinterpret_cast<const __m128i*>(column_data + chunk_start_row);
      const __m128i compare_result = _mm_cmplt_epi32(*table_values, _mm_set1_epi32(static_cast<int>(filter_val)));
      const unsigned int packed_compare_result = _mm_movemask_ps(reinterpret_cast<const __m128&>(compare_result));

      const auto* matching_row_offsets =
          reinterpret_cast<const __m128i*>(MATCHES_TO_ROW_OFFSETS.data() + packed_compare_result);
      const __m128i compressed_matching_rows =
          _mm_set1_epi32(static_cast<int>(chunk_start_row)) + *matching_row_offsets;

      _mm_storeu_si128(reinterpret_cast<__m128i_u*>(output + num_matching_rows), compressed_matching_rows);

      num_matching_rows += _mm_popcnt_u32(packed_compare_result);
    }
    return num_matching_rows;
  }
};

struct x86_256_avx2_scan_shuffle {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m256i) / sizeof(DictEntry);

  // Because AVX2 can only shuffle within each of the two 128bit-lanes, we can reuse the old shuffle indices
  alignas(16) static constexpr std::array<std::array<uint8_t, 16>, 16> MATCHES_TO_SHUFFLE_MASK =
      x86_128_scan_shuffle::MATCHES_TO_SHUFFLE_MASK;

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict column_data = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);

    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      const auto* table_values = reinterpret_cast<const __m256i*>(column_data + chunk_start_row);
      const __m256i compare_result = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(filter_val)), *table_values);
      const __m256i row_indices = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int>(chunk_start_row)),
                                                   _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

      const unsigned int packed_compare_result = _mm256_movemask_ps(reinterpret_cast<const __m256&>(compare_result));
      const unsigned int packed_compare_result_low = packed_compare_result & 0b1111u;
      const unsigned int packed_compare_result_high = (packed_compare_result >> 4u) & 0b1111u;

      const auto* shuffle_mask_low =
          reinterpret_cast<const __m128i*>(MATCHES_TO_SHUFFLE_MASK[packed_compare_result_low].data());
      const auto* shuffle_mask_high =
          reinterpret_cast<const __m128i*>(MATCHES_TO_SHUFFLE_MASK[packed_compare_result_high].data());
      const __m256i shuffle_mask = _mm256_set_m128i(*shuffle_mask_high, *shuffle_mask_low);

      const __m256i compressed_indices = _mm256_shuffle_epi8(row_indices, shuffle_mask);

      const int match_count_low = _mm_popcnt_u32(packed_compare_result_low);
      RowId* chunk_output = (output + num_matching_rows);
      auto* low_chunk_output = reinterpret_cast<__m128i*>(chunk_output);
      auto* high_chunk_output = reinterpret_cast<__m128i*>(chunk_output + match_count_low);
      _mm256_storeu2_m128i(high_chunk_output, low_chunk_output, compressed_indices);

      num_matching_rows += _mm_popcnt_u32(packed_compare_result);
    }
    return num_matching_rows;
  }
};

struct x86_256_avx2_scan_add {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m256i) / sizeof(DictEntry);

  static_assert(NUM_MATCHES_PER_VECTOR == 8);
  alignas(16) static constexpr std::array<std::array<uint32_t, 4>, 16> MATCHES_TO_ROW_OFFSETS =
      lookup_table_for_compressed_offsets_by_comparison_result<4, uint32_t, 0>();

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict column_data = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);

    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      const auto* table_values = reinterpret_cast<const __m256i*>(column_data + chunk_start_row);
      const __m256i compare_result = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(filter_val)), *table_values);
      const unsigned int packed_compare_result = _mm256_movemask_ps(reinterpret_cast<const __m256&>(compare_result));
      const unsigned int packed_compare_result_low = packed_compare_result & 0b1111u;
      const unsigned int packed_compare_result_high = (packed_compare_result >> 4u) & 0b1111u;

      const int match_count_low = _mm_popcnt_u32(packed_compare_result_low);

      const auto* matching_row_offsets_low =
          reinterpret_cast<const __m128i*>(MATCHES_TO_ROW_OFFSETS[packed_compare_result_low].data());
      const auto* matching_row_offsets_high =
          reinterpret_cast<const __m128i*>(MATCHES_TO_ROW_OFFSETS[packed_compare_result_high].data());
      const __m256i matching_row_offsets = _mm256_set_m128i(*matching_row_offsets_high, *matching_row_offsets_low);

      const __m256i compressed_matching_rows = _mm256_set_m128i(_mm_set1_epi32(static_cast<int>(chunk_start_row + 4)),
                                                                _mm_set1_epi32(static_cast<int>(chunk_start_row))) +
                                               matching_row_offsets;

      RowId* chunk_output = (output + num_matching_rows);
      auto* low_chunk_output = reinterpret_cast<__m128i*>(chunk_output);
      auto* high_chunk_output = reinterpret_cast<__m128i*>(chunk_output + match_count_low);
      _mm256_storeu2_m128i(high_chunk_output, low_chunk_output, compressed_matching_rows);

      num_matching_rows += _mm_popcnt_u32(packed_compare_result);
    }
    return num_matching_rows;
  }
};
#endif

#if AVX512_AVAILABLE
enum class X86512ScanStrategy { COMPRESSSTORE, COMPRESS_PLUS_STORE };

template <X86512ScanStrategy STRATEGY>
struct x86_512_scan {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m512i) / sizeof(DictEntry);

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    const __m512i filter_vec = _mm512_set1_epi32(static_cast<int>(filter_val));
    const __m512i row_id_offsets = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);
    for (RowId chunk_start_row = 0; chunk_start_row < NUM_ROWS; chunk_start_row += NUM_MATCHES_PER_VECTOR) {
      // x86: Doing this instead of {start_row + 0, start_row + 1, ...} has a 3x performance improvement! Also applies
      // to the gcc-vec versions.
      const __m512i row_ids = _mm512_set1_epi32(static_cast<int>(chunk_start_row)) + row_id_offsets;

      const __m512i rows_to_match = _mm512_load_epi32(rows + chunk_start_row);
      const __mmask16 matches = _mm512_cmplt_epi32_mask(rows_to_match, filter_vec);

      if constexpr (STRATEGY == X86512ScanStrategy::COMPRESSSTORE) {
        _mm512_mask_compressstoreu_epi32(output + num_matching_rows, matches, row_ids);
      } else {
        static_assert(STRATEGY == X86512ScanStrategy::COMPRESS_PLUS_STORE);
        auto compressed_rows = _mm512_mask_compress_epi32(row_ids, matches, row_ids);
        _mm512_storeu_epi32(output + num_matching_rows, compressed_rows);
      }

      num_matching_rows += std::popcount(matches);
    }

    return num_matching_rows;
  }
};
#endif

template <typename ScanFn>
void BM_dictionary_scan(benchmark::State& state) {
  DictColumn column{NUM_ROWS};
  MatchingRows matching_rows{NUM_ROWS};

  static_assert(NUM_ROWS % NUM_UNIQUE_VALUES == 0, "Number of rows must be a multiple of num unique values.");
  const int64_t input_percentage = state.range(0);
  const auto percentage_to_pass_filter = static_cast<double>(input_percentage) / 100;

  // Our filter value comparison is `row < filter_value`, so we can control the selectivity as follows:
  //   For percentage =   0, the filter value is                     0, i.e., no values will match.
  //   For percentage =  50, the filter value is NUM_UNIQUE_VALUES / 2, i.e., 50% of all values will match.
  //   For percentage = 100, the filter value is     NUM_UNIQUE_VALUES, i.e., all values will match.
  const auto filter_value = static_cast<DictEntry>(NUM_UNIQUE_VALUES * percentage_to_pass_filter);

  DictEntry* column_data = column.aligned_data();
  for (size_t i = 0; i < NUM_ROWS; ++i) {
    column_data[i] = i % NUM_UNIQUE_VALUES;
  }
  std::mt19937 rng{std::random_device{}()};
  std::shuffle(column_data, column_data + NUM_ROWS, rng);

  // Correctness check with naive implementation
  ScanFn scan_fn{};
  MatchingRows matching_rows_naive{NUM_ROWS};
  const RowId num_matches_naive = naive_scalar_scan{}(column, filter_value, &matching_rows_naive);
  const RowId num_matches_specialized = scan_fn(column, filter_value, &matching_rows);

  if (num_matches_naive != num_matches_specialized) {
    throw std::runtime_error{"Bad result. Expected " + std::to_string(num_matches_naive) + " rows to match, but got " +
                             std::to_string(num_matches_specialized)};
  }
  for (size_t i = 0; i < num_matches_naive; ++i) {
    if (matching_rows_naive.aligned_data()[i] != matching_rows.aligned_data()[i]) {
      throw std::runtime_error{"Bad result compare at position: " + std::to_string(i)};
    }
  }

  // Sanity check that the 100 and 0 percent math works out.
  if (input_percentage == 100 && num_matches_specialized != NUM_ROWS) {
    throw std::runtime_error{"Bad result. Did not match all rows."};
  }
  if (input_percentage == 0 && num_matches_specialized != 0) {
    throw std::runtime_error{"Bad result. Did not match 0 rows."};
  }

  benchmark::DoNotOptimize(column.aligned_data());
  benchmark::DoNotOptimize(matching_rows.aligned_data());

  for (auto _ : state) {
    const RowId num_matches = scan_fn(column, filter_value, &matching_rows);
    benchmark::DoNotOptimize(num_matches);
  }

  state.counters["PerValue"] = benchmark::Counter(static_cast<double>(state.iterations() * NUM_ROWS),
                                                  benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

// #define BM_ARGS
// Unit(benchmark::kMicrosecond)->Arg(0)->Arg(10)->Arg(33)->Arg(50)->Arg(66)->Arg(100)->ReportAggregatesOnly()
#define BM_ARGS Unit(benchmark::kMicrosecond)->Arg(50)

BENCHMARK(BM_dictionary_scan<naive_scalar_scan>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<autovec_scalar_scan>)->BM_ARGS;

BENCHMARK(BM_dictionary_scan<vector_128_scan_shuffle>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<vector_128_scan_predication>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<vector_128_scan_add>)->BM_ARGS;

BENCHMARK(BM_dictionary_scan<vector_512_scan<Vector512ScanStrategy::SHUFFLE_MASK_16_BIT>>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<vector_512_scan<Vector512ScanStrategy::SHUFFLE_MASK_8_BIT>>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<vector_512_scan<Vector512ScanStrategy::SHUFFLE_MASK_4_BIT>>)->BM_ARGS;

#if defined(__aarch64__)
BENCHMARK(BM_dictionary_scan<neon_scan>)->BM_ARGS;
#endif

#if defined(__x86_64__)
BENCHMARK(BM_dictionary_scan<x86_128_scan_manual_inner_loop>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<x86_128_scan_predication>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<x86_128_scan_pext>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<x86_128_scan_shuffle>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<x86_128_scan_add>)->BM_ARGS;

BENCHMARK(BM_dictionary_scan<x86_256_avx2_scan_shuffle>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<x86_256_avx2_scan_add>)->BM_ARGS;
#endif

#if AVX512_AVAILABLE
BENCHMARK(BM_dictionary_scan<x86_512_scan<X86512ScanStrategy::COMPRESSSTORE>>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<x86_512_scan<X86512ScanStrategy::COMPRESS_PLUS_STORE>>)->BM_ARGS;
#endif

BENCHMARK_MAIN();
