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
static constexpr size_t SCALE_FACTOR = 1024 * 64;
static constexpr size_t NUM_ROWS = NUM_BASE_ROWS * SCALE_FACTOR;
static constexpr size_t NUM_UNIQUE_VALUES = 16;

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
    // According to the issue, ICX can autovectorize this.
    // Godbolt playground: https://godbolt.org/z/aahTPczdr

    const DictEntry* __restrict column_data = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    RowId num_matching_rows = 0;
    for (RowId row = 0; row < NUM_ROWS; ++row) {
      output[num_matching_rows] = row;
      num_matching_rows += column_data[row] < filter_val;
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
    const auto filter_vec = simd::broadcast<DictVec>(filter_val);

    static_assert(NUM_ROWS % (NUM_MATCHES_PER_VECTOR) == 0);
    for (RowId i = 0; i < NUM_ROWS; i += NUM_MATCHES_PER_VECTOR) {
      auto rows_to_match = simd::load<DictVec>(rows + i);
      DictVec matches = rows_to_match < filter_vec;

      for (RowId row = 0; row < NUM_MATCHES_PER_VECTOR; ++row) {
        output[num_matching_rows] = i + row;
        num_matching_rows += matches[row] & 1;
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
  using ShuffleVec = RowVec;
  static constexpr RowId SDC = -1;  // SDC == SHUFFLE_DONT_CARE
#else
  // The entire lookup table fits into a single cache line (4B x 16 = 64B).
  using ShuffleVec = simd::GccVec<uint8_t, NUM_MATCHES_PER_VECTOR>::T;
  static constexpr uint8_t SDC = -1;  // SDC == SHUFFLE_DONT_CARE
#endif

  static_assert(NUM_MATCHES_PER_VECTOR == 4);
  alignas(64) static constexpr std::array MATCHES_TO_SHUFFLE_MASK = {
      ShuffleVec{/*0000*/ SDC, SDC, SDC, SDC}, ShuffleVec{/*0001*/ 0, SDC, SDC, SDC},
      ShuffleVec{/*0010*/ 1, SDC, SDC, SDC},   ShuffleVec{/*0011*/ 0, 1, SDC, SDC},
      ShuffleVec{/*0100*/ 2, SDC, SDC, SDC},   ShuffleVec{/*0101*/ 0, 2, SDC, SDC},
      ShuffleVec{/*0110*/ 1, 2, SDC, SDC},     ShuffleVec{/*0111*/ 0, 1, 2, SDC},
      ShuffleVec{/*1000*/ 3, SDC, SDC, SDC},   ShuffleVec{/*1001*/ 0, 3, SDC, SDC},
      ShuffleVec{/*1010*/ 1, 3, SDC, SDC},     ShuffleVec{/*1011*/ 0, 1, 3, SDC},
      ShuffleVec{/*1100*/ 2, 3, SDC, SDC},     ShuffleVec{/*1101*/ 0, 2, 3, SDC},
      ShuffleVec{/*1110*/ 1, 2, 3, SDC},       ShuffleVec{/*1111*/ 0, 1, 2, 3}};

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();
    const auto filter_vec = simd::broadcast<DictVec>(filter_val);

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);
    for (RowId i = 0; i < NUM_ROWS; i += NUM_MATCHES_PER_VECTOR) {
      auto rows_to_match = simd::load<DictVec>(rows + i);
      DictVec matches = rows_to_match < filter_vec;

      static_assert(NUM_MATCHES_PER_VECTOR == 4);
      constexpr RowVec row_offsets{0, 1, 2, 3};
      RowVec row_ids = simd::broadcast<RowVec>(i) + row_offsets;
      uint8_t mask = simd::comparison_to_bitmask<DictVec, 4>(matches);
      assert(mask < 16 && "Mask cannot have more than 4 bits set.");

      ShuffleVec shuffle_mask = MATCHES_TO_SHUFFLE_MASK[mask];
      RowVec compressed_rows = simd::shuffle_vector(row_ids, shuffle_mask);
      simd::store_unaligned(output + num_matching_rows, compressed_rows);
      num_matching_rows += std::popcount(mask);
    }

    return num_matching_rows;
  }
};

// TODO: This is currently a tiny bit faster than the version below with 64k masks. Leaving both in the code for now, so
//       we can clean this up in two "shuffle strategies".
enum class vector_512_scan_strategy { SHUFFLE_MASK_16_BIT, SHUFFLE_MASK_8_BIT, SHUFFLE_MASK_4_BIT };

template <vector_512_scan_strategy STRATEGY>
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

  template <size_t ComparisonResultBytes, typename ShuffleIndexT>
  static constexpr auto lookup_table_for_shuffle_mask_by_comparison_result() {
    std::array<std::array<ShuffleIndexT, ComparisonResultBytes * 8>, 1 << (ComparisonResultBytes * 8)> lookup_table;

    for (size_t index = 0; index < lookup_table.size(); ++index) {
      auto& shuffle_mask = lookup_table[index];
      std::fill(shuffle_mask.begin(), shuffle_mask.end(), SDC);

      size_t first_empty_output_slot = 0;
      for (size_t comparison_result_rest = index; comparison_result_rest != 0;
           comparison_result_rest &= comparison_result_rest - 1) {
        shuffle_mask[first_empty_output_slot++] = std::countr_zero(comparison_result_rest);
      }
    }

    return lookup_table;
  }

  ShuffleMask16Elements get_shuffle_mask_from_16bit(uint16_t mask) {
    // TODO: currently not constexpr because we hit compiler constant evaluation limits
    // Check if increasing the limits and making it constexpr gives any benefit.
    alignas(64) static auto MATCHES_TO_SHUFFLE_MASK_16_BIT =
        lookup_table_for_shuffle_mask_by_comparison_result<2, ShuffleVecElementT>();

    return simd::load<ShuffleMask16Elements>(MATCHES_TO_SHUFFLE_MASK_16_BIT[mask].data());
  }

  alignas(64) static constexpr auto MATCHES_TO_SHUFFLE_MASK_8_BIT =
      lookup_table_for_shuffle_mask_by_comparison_result<1, ShuffleVecElementT>();
  ShuffleMask16Elements get_shuffle_mask_from_8bit(uint16_t mask) {
    uint8_t lo_mask = mask;
    uint8_t hi_mask = mask >> 8;
    size_t num_lo_matches = std::popcount(lo_mask);

    auto& lo_shuffle_mask = MATCHES_TO_SHUFFLE_MASK_8_BIT[lo_mask];
    auto& hi_shuffle_mask = MATCHES_TO_SHUFFLE_MASK_8_BIT[hi_mask];

    alignas(64) std::array<ShuffleVecElementT, 16> combined_mask{0};
    std::memcpy(combined_mask.data(), lo_shuffle_mask.data(), lo_shuffle_mask.size());

    using ShuffleMask8Elements = simd::GccVec<ShuffleVecElementT, 8 * sizeof(ShuffleVecElementT)>::T;
    auto upper_mask = simd::load<ShuffleMask8Elements>(hi_shuffle_mask.data());
    upper_mask += 8;
    simd::store_unaligned(combined_mask.data() + num_lo_matches, upper_mask);

    return simd::load<ShuffleMask16Elements>(combined_mask.data());
  }

  ShuffleMask16Elements get_shuffle_mask_from_4bit(uint16_t mask) {
    using VecScan = vector_128_scan_shuffle;
    using ShuffleVec4 = VecScan::ShuffleVec;

    alignas(64) std::array<ShuffleVecElementT, 16> combined_mask{};
    size_t first_empty_slot = 0;

    for (uint8_t i = 0; i < 16; i += 4) {
      uint8_t current_mask = (mask >> i) & 0xF;
      ShuffleVec4 current_shuffle_mask = VecScan::MATCHES_TO_SHUFFLE_MASK[current_mask] + i;
      std::memcpy(combined_mask.data() + first_empty_slot, &current_shuffle_mask, sizeof(current_shuffle_mask));
      first_empty_slot += std::popcount(current_mask);
    }

    return simd::load<ShuffleMask16Elements>(combined_mask.data());
  }

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();
    const auto filter_vec = simd::broadcast<DictVec>(filter_val);

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % (NUM_MATCHES_PER_VECTOR) == 0);
    for (RowId i = 0; i < NUM_ROWS; i += NUM_MATCHES_PER_VECTOR) {
      static_assert(NUM_MATCHES_PER_VECTOR == 16);
      constexpr RowVec row_offsets{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
      RowVec row_ids = simd::broadcast<RowVec>(i) + row_offsets;

      auto rows_to_match = simd::load<DictVec>(rows + i);
      DictVec matches = rows_to_match < filter_vec;

      uint16_t mask = simd::comparison_to_bitmask(matches);

      static_assert(NUM_MATCHES_PER_VECTOR == 16);
      ShuffleMask16Elements shuffle_mask;
      if constexpr (STRATEGY == vector_512_scan_strategy::SHUFFLE_MASK_16_BIT) {
        shuffle_mask = get_shuffle_mask_from_16bit(mask);
      } else if constexpr (STRATEGY == vector_512_scan_strategy::SHUFFLE_MASK_8_BIT) {
        shuffle_mask = get_shuffle_mask_from_8bit(mask);
      } else {
        static_assert(STRATEGY == vector_512_scan_strategy::SHUFFLE_MASK_4_BIT);
        shuffle_mask = get_shuffle_mask_from_4bit(mask);
      }

      RowVec compressed_rows = simd::shuffle_vector(row_ids, shuffle_mask);
      simd::store_unaligned(output + num_matching_rows, compressed_rows);
      num_matching_rows += std::popcount(mask);
    }

    return num_matching_rows;
  }
};

#if defined(__aarch64__)
struct neon_scan {
  using DictVec = simd::NeonVecT<sizeof(DictEntry)>;
  using RowVec = simd::NeonVecT<sizeof(RowId)>;

  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(DictVec) / sizeof(DictEntry);

  // We only need 4x 32 Bit here, but NEON does not provide a VTBL/VTBX instruction for this, so we need to shuffle on a
  // Byte level.
  static_assert(NUM_MATCHES_PER_VECTOR == 4);
  using TableVec = uint8x16_t;
  static constexpr uint8_t SDC = -1;  // SDC == SHUFFLE_DONT_CARE
  // clang-format off
  static constexpr std::array<TableVec, 16> MATCHES_TO_SHUFFLE_MASK = {
      TableVec{/*0000*/ /*0:*/ SDC, SDC, SDC, SDC, /*1:*/ SDC, SDC, SDC, SDC, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*0001*/ /*0:*/ 0, 1, 2, 3, /*1:*/ SDC, SDC, SDC, SDC, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*0010*/ /*0:*/ 4, 5, 6, 7, /*1:*/ SDC, SDC, SDC, SDC, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*0011*/ /*0:*/ 0, 1, 2, 3, /*1:*/ 4, 5, 6, 7, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*0100*/ /*0:*/ 8, 9, 10, 11, /*1:*/ SDC, SDC, SDC, SDC, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*0101*/ /*0:*/ 0, 1, 2, 3, /*1:*/ 8, 9, 10, 11, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*0110*/ /*0:*/ 4, 5, 6, 7, /*1:*/ 8, 9, 10, 11, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*0111*/ /*0:*/ 0, 1, 2, 3, /*1:*/ 4, 5, 6, 7, /*2:*/ 8, 9, 10, 11, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*1000*/ /*0:*/ 12, 13, 14, 15, /*1:*/ SDC, SDC, SDC, SDC, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*1001*/ /*0:*/ 0, 1, 2, 3, /*1:*/ 12, 13, 14, 15, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*1010*/ /*0:*/ 4, 5, 6, 7, /*1:*/ 12, 13, 14, 15, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*1011*/ /*0:*/ 0, 1, 2, 3, /*1:*/ 4, 5, 6, 7, /*2:*/ 12, 13, 14, 15, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*1100*/ /*0:*/ 8, 9, 10, 11, /*1:*/ 12, 13, 14, 15, /*2:*/ SDC, SDC, SDC, SDC, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*1101*/ /*0:*/ 0, 1, 2, 3, /*1:*/ 8, 9, 10, 11, /*2:*/ 12, 13, 14, 15, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*1110*/ /*0:*/ 4, 5, 6, 7, /*1:*/ 8, 9, 10, 11, /*2:*/ 12, 13, 14, 15, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*1110*/ /*0:*/ 0, 1, 2, 3, /*1:*/ 4, 5, 6, 7, /*2:*/ 8, 9, 10, 11, /*3:*/ 12, 13, 14, 15},
  };
  // clang-format on

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();
    const DictVec filter_vec = vmovq_n_u32(filter_val);

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);
    for (RowId i = 0; i < NUM_ROWS; i += NUM_MATCHES_PER_VECTOR) {
      static_assert(NUM_MATCHES_PER_VECTOR == 4);
      constexpr RowVec row_offsets = {0, 1, 2, 3};
      RowVec row_ids = vmovq_n_u32(start_row) + row_offsets;

      DictVec rows_to_match = vld1q_u32(rows + start_row);
      DictVec matches = vcltq_u32(rows_to_match, filter_vec);

      // TODO: if constexpr shuffle strategy

      constexpr DictVec bit_mask = {1, 2, 4, 8};
      uint8_t mask = vaddvq_u32(vandq_u32(matches, bit_mask));
      assert(mask >> 4 == 0 && "High 4 bits must be 0");

      TableVec shuffle_mask = MATCHES_TO_SHUFFLE_MASK[mask];
      // TODO: check if we can do this differently with: vqtbx1q_u8
      RowVec compressed_rows = vqtbl1q_u8(row_ids, shuffle_mask);
      vst1q_u32(output + num_matching_rows, compressed_rows);
      num_matching_rows += std::popcount(mask);
    }

    return num_matching_rows;
  }
};
#endif

#if defined(__x86_64__)

// Important: _mm*_mask_compressstore[u]() is not available until AVX512. So anything older must run without it.
struct x86_128_scan {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m128i) / sizeof(DictEntry);

  // TODO
  // Approaches from https://stackoverflow.com/a/41958528/12345551
  //  * Movemask + manual inner countr_zero loop
  //  * PEXT to extract the bytes where the comparison matched.
  //
  //  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
  //  }
};

#endif

#if AVX512_AVAILABLE

enum class x86_512_scan_strategy { COMPRESSSTORE, COMPRESS_PLUS_STORE };

template <x86_512_scan_strategy STRATEGY>
struct x86_512_scan {
  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(__m512i) / sizeof(DictEntry);

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();

    const __m512i filter_vec = _mm512_set1_epi32(filter_val);
    const __m512i row_id_offsets = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    RowId num_matching_rows = 0;
    static_assert(NUM_ROWS % NUM_MATCHES_PER_VECTOR == 0);
    for (RowId i = 0; i < NUM_ROWS; i += NUM_MATCHES_PER_VECTOR) {
      // x86: Doing this instead of {start_row + 0, start_row + 1, ...} has a 3x performance improvement! Also applies
      // to the gcc-vec versions.
      const __m512i row_ids = _mm512_set1_epi32(i) + row_id_offsets;

      __m512i rows_to_match = _mm512_load_epi32(rows + i);
      __mmask16 matches = _mm512_cmplt_epi32_mask(rows_to_match, filter_vec);

      if constexpr (STRATEGY == x86_512_scan_strategy::COMPRESSSTORE) {
        _mm512_mask_compressstoreu_epi32(output + num_matching_rows, matches, row_ids);
      } else if (STRATEGY == x86_512_scan_strategy::COMPRESS_PLUS_STORE) {
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
  // Seed rng for same benchmark runs.
  std::mt19937 rng{38932239};
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
}

// #define BM_ARGS
// Unit(benchmark::kMicrosecond)->Arg(0)->Arg(10)->Arg(33)->Arg(50)->Arg(66)->Arg(100)->ReportAggregatesOnly()
#define BM_ARGS Unit(benchmark::kMicrosecond)->Arg(50)

BENCHMARK(BM_dictionary_scan<naive_scalar_scan>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<autovec_scalar_scan>)->BM_ARGS;

BENCHMARK(BM_dictionary_scan<vector_128_scan_shuffle>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<vector_128_scan_predication>)->BM_ARGS;

BENCHMARK(BM_dictionary_scan<vector_512_scan<vector_512_scan_strategy::SHUFFLE_MASK_16_BIT>>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<vector_512_scan<vector_512_scan_strategy::SHUFFLE_MASK_8_BIT>>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<vector_512_scan<vector_512_scan_strategy::SHUFFLE_MASK_4_BIT>>)->BM_ARGS;

#if defined(__aarch64__)
BENCHMARK(BM_dictionary_scan<neon_scan>)->BM_ARGS;
#endif

#if defined(__x86_64__)
// BENCHMARK(BM_dictionary_scan<x86_128_scan>)->BM_ARGS;
#endif

#if AVX512_AVAILABLE
BENCHMARK(BM_dictionary_scan<x86_512_scan<x86_512_scan_strategy::COMPRESSSTORE>>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<x86_512_scan<x86_512_scan_strategy::COMPRESS_PLUS_STORE>>)->BM_ARGS;
#endif

BENCHMARK_MAIN();
