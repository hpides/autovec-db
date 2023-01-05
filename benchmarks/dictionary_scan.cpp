//
//
///**
// * use 2^8 or 2^16 element array for shuffle mask based on comparison mask
// * TODO: how to get mask from comparison vector with 1 byte per result
// */
//
//// RANDOM DUMP FROM godbolt: https://godbolt.org/z/acTbK3M7o
//
// using Vec64x8 __attribute__((vector_size(64))) = uint64_t;
// using Vec32x16 __attribute__((vector_size(64))) = uint32_t;
//
// using Vec8x8 __attribute__((vector_size(8))) = uint8_t;
// using Vec8x16 __attribute__((vector_size(16))) = uint8_t;
//
// using Mask8 __attribute__((ext_vector_type(8))) = bool;
// using Mask16 __attribute__((ext_vector_type(16))) = bool;
//
// extern std::array<Vec8x8, 256> shuffle_mask_8_entries;
//
// Vec64x8 shuffle_8(Vec64x8 dict_keys, uint64_t val) {
//  auto matches = dict_keys == val;
//  auto mask = __builtin_convertvector(matches, Mask8);
//  uint8_t mask_primitive;
//  std::memcpy(&mask_primitive, &mask, sizeof(mask_primitive));
//  auto shuffle_mask = shuffle_mask_8_entries[mask_primitive];
//  return __builtin_shufflevector(dict_keys, shuffle_mask);
//}
//
// Vec32x16 only_shuffle_16(Vec32x16 values, uint16_t mask_primitive) {
//  uint8_t lo_mask = mask_primitive;
//  uint8_t hi_mask = mask_primitive >> 8;
//
//  size_t num_lo_matches = std::popcount(lo_mask);
//
//  auto lo_shuffle_mask = shuffle_mask_8_entries[lo_mask];
//  auto hi_shuffle_mask = shuffle_mask_8_entries[hi_mask] + 8;
//
//  alignas(16) std::array<uint8_t, 16> raw_shuffle_mask{};
//  raw_shuffle_mask.fill(-1);  // Fill with -1 for indices we don't care about.
//  std::memcpy(raw_shuffle_mask.data(), &lo_shuffle_mask, sizeof(lo_shuffle_mask));
//  std::memcpy(raw_shuffle_mask.data() + num_lo_matches, &hi_shuffle_mask, sizeof(hi_shuffle_mask));
//
//  Vec8x16 shuffle_mask;
//  std::memcpy(&shuffle_mask, raw_shuffle_mask.data(), sizeof(shuffle_mask));
//  return __builtin_shufflevector(values, shuffle_mask);
//}
//
// void compressed_write(Vec32x16 values, uint32_t val, uint32_t* out) {
//  auto matches = values == val;
//  auto mask = __builtin_convertvector(matches, Mask16);
//  uint16_t mask_primitive = reinterpret_cast<uint16_t&>(mask);
//  auto shuffled_values = only_shuffle_16(values, mask_primitive);
//  *reinterpret_cast<Vec32x16*>(out) = shuffled_values;
//  out += std::popcount(mask_primitive);
//}
//
// Vec32x16 shuffle_16(Vec32x16 dict_keys, uint32_t val) {
//  auto matches = dict_keys == val;
//  auto mask = __builtin_convertvector(matches, Mask16);
//  uint16_t mask_primitive = reinterpret_cast<uint16_t&>(mask);
//
//  uint8_t lo_mask = mask_primitive;
//  uint8_t hi_mask = mask_primitive >> 8;
//
//  size_t num_lo_matches = std::popcount(lo_mask);
//  size_t num_hi_matches = std::popcount(hi_mask);
//
//  auto lo_shuffle_mask = shuffle_mask_8_entries[lo_mask];
//  auto hi_shuffle_mask = shuffle_mask_8_entries[hi_mask] + 8;
//
//  alignas(16) std::array<uint8_t, 16> raw_shuffle_mask{};
//  std::memcpy(raw_shuffle_mask.data(), &lo_shuffle_mask, sizeof(lo_shuffle_mask));
//  std::memcpy(raw_shuffle_mask.data() + num_lo_matches, &hi_shuffle_mask, sizeof(hi_shuffle_mask));
//
//  Vec8x16 shuffle_mask;
//  std::memcpy(&shuffle_mask, raw_shuffle_mask.data(), sizeof(shuffle_mask));
//  return __builtin_shufflevector(dict_keys, reinterpret_cast<Vec8x16&>(shuffle_mask));
//}
//
// Vec32x16 shuffle_16_old(Vec32x16 dict_keys, uint32_t val) {
//  auto matches = dict_keys == val;
//  auto mask = __builtin_convertvector(matches, Mask16);
//  uint16_t mask_primitive = reinterpret_cast<uint16_t&>(mask);
//
//  // Vec8x16 shuffle_mask = {255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
//  std::array<uint8_t, 16> shuffle_mask = {255, 255, 255, 255, 255, 255, 255, 255,
//                                          255, 255, 255, 255, 255, 255, 255, 255};
//  uint8_t current_idx = 0;
//
//  for (uint8_t i = 0; i < 16; ++i) {
//    shuffle_mask[current_idx] = i;
//    current_idx += mask[i];
//  }
//
//  return __builtin_shufflevector(dict_keys, reinterpret_cast<Vec8x16&>(shuffle_mask));
//
//  // TODO: check if these are correct
//  // return __builtin_shufflevector(dict_keys, mask);
//  // return __builtin_shufflevector(dict_keys, matches); // This is most likely wrong
//}
//
//// This looks a lot better in godbolt: https://godbolt.org/z/e89eE31hG
//// This only needs to work with GCC. Clang has the direct mask available.
//// This still looks bad on ARM :/
//// TODO: check that it is correct
// uint16_t get_mask_and_add(Vec32x16 matches) {
//   constexpr Vec32x16 and_mask = {// Order does not seem to make a difference here
//                                  // 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
//                                  32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1};
//
//   auto single_bits = matches & and_mask;
//   alignas(64) std::array<uint32_t, 16> single_bit_array;
//   std::memcpy(&single_bit_array, &single_bits, sizeof(single_bit_array));
//   return std::accumulate(single_bit_array.begin(), single_bit_array.end(), 0);
// }

/////////////////////////////
/////////////////////////////
#include <array>
#include <bit>
#include <cstdint>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"

namespace {

template <typename VecT, size_t NUM_ELEMENTS = sizeof(VecT) / sizeof(decltype(VecT{}[0]))>
uint8_t TEMPORARY_get_mask(VecT input) {
  static_assert(NUM_ELEMENTS <= 8);
  uint8_t result = 0;
  for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
    if (input[i]) {
      result |= 1ull << i;
    }
  }
  return result;
}

}  // namespace

using RowId = uint32_t;
using DictEntry = uint32_t;
static_assert(sizeof(RowId) == sizeof(DictEntry), "If we change these sizes, writing results must be updated.");

using DictColumn = AlignedData<DictEntry, 64>;
using MatchingRows = AlignedData<RowId, 64>;

// TODO: check this comment
// Must be a multiple of 16 for 512 Bit processing.
static constexpr size_t NUM_ROWS = 16;
static constexpr size_t NUM_UNIQUE_VALUES = 8;

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

struct autovec_scalar_find {
  // TODO
};

struct vector_128_scan {
  static_assert(sizeof(DictEntry) == 4, "Need to change DictVec below.");
  using DictVec = GccVec<uint32_t, 16>::T;

  static_assert(sizeof(RowId) == 4, "Need to change RowVec below.");
  using RowVec = GccVec<uint32_t, 16>::T;
  using UnalignedRowVec __attribute__((aligned(1))) = RowVec;

  static constexpr size_t NUM_MATCHES_PER_VECTOR = sizeof(DictVec) / sizeof(DictEntry);

  // We need 4 values here, but we don't need all 32 Bits, so we chose uint8_t.
  using ShuffleVec = GccVec<uint8_t, 4>::T;
  static constexpr uint8_t SDC = -1;  // SDC == SHUFFLE_DONT_CARE
  static constexpr std::array<ShuffleVec, 16> MATCHES_TO_SHUFFLE_MASK = {
      ShuffleVec{/*0000*/ SDC, SDC, SDC, SDC}, ShuffleVec{/*0001*/ 0, SDC, SDC, SDC},
      ShuffleVec{/*0010*/ 1, SDC, SDC, SDC},   ShuffleVec{/*0011*/ 0, 1, SDC, SDC},
      ShuffleVec{/*0100*/ 2, SDC, SDC, SDC},   ShuffleVec{/*0101*/ 0, 2, SDC, SDC},
      ShuffleVec{/*0110*/ 1, 2, SDC, SDC},     ShuffleVec{/*0111*/ 0, 1, 2, SDC},
      ShuffleVec{/*1000*/ 3, SDC, SDC, SDC},   ShuffleVec{/*1001*/ 0, 3, SDC, SDC},
      ShuffleVec{/*1010*/ 1, 3, SDC, SDC},     ShuffleVec{/*1011*/ 0, 1, 3, SDC},
      ShuffleVec{/*1100*/ 2, 3, SDC, SDC},     ShuffleVec{/*1101*/ 0, 2, 3, SDC},
      ShuffleVec{/*1110*/ 1, 2, 3, SDC},       ShuffleVec{/*1111*/ 0, 1, 2, 3}};

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    RowId num_matching_rows = 0;

    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();
    const auto filter_vec = simd::broadcast<DictVec>(filter_val);

    static_assert(NUM_ROWS % (NUM_MATCHES_PER_VECTOR) == 0);
    constexpr size_t iterations = NUM_ROWS / NUM_MATCHES_PER_VECTOR;
    for (RowId i = 0; i < iterations; ++i) {
      auto rows_to_match = simd::load<DictVec>(rows + (i * NUM_MATCHES_PER_VECTOR));
      DictVec matches = rows_to_match < filter_vec;
      RowVec row_ids = {i + 0, i + 1, i + 2, i + 3};

      // TODO: if constexpr shuffle strategy
      uint8_t mask = TEMPORARY_get_mask(matches);  // TODO: update with real method/strategy here
      ShuffleVec shuffle_mask = MATCHES_TO_SHUFFLE_MASK[mask];
      RowVec compressed_rows = simd::shuffle_vector(row_ids, shuffle_mask);
      simd::store(output + num_matching_rows, compressed_rows);
      num_matching_rows += std::popcount(mask);
    }

    return num_matching_rows;
  }
};

#if defined(__aarch64__)
#include <arm_neon.h>

struct neon_scan {
  static_assert(sizeof(DictEntry) == 4, "Need to change DictVec below.");
  using DictVec = uint32x4_t;

  static_assert(sizeof(RowId) == 4, "Need to change RowVec below.");
  using RowVec = uint32x4_t;

  static constexpr uint32_t NUM_MATCHES_PER_VECTOR = sizeof(DictVec) / sizeof(DictEntry);

  // We only need 4x 32 Bit here, but NEON does not provide a VTBL/VTBX instruction for this, so we need to shuffle on a
  // Byte level.
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
      TableVec{/*1101*/ /*0:*/ 0, 1, 2, 3, /*1:*/ 12, 13, 14, 15, /*2:*/ 12, 13, 14, 15, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*1110*/ /*0:*/ 4, 5, 6, 7, /*1:*/ 8, 9, 10, 11, /*2:*/ 12, 13, 14, 15, /*3:*/ SDC, SDC, SDC, SDC},
      TableVec{/*1110*/ /*0:*/ 0, 1, 2, 3, /*1:*/ 4, 5, 6, 7, /*2:*/ 8, 9, 10, 11, /*3:*/ 12, 13, 14, 15},
  };
  // clang-format on

  RowId operator()(const DictColumn& column, DictEntry filter_val, MatchingRows* matching_rows) {
    RowId num_matching_rows = 0;

    const DictEntry* __restrict rows = column.aligned_data();
    RowId* __restrict output = matching_rows->aligned_data();
    const DictVec filter_vec = vmovq_n_u32(filter_val);

    static_assert(NUM_ROWS % (NUM_MATCHES_PER_VECTOR) == 0);
    constexpr size_t iterations = NUM_ROWS / NUM_MATCHES_PER_VECTOR;
    for (RowId i = 0; i < iterations; ++i) {
      DictVec rows_to_match = vld1q_u32(rows + (i * NUM_MATCHES_PER_VECTOR));
      DictVec matches = vcltq_u32(rows_to_match, filter_vec);

      const RowId start_row = NUM_MATCHES_PER_VECTOR * i;
      RowVec row_ids = {start_row + 0, start_row + 1, start_row + 2, start_row + 3};

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
struct x86_find {
  // TODO
};
#endif

template <typename ScanFn>
void BM_dictionary_scan(benchmark::State& state) {
  // Seed rng for same benchmark runs.
  std::mt19937 rng{38932239};

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
    benchmark::DoNotOptimize(matching_rows.aligned_data());
  }
}

// #define BM_ARGS Arg(0)->Arg(33)->Arg(50)->Arg(66)->Arg(100)
#define BM_ARGS Arg(50)

BENCHMARK(BM_dictionary_scan<naive_scalar_scan>)->BM_ARGS;

#if defined(__aarch64__)
BENCHMARK(BM_dictionary_scan<neon_scan>)->BM_ARGS;
#endif

#if defined(__x86_64__)
BENCHMARK(BM_dictionary_scan<x86_128_scan>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<x86_512_scan>)->BM_ARGS;
#endif

BENCHMARK_MAIN();
