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
// uint16_t get_mask(Vec32x16 matches) {
//  // TODO: check with GCC
//  std::bitset<16> bitset{};
//  for (size_t i = 0; i < 16; ++i) {
//    bitset[i] = matches[i];
//  }
//  return bitset.to_ulong();
//  // return
//  //     ((matches[0] & 1) << 0) ||
//  //     ((matches[1] & 1) << 1) ||
//  //     ((matches[2] & 1) << 2) ||
//  //     ((matches[3] & 1) << 3) ||
//  //     ((matches[4] & 1) << 4) ||
//  //     ((matches[5] & 1) << 5) ||
//  //     ((matches[6] & 1) << 6) ||
//  //     ((matches[7] & 1) << 7) ||
//  //     ((matches[8] & 1) << 8) ||
//  //     ((matches[9] & 1) << 9) ||
//  //     ((matches[10] & 1) << 10) ||
//  //     ((matches[11] & 1) << 11) ||
//  //     ((matches[12] & 1) << 12) ||
//  //     ((matches[13] & 1) << 13) ||
//  //     ((matches[14] & 1) << 14) ||
//  //     ((matches[15] & 1) << 15);
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

using RowId = uint32_t;
using DictEntry = uint32_t;

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

struct vector_find {
  // TODO
};

#if defined(__aarch64__)
#include <arm_neon.h>

struct neon_find {
  // TODO
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

#define BM_ARGS Arg(0)->Arg(33)->Arg(50)->Arg(66)->Arg(100)

BENCHMARK(BM_dictionary_scan<naive_scalar_scan>)->BM_ARGS;

#if defined(__aarch64__)
// BENCHMARK(BM_dictionary_scan<neon_scan>)->BM_ARGS;
#endif

#if defined(__x86_64__)
BENCHMARK(BM_dictionary_scan<x86_128_scan>)->BM_ARGS;
BENCHMARK(BM_dictionary_scan<x86_512_scan>)->BM_ARGS;
#endif

BENCHMARK_MAIN();
