#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"

#define BM_ARGS UseRealTime()->Repetitions(1);

// TODO: make large but keep at multiple of 16
static constexpr uint64_t NUM_KEYS = 1024;
static constexpr size_t COMPRESS_BITS = 9;

namespace {

template <typename T, size_t ALIGN>
struct AlignedData {
  explicit AlignedData(size_t num_entries) : data{static_cast<T*>(std::aligned_alloc(ALIGN, num_entries * sizeof(T)))} {
    if (data == nullptr) {
      throw std::runtime_error{"Could not allocate memory. " + std::string{std::strerror(errno)}};
    }
  }
  T* data;
};

using CompressedColumn = AlignedData<uint64_t, 512>;
using DecompressedColumn = AlignedData<uint32_t, 512>;

CompressedColumn compress_input(const std::vector<uint32_t>& input) {
  constexpr uint64_t U64_BITS = 64;
  constexpr uint64_t MASK = (1 << COMPRESS_BITS) - 1;

  const size_t tuples_per_u64 = U64_BITS / COMPRESS_BITS;
  CompressedColumn compressed_data(input.size() * (tuples_per_u64 + 1));
  uint64_t* buffer = compressed_data.data;

  uint64_t bits_left = U64_BITS;
  size_t idx = 0;

  for (uint32_t i : input) {
    uint64_t val = i & MASK;
    buffer[idx] |= val << (U64_BITS - bits_left);

    if (bits_left < COMPRESS_BITS) {
      buffer[++idx] |= val >> bits_left;
      bits_left += U64_BITS;
    }
    bits_left -= COMPRESS_BITS;
    if (bits_left == 0) {
      bits_left = U64_BITS;
      idx++;
    }
  }

  return compressed_data;
}

}  // namespace

template <typename ScanFn>
void BM_scanning(benchmark::State& state) {
  ScanFn scan_fn{};
  uint64_t bits_needed = COMPRESS_BITS;  // hard-coded for now. otherwise: state.range(0);

  // Seed rng for same benchmark runs.
  std::mt19937_64 rng{82323457236434673ul};

  std::vector<uint32_t> tuples(NUM_KEYS);
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    // TODO: make random
//    tuples[i] = rng() % (1 << bits_needed);
    tuples[i] = 42;
  }

  CompressedColumn compressed_column = compress_input(tuples);

  // Do one pass to test that the code is correct
  {
    DecompressedColumn decompressed_column{NUM_KEYS};
    scan_fn(compressed_column.data, decompressed_column.data, NUM_KEYS);
    if (std::memcmp(decompressed_column.data, tuples.data(), NUM_KEYS * sizeof(uint32_t)) != 0) {
      throw std::runtime_error{"Wrong decompression result."};
    }
  }

  DecompressedColumn decompressed_column{NUM_KEYS};
  for (auto _ : state) {
    scan_fn(compressed_column.data, decompressed_column.data, NUM_KEYS);
    benchmark::DoNotOptimize(decompressed_column.data);
  }
}

#if defined(__aarch64__)
#include <arm_neon.h>

struct neon_scan {
  static constexpr size_t VALUES_PER_ITERATION = 4;
  static constexpr size_t ITERATIONS_PER_BATCH = (16 * 8) / COMPRESS_BITS / VALUES_PER_ITERATION;
  static constexpr size_t VALUES_PER_BATCH = VALUES_PER_ITERATION * ITERATIONS_PER_BATCH;
  static constexpr size_t BITS_PER_BATCH = VALUES_PER_BATCH * COMPRESS_BITS;
  static constexpr size_t DANGLING_BITS_PER_BATCH = BITS_PER_BATCH % 8;

  static constexpr uint8x16_t SHUFFLE_MASKS[3] = {
      uint8x16_t{6, 5, 4, 3, 5, 4, 3, 2, 5, 3, 2, 1, 3, 2, 1, 0},
      uint8x16_t{10, 9, 8, 7, 9, 8, 7, 6, 8, 7, 6, 5, 7, 6, 5, 4},
      uint8x16_t{14, 13, 12, 11, 13, 12, 11, 10, 12, 11, 10, 9, 11, 10, 9, 8},
  };

  static constexpr uint32x4_t BYTE_ALIGN_MASK = {1 << 0, 1 << 1, 1 << 2, 1 << 3};
  static constexpr uint8x16_t AND_MASK = {0, 0, 1, 255, 0, 0, 1, 255, 0, 0, 1, 255, 0, 0, 1, 255};

  template <size_t ITER, typename CallbackFn>
  inline void decompress_iteration(uint8x16_t batch_lane, CallbackFn callback, const uint8_t dangling_bits) {
    auto shuffle_input = [&] {
      static_assert(ITER < 3, "Cannot do more than 3 iterations per lane.");
      // clang-format off
      switch (ITER) {
        case 0: return __builtin_shufflevector(batch_lane, batch_lane, 0, 1, 2, 3, 1, 2, 3, 5, 2, 3, 4, 5, 3, 4, 5, 6);
        case 1: return __builtin_shufflevector(batch_lane, batch_lane, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
        case 2: return __builtin_shufflevector(batch_lane, batch_lane, 8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14);
      }
      // clang-format on
    };

    uint8x16_t lane = shuffle_input();
    lane = vshlq_s32(vreinterpretq_s32_u8(lane), BYTE_ALIGN_MASK);

    // There are 4 values per iteration, the first is shifted by 0 bits, the second by 1, the third by 2, and the fourth
    // by 3. So we need to always shift by 3. In the next iteration, the first is shifted by 4 bits, and so on. So we
    // need to shift by 4 * `iter` to include this shift.
    const int32_t shift = 3 + (ITER * 4) + dangling_bits;

    // NEON does not support right-shifting with runtime values. So we shift left by a negative value.
    int32x4_t shift_lane = vmovq_n_s32(-shift);

    lane = vshlq_s32(vreinterpretq_s32_u8(lane), shift_lane);
    lane = vandq_u8(lane, AND_MASK);
    callback(lane);
  }

  template <typename CallbackFn>
  void decompressor(const void* __restrict input_compressed, size_t num_tuples, CallbackFn callback) {
    const size_t num_batches = num_tuples / VALUES_PER_BATCH;

    const auto* compressed_data = static_cast<const uint8_t*>(input_compressed);

    for (size_t batch = 0; batch < num_batches; ++batch) {
      // In odd runs, we may have dangling bits at beginning of batch.
      const uint8_t dangling_bits = DANGLING_BITS_PER_BATCH * (batch % 2);
      const size_t offset = (batch * BITS_PER_BATCH) / 8;
      const uint8_t* pos = compressed_data + offset;

      uint8x16_t batch_lane = vld1q_u8(pos);
      decompress_iteration<0>(batch_lane, callback, dangling_bits);
      decompress_iteration<1>(batch_lane, callback, dangling_bits);
      decompress_iteration<2>(batch_lane, callback, dangling_bits);
    }
  }

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    auto store_fn = [&](uint32x4_t decompressed_values) {
      vst1q_u32(output, decompressed_values);
      output += VALUES_PER_ITERATION;
    };

    decompressor(input, num_tuples, store_fn);
  }
};

BENCHMARK(BM_scanning<neon_scan>)->BM_ARGS;

#endif

#if defined(__x86_64__)
#include <immintrin.h>

struct x86_128_scan {
  static constexpr size_t VALUES_PER_ITERATION = 4;
  static constexpr size_t ITERATIONS_PER_BATCH = (16 * 8) / COMPRESS_BITS / VALUES_PER_ITERATION;
  static constexpr size_t VALUES_PER_BATCH = VALUES_PER_ITERATION * ITERATIONS_PER_BATCH;
  static constexpr size_t BITS_PER_BATCH = VALUES_PER_BATCH * COMPRESS_BITS;
  static constexpr size_t DANGLING_BITS_PER_BATCH = BITS_PER_BATCH % 8;

  const __m128i SHUFFLE_MASKS[3] = {
      _mm_set_epi8(6, 5, 4, 3, 5, 4, 3, 2, 5, 3, 2, 1, 3, 2, 1, 0),
      _mm_set_epi8(10, 9, 8, 7, 9, 8, 7, 6, 8, 7, 6, 5, 7, 6, 5, 4),
      _mm_set_epi8(14, 13, 12, 11, 13, 12, 11, 10, 12, 11, 10, 9, 11, 10, 9, 8),
  };

  const __m128i BYTE_ALIGN_MASK = _mm_set_epi32(1 << 0, 1 << 1, 1 << 2, 1 << 3);
  const __m128i AND_MASK = _mm_set_epi8(0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1);

  template <typename CallbackFn>
  inline void decompress_iteration(__m128i batch_lane, CallbackFn callback, const uint8_t dangling_bits, uint8_t iter) {
    __m128i lane = _mm_shuffle_epi8(batch_lane, SHUFFLE_MASKS[iter]);
    lane = _mm_mullo_epi32(lane, BYTE_ALIGN_MASK);

    // There are 4 values per iteration, the first is shifted by 0 bits, the second by 1, the third by 2, and the fourth
    // by 3. So we need to always shift by 3. In the next iteration, the first is shifted by 4 bits, and so on. So we
    // need to shift by 4 * `iter` to include this shift.
    const int32_t shift = 3 + (iter * 4) + dangling_bits;

    lane = _mm_srli_epi32(lane, shift);
    lane = _mm_and_si128(lane, AND_MASK);
    callback(lane);
  }

  template <typename CallbackFn>
  void decompressor(const void* __restrict input_compressed, size_t num_tuples, CallbackFn callback) {
    const size_t num_batches = num_tuples / VALUES_PER_BATCH;

    const auto* compressed_data = static_cast<const uint8_t*>(input_compressed);

    for (size_t batch = 0; batch < num_batches; ++batch) {
      // In odd runs, we may have dangling bits at beginning of batch.
      const uint8_t dangling_bits = DANGLING_BITS_PER_BATCH * (batch % 2);
      const size_t offset = (batch * BITS_PER_BATCH) / 8;
      const auto* pos = reinterpret_cast<const __m128i*>(compressed_data + offset);

      __m128i batch_lane = _mm_loadu_si128(pos);
      decompress_iteration(batch_lane, callback, dangling_bits, 0);
      decompress_iteration(batch_lane, callback, dangling_bits, 1);
      decompress_iteration(batch_lane, callback, dangling_bits, 2);
      decompress_iteration(batch_lane, callback, dangling_bits, 3);
    }
  }

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    auto store_fn = [&](__m128i decompressed_values) {
      _mm_stream_si128(reinterpret_cast<__m128i*>(output), decompressed_values);
      output += VALUES_PER_ITERATION;
    };

    decompressor(input, num_tuples, store_fn);
  }
};

struct x86_512_scan {
  static constexpr size_t VALUES_PER_BATCH = (16 * 3) + 8;
  static constexpr size_t BYTES_PER_BATCH = (VALUES_PER_BATCH * COMPRESS_BITS) / 8;
  static constexpr size_t BYTE_PER_ITERATION16 = 16 * 9 / 8;
  static constexpr size_t BYTE_PER_ITERATION8 = 8 * 9 / 8;

  // clang-format off
  const __m512i LANE_SHUFFLE_MASKS[4] = {
    _mm512_set_epi16(13, 12, 11, 10, 9, 8, 7, 6, 11, 10, 9, 8, 7, 6, 5, 4, 9, 8, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm512_set_epi16(22, 21, 20, 19, 18, 17, 16, 15, 20, 19, 18, 17, 16, 15, 14, 13, 18, 17, 16, 15, 14, 13, 12, 11, 16, 15, 14, 13, 12, 11, 10, 9),
    _mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 29, 28, 27, 26, 25, 24, 23, 22, 27, 26, 25, 24, 23, 22, 21, 20, 25, 24, 23, 22, 21, 20, 19, 18),
    _mm512_set_epi16(40, 39, 38, 37, 36, 35, 34, 33, 38, 37, 36, 35, 34, 33, 32, 31, 36, 35, 34, 33, 32, 31, 30, 29, 34, 33, 32, 31, 30, 29, 28, 27)
  };
  // clang-format on

  const __m512i SHUFFLE_MASK =
      _mm512_set_epi8(6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0, 6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0, 6,
                      5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0, 6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0);

  const __m512i SHIFT_MASK = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  const __m512i AND_MASK = _mm512_set1_epi32((1 << COMPRESS_BITS) - 1);

  inline __m512i decompress(__m512i batch_lane, uint16_t iter) {
    __m512i lane = _mm512_permutexvar_epi16(LANE_SHUFFLE_MASKS[iter], batch_lane);
    lane = _mm512_shuffle_epi8(lane, SHUFFLE_MASK);
    lane = _mm512_srlv_epi32(lane, SHIFT_MASK);
    lane = _mm512_and_epi32(lane, AND_MASK);
    return lane;
  }

  template <typename CallbackFn>
  inline void decompress16(__m512i batch_lane, CallbackFn callback, uint16_t iter) {
    __m512i lane = decompress(batch_lane, iter);
    callback(lane);
  }

  template <typename CallbackFn>
  inline void decompress8(__m512i batch_lane, CallbackFn callback) {
    __m512i lane = decompress(batch_lane, 3);
    __m256i half_lane = _mm512_castsi512_si256(lane);
    callback(half_lane);
  }

  template <typename CallbackFn>
  inline void decompress_batch(const __m512i batch_lane, CallbackFn callback) {
    decompress16(batch_lane, callback, 0);
    decompress16(batch_lane, callback, 1);
    decompress16(batch_lane, callback, 2);
    decompress8(batch_lane, callback);
  }

  template <typename CallbackFn>
  void decompressor(void* input_compressed, size_t input_size, CallbackFn callback) {
    const size_t num_batches = input_size / VALUES_PER_BATCH;
    auto* compressed_data = static_cast<uint8_t*>(input_compressed);

    for (size_t batch = 0; batch < num_batches; ++batch) {
      const size_t offset = batch * BYTES_PER_BATCH;
      auto* pos = reinterpret_cast<__m512i*>(compressed_data + offset);
      __m512i batch_lane = _mm512_loadu_si512(pos);
      decompress_batch(batch_lane, callback);
    }
  }

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t bits_needed) {}
};

BENCHMARK(BM_scaning<x86_128_scan>)->BM_ARGS;
BENCHMARK(BM_scaning<x86_512_scan>)->BM_ARGS;
#endif

struct naive_scalar_scan {
  void operator()(const uint64_t* input, uint32_t* output, size_t num_tuples) {
    constexpr size_t U64_BITS = sizeof(uint64_t) * 8;
    uint64_t bits_left = 0;
    constexpr uint64_t MASK = (1 << COMPRESS_BITS) - 1;

    size_t block_idx = 0;

    for (size_t number_index = 0; number_index < num_tuples; ++number_index) {
      output[number_index] = (input[block_idx] >> bits_left) & MASK;

      // We have a few most significant bits left over in the next block
      if (bits_left > U64_BITS) {
        output[number_index] |= input[++block_idx] << (U64_BITS - bits_left) & MASK;
      }

      bits_left += COMPRESS_BITS;
      if (bits_left == U64_BITS) {
        block_idx++;
        bits_left = 0;
      }
    }
  }
};

struct autovec_scalar_scan {
  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    constexpr size_t U64_BITS = sizeof(uint64_t) * 8;
    uint64_t bits_left = 0;
    constexpr uint64_t MASK = (1 << COMPRESS_BITS) - 1;

    size_t block_idx = 0;

    for (size_t number_index = 0; number_index < num_tuples; ++number_index) {
      output[number_index] = (input[block_idx] >> bits_left) & MASK;

      // We have a few most significant bits left over in the next block
      if (bits_left > U64_BITS) {
        output[number_index] |= input[++block_idx] << (U64_BITS - bits_left) & MASK;
      }

      bits_left += COMPRESS_BITS;
      if (bits_left == U64_BITS) {
        block_idx++;
        bits_left = 0;
      }
    }
  }
};

struct vector_scan {
  static constexpr size_t VALUES_PER_ITERATION = 4;
  static constexpr size_t ITERATIONS_PER_BATCH = (16 * 8) / COMPRESS_BITS / VALUES_PER_ITERATION;
  static constexpr size_t VALUES_PER_BATCH = VALUES_PER_ITERATION * ITERATIONS_PER_BATCH;
  static constexpr size_t BITS_PER_BATCH = VALUES_PER_BATCH * COMPRESS_BITS;
  static constexpr size_t DANGLING_BITS_PER_BATCH = BITS_PER_BATCH % 8;

  template <typename T>
  using VecT __attribute__((vector_size(16))) = T;

  using VecU8x16 = VecT<uint8_t>;
  using VecS32x4 = VecT<int32_t>;

  const VecU8x16 SHUFFLE_MASKS[3] = {
      VecU8x16{6, 5, 4, 3, 5, 4, 3, 2, 5, 3, 2, 1, 3, 2, 1, 0},
      VecU8x16{10, 9, 8, 7, 9, 8, 7, 6, 8, 7, 6, 5, 7, 6, 5, 4},
      VecU8x16{14, 13, 12, 11, 13, 12, 11, 10, 12, 11, 10, 9, 11, 10, 9, 8},
  };

  static constexpr VecS32x4 BYTE_ALIGN_MASK = {1 << 0, 1 << 1, 1 << 2, 1 << 3};
  static constexpr VecU8x16 AND_MASK = {0, 0, 1, 255, 0, 0, 1, 255, 0, 0, 1, 255, 0, 0, 1, 255};

  template <size_t ITER, typename CallbackFn>
  inline void decompress_iteration(VecU8x16 batch_lane, CallbackFn callback, const uint8_t dangling_bits) {
    auto shuffle_input = [&] {
      static_assert(ITER < 3, "Cannot do more than 3 iterations per lane.");
      // We can do __builtin_shufflevector in both clang and gcc, but we want to show that this works slightly
      // differently in both, as clang does not support __builtin_shuffle with a runtime mask.
#if GCC_COMPILER
      return __builtin_shuffle(batch_lane, SHUFFLE_MAKS[ITER]);
#else
      // clang-format off
      switch (ITER) {
        case 0: return __builtin_shufflevector(batch_lane, batch_lane, 0, 1, 2, 3, 1, 2, 3, 5, 2, 3, 4, 5, 3, 4, 5, 6);
        case 1: return __builtin_shufflevector(batch_lane, batch_lane, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
        case 2: return __builtin_shufflevector(batch_lane, batch_lane, 8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14);
      }
        // clang-format on
#endif
    };

    VecU8x16 lane = shuffle_input();
    lane = reinterpret_cast<VecS32x4&>(lane) * BYTE_ALIGN_MASK;

    // There are 4 values per iteration, the first is shifted by 0 bits, the second by 1, the third by 2, and the fourth
    // by 3. So we need to always shift by 3. In the next iteration, the first is shifted by 4 bits, and so on. So we
    // need to shift by 4 * `iter` to include this shift.
    const int32_t shift = 3 + (ITER * 4) + dangling_bits;

    lane = reinterpret_cast<VecS32x4&>(lane) >> shift;
    lane = lane && AND_MASK;
    callback(lane);
  }

  template <typename CallbackFn>
  void decompressor(const void* __restrict input_compressed, size_t num_tuples, CallbackFn callback) {
    const size_t num_batches = num_tuples / VALUES_PER_BATCH;

    const auto* compressed_data = static_cast<const uint8_t*>(input_compressed);

    for (size_t batch = 0; batch < num_batches; ++batch) {
      // In odd runs, we may have dangling bits at beginning of batch.
      const uint8_t dangling_bits = DANGLING_BITS_PER_BATCH * (batch % 2);
      const size_t offset = (batch * BITS_PER_BATCH) / 8;
      const auto* pos = compressed_data + offset;

      auto batch_lane = *reinterpret_cast<const VecU8x16*>(pos);
      decompress_iteration<0>(batch_lane, callback, dangling_bits);
      decompress_iteration<1>(batch_lane, callback, dangling_bits);
      decompress_iteration<2>(batch_lane, callback, dangling_bits);
    }
  }

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    auto store_fn = [&](VecS32x4 decompressed_values) {
      *reinterpret_cast<VecS32x4*>(output) = decompressed_values;
      output += VALUES_PER_ITERATION;
    };

    decompressor(input, num_tuples, store_fn);
  }
};

BENCHMARK(BM_scanning<naive_scalar_scan>)->BM_ARGS;
BENCHMARK(BM_scanning<autovec_scalar_scan>)->BM_ARGS;
BENCHMARK(BM_scanning<vector_scan>)->BM_ARGS;

BENCHMARK_MAIN();
