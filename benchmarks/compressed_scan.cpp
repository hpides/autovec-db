#include <array>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"

// This is the lowest common multiple of 12 and 56. We need 12 for the 128-Bit version and 56 for the 512-Bit one.
static constexpr uint64_t NUM_BASE_TUPLES = 168;

// static constexpr uint64_t SCALE_FACTOR = 1;
static constexpr uint64_t SCALE_FACTOR = 6000;  // With a 168 base, this gives us 1'008'000 tuples.

static constexpr uint64_t NUM_TUPLES = NUM_BASE_TUPLES * SCALE_FACTOR;
static constexpr size_t COMPRESS_BITS = 9;

namespace {

using CompressedColumn = AlignedData<uint64_t, 64>;
using DecompressedColumn = AlignedData<uint32_t, 64>;

CompressedColumn compress_input(const std::vector<uint32_t>& input) {
  constexpr uint64_t U64_BITS = 64;
  constexpr uint64_t MASK = (1 << COMPRESS_BITS) - 1;

  const size_t tuples_per_u64 = U64_BITS / COMPRESS_BITS;
  CompressedColumn compressed_data(input.size() * (tuples_per_u64 + 1));
  uint64_t* buffer = compressed_data.aligned_data();

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

[[maybe_unused]] void print_bytes_right_to_left(void* data, size_t num_bytes, std::ostream& os) {
  auto* bytes = reinterpret_cast<uint8_t*>(data);
  for (size_t offset = num_bytes; offset > 0; --offset) {
    os << std::hex << std::setfill('0') << std::setw(2) << std::bitset<8>(bytes[offset - 1]).to_ulong() << ' ';
  }
  os << std::endl;
}

[[maybe_unused]] void print_bits_right_to_left(void* data, size_t num_bytes, std::ostream& os) {
  auto* bytes = reinterpret_cast<uint8_t*>(data);
  for (size_t offset = num_bytes; offset > 0; --offset) {
    os << std::bitset<8>(bytes[offset - 1]) << ' ';
  }
  os << std::endl;
}

template <typename SimdLane>
void print_lane(SimdLane* lane, bool as_bits = true, std::ostream& os = std::cout) {
  constexpr size_t num_bytes = sizeof(SimdLane);
  if (as_bits) {
    return print_bits_right_to_left(lane, num_bytes, os);
  }
  return print_bytes_right_to_left(lane, num_bytes, os);
}

}  // namespace

template <typename ScanFn>
void BM_scanning(benchmark::State& state) {
  ScanFn scan_fn{};

  // Seed rng for same benchmark runs.
  std::mt19937 rng{13131313};

  std::vector<uint32_t> tuples(NUM_TUPLES);
  for (size_t i = 0; i < NUM_TUPLES; ++i) {
    tuples[i] = rng() & ((1 << COMPRESS_BITS) - 1);
    //    tuples[i] = 42;  // for debugging
  }

  CompressedColumn compressed_column = compress_input(tuples);
  DecompressedColumn decompressed_column{NUM_TUPLES};

  // Do one pass to test that the code is correct.
  {
    scan_fn(compressed_column.aligned_data(), decompressed_column.aligned_data(), NUM_TUPLES);
    for (size_t i = 0; i < NUM_TUPLES; ++i) {
      if (decompressed_column.aligned_data()[i] != tuples[i]) {
        throw std::runtime_error{"Wrong decompression result at [" + std::to_string(i) + "]"};
      }
    }
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(compressed_column.aligned_data());
    scan_fn(compressed_column.aligned_data(), decompressed_column.aligned_data(), NUM_TUPLES);
    benchmark::DoNotOptimize(decompressed_column.aligned_data());
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

  // Note: the masks are inverted, i.e., they are ordered as {nth, (n-1)th, (n-2)th, ..., 0th}.
  static constexpr uint32x4_t BYTE_ALIGN_MASK = {3, 2, 1, 0};
  static constexpr uint8x16_t AND_MASK = {255, 1, 0, 0, 255, 1, 0, 0, 255, 1, 0, 0, 255, 1, 0, 0};

  template <size_t ITER, typename CallbackFn>
  inline void decompress_iteration(uint8x16_t batch_lane, CallbackFn callback, const uint8_t dangling_bits) {
    auto shuffle_input = [&] {
      static_assert(ITER < 3, "Cannot do more than 3 iterations per lane.");
      // clang-format off
      switch (ITER) {
        case 0: return __builtin_shufflevector(batch_lane, batch_lane, 0, 1, 2, 3, 1, 2, 3, 5, 2, 3, 4, 5, 3, 4, 5, 6);
        case 1: return __builtin_shufflevector(batch_lane, batch_lane, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
        case 2: return __builtin_shufflevector(batch_lane, batch_lane, 8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14);
        default: __builtin_unreachable();
      }
      // clang-format on
    };

    DEBUG_DO(std::cout << "load:  "; print_lane(&batch_lane););

    uint8x16_t lane = shuffle_input();
    DEBUG_DO(std::cout << "a16#" << ITER << ": "; print_lane(&lane););

    lane = vshlq_s32(vreinterpretq_s32_u8(lane), BYTE_ALIGN_MASK);
    DEBUG_DO(std::cout << "a4 #" << ITER << ": "; print_lane(&lane););

    // There are 4 values per iteration, the first is shifted by 0 bits, the second by 1, the third by 2, and the fourth
    // by 3. So we need to always shift by 3. In the next iteration, the first is shifted by 4 bits, and so on. So we
    // need to shift by 4 * `iter` to include this shift.
    const int32_t shift = 3 + (ITER * 4) + dangling_bits;

    // NEON does not support right-shifting with runtime values. So we shift left by a negative value.
    int32x4_t shift_lane = vmovq_n_s32(-shift);

    lane = vshlq_s32(vreinterpretq_s32_u8(lane), shift_lane);
    DEBUG_DO(std::cout << "bit#" << ITER << ": "; print_lane(&lane););

    lane = vandq_u8(lane, AND_MASK);
    DEBUG_DO(std::cout << "and#" << ITER << ": "; print_lane(&lane););

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

  template <size_t ITER, uint8_t DANGLING_BITS, typename CallbackFn>
  inline void decompress_iteration(__m128i batch_lane, CallbackFn callback) {
    DEBUG_DO(std::cout << "load:  "; print_lane(&batch_lane););

    __m128i lane = _mm_shuffle_epi8(batch_lane, SHUFFLE_MASKS[ITER]);
    DEBUG_DO(std::cout << "a16#" << ITER << ": "; print_lane(&lane););

    // We need to use a multiply here instead of a shift, as _mm_sllv_epi32 is not available until AVX2.
    lane = _mm_mullo_epi32(lane, BYTE_ALIGN_MASK);
    DEBUG_DO(std::cout << "a4 #" << ITER << ": "; print_lane(&lane););

    // There are 4 values per iteration, the first is shifted by 0 bits, the second by 1, the third by 2, and the fourth
    // by 3. So we need to always shift by 3. In the next iteration, the first is shifted by 4 bits, and so on. So we
    // need to shift by 4 * `iter` to include this shift.
    constexpr int32_t shift = 3 + (ITER * 4) + DANGLING_BITS;

    lane = _mm_srli_epi32(lane, shift);
    DEBUG_DO(std::cout << "bit#" << ITER << ": "; print_lane(&lane););

    lane = _mm_and_si128(lane, AND_MASK);
    DEBUG_DO(std::cout << "and#" << ITER << ": "; print_lane(&lane););

    callback(lane);
  }

  template <typename CallbackFn>
  void decompressor(const void* __restrict input_compressed, size_t num_tuples, CallbackFn callback) {
    const size_t num_batches = num_tuples / VALUES_PER_BATCH;

    const auto* compressed_data = static_cast<const uint8_t*>(input_compressed);

    assert(num_batches % 2 == 0);
    for (size_t batch = 0; batch < num_batches; batch += 2) {
      {
        const size_t offset = ((batch + 0) * BITS_PER_BATCH) / 8;
        const auto* pos = reinterpret_cast<const __m128i*>(compressed_data + offset);
        __m128i batch_lane = _mm_loadu_si128(pos);
        decompress_iteration<0, 0>(batch_lane, callback);
        decompress_iteration<1, 0>(batch_lane, callback);
        decompress_iteration<2, 0>(batch_lane, callback);
      }
      {
        const size_t offset = ((batch + 1) * BITS_PER_BATCH) / 8;
        const auto* pos = reinterpret_cast<const __m128i*>(compressed_data + offset);
        __m128i batch_lane = _mm_loadu_si128(pos);
        // In odd runs, we may have dangling bits at beginning of batch.
        decompress_iteration<0, DANGLING_BITS_PER_BATCH>(batch_lane, callback);
        decompress_iteration<1, DANGLING_BITS_PER_BATCH>(batch_lane, callback);
        decompress_iteration<2, DANGLING_BITS_PER_BATCH>(batch_lane, callback);
      }
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
#endif

#if defined(AVX512_AVAILABLE)
struct x86_512_scan {
  static constexpr size_t VALUES_PER_BATCH = (16 * 3) + 8;
  static constexpr size_t BYTES_PER_BATCH = (VALUES_PER_BATCH * COMPRESS_BITS) / 8;

  // clang-format off
  const __m512i LANE_SHUFFLE_MASKS[4] = {
    _mm512_set_epi16(13, 12, 11, 10, 9, 8, 7, 6, 11, 10, 9, 8, 7, 6, 5, 4, 9, 8, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2, 1, 0),
    _mm512_set_epi16(22, 21, 20, 19, 18, 17, 16, 15, 20, 19, 18, 17, 16, 15, 14, 13, 18, 17, 16, 15, 14, 13, 12, 11, 16, 15, 14, 13, 12, 11, 10, 9),
    _mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 29, 28, 27, 26, 25, 24, 23, 22, 27, 26, 25, 24, 23, 22, 21, 20, 25, 24, 23, 22, 21, 20, 19, 18),
    _mm512_set_epi16(40, 39, 38, 37, 36, 35, 34, 33, 38, 37, 36, 35, 34, 33, 32, 31, 36, 35, 34, 33, 32, 31, 30, 29, 34, 33, 32, 31, 30, 29, 28, 27)
  };

  const __m512i SHUFFLE_MASK = _mm512_set_epi8(6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0,
                                               6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0,
                                               6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0,
                                               6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0);
  // clang-format on

  const __m512i SHIFT_MASK = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  const __m512i AND_MASK = _mm512_set1_epi32((1 << COMPRESS_BITS) - 1);

  template <size_t ITER>
  inline __m512i decompress(__m512i batch_lane) {
    DEBUG_DO(std::cout << "load: "; print_lane(&batch_lane););

    __m512i lane = _mm512_permutexvar_epi16(LANE_SHUFFLE_MASKS[ITER], batch_lane);
    DEBUG_DO(std::cout << "a16 : "; print_lane(&lane););

    lane = _mm512_shuffle_epi8(lane, SHUFFLE_MASK);
    DEBUG_DO(std::cout << "a4  : "; print_lane(&lane););

    lane = _mm512_srlv_epi32(lane, SHIFT_MASK);
    DEBUG_DO(std::cout << "bit : "; print_lane(&lane););

    lane = _mm512_and_epi32(lane, AND_MASK);
    DEBUG_DO(std::cout << "and : "; print_lane(&lane););

    return lane;
  }

  template <typename CallbackFn>
  inline void decompress_batch(const __m512i batch_lane, CallbackFn callback) {
    callback(decompress<0>(batch_lane));
    callback(decompress<1>(batch_lane));
    callback(decompress<2>(batch_lane));

    // Last iteration are only 8 values.
    callback(_mm512_castsi512_si256(decompress<3>(batch_lane)));
  }

  template <typename CallbackFn>
  void decompressor(const void* input_compressed, size_t num_tuples, CallbackFn callback) {
    const size_t num_batches = num_tuples / VALUES_PER_BATCH;
    const auto* compressed_data = static_cast<const uint8_t*>(input_compressed);

    for (size_t batch = 0; batch < num_batches; ++batch) {
      const size_t offset = batch * BYTES_PER_BATCH;
      const auto* pos = reinterpret_cast<const __m512i*>(compressed_data + offset);
      __m512i batch_lane = _mm512_loadu_si512(pos);
      decompress_batch(batch_lane, callback);
    }
  }

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    // We don't care about the attributes here, and we know that this is okay.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
    auto store_fn = [&]<typename Lane>(Lane decompressed_values) {
      if constexpr (std::is_same_v<Lane, __m512i>) {
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(output), reinterpret_cast<__m512i&>(decompressed_values));
        output += 16;
      } else if constexpr (std::is_same_v<Lane, __m256i>) {
        _mm256_store_si256(reinterpret_cast<__m256i*>(output), reinterpret_cast<__m256i&>(decompressed_values));
        output += 8;
      }
    };
#pragma GCC diagnostic pop

    decompressor(input, num_tuples, store_fn);
  }
};
#endif

struct naive_scalar_scan {
  void operator()(const uint64_t* input, uint32_t* output, size_t num_tuples) {
    constexpr size_t U64_BITS = sizeof(uint64_t) * 8;
    constexpr uint64_t MASK = (1 << COMPRESS_BITS) - 1;
    uint64_t bits_left = U64_BITS;

    size_t block_idx = 0;

    for (size_t i = 0; i < num_tuples; ++i) {
      const size_t bits_for_current_value = U64_BITS - bits_left;
      output[i] |= (input[block_idx] >> bits_for_current_value) & MASK;

      // Did not get all bits for value, check next block to get them.
      if (bits_left < COMPRESS_BITS) {
        // Shift remaining bits to correct position for bit-OR.
        output[i] |= (input[++block_idx] << bits_left) & MASK;
        bits_left += U64_BITS;
      }

      bits_left -= COMPRESS_BITS;

      if (bits_left == 0) {
        block_idx++;
        bits_left = U64_BITS;
      }
    }
  }
};

struct autovec_scalar_scan {
  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    constexpr size_t U64_BITS = sizeof(uint64_t) * 8;
    constexpr uint64_t MASK = (1 << COMPRESS_BITS) - 1;
    uint64_t bits_left = U64_BITS;

    size_t block_idx = 0;

    for (size_t i = 0; i < num_tuples; ++i) {
      const size_t bits_for_current_value = U64_BITS - bits_left;
      output[i] |= (input[block_idx] >> bits_for_current_value) & MASK;

      // Did not get all bits for value, check next block to get them.
      if (bits_left < COMPRESS_BITS) {
        // Shift remaining bits to correct position for bit-OR.
        output[i] |= (input[++block_idx] << bits_left) & MASK;
        bits_left += U64_BITS;
      }

      bits_left -= COMPRESS_BITS;

      if (bits_left == 0) {
        block_idx++;
        bits_left = U64_BITS;
      }
    }
  }
};

struct vector_128_scan {
  static constexpr size_t VALUES_PER_ITERATION = 4;
  static constexpr size_t ITERATIONS_PER_BATCH = (16 * 8) / COMPRESS_BITS / VALUES_PER_ITERATION;
  static constexpr size_t VALUES_PER_BATCH = VALUES_PER_ITERATION * ITERATIONS_PER_BATCH;
  static constexpr size_t BITS_PER_BATCH = VALUES_PER_BATCH * COMPRESS_BITS;
  static constexpr size_t DANGLING_BITS_PER_BATCH = BITS_PER_BATCH % 8;

  static constexpr size_t VECTOR_SIZE_IN_BYTES = 16;

  using VecU8x16 = VecT<uint8_t, VECTOR_SIZE_IN_BYTES>;
  using UnalignedVecU8x16 UNALIGNED = VecT<uint8_t, VECTOR_SIZE_IN_BYTES>;
  using VecU32x4 = VecT<uint32_t, VECTOR_SIZE_IN_BYTES>;

  // Note: the masks are regular, i.e., they are ordered as {0th, 1st, ..., nth}.
  static constexpr VecU8x16 SHUFFLE_MASKS[3] = {
      VecU8x16{0, 1, 2, 3, 1, 2, 3, 5, 2, 3, 4, 5, 3, 4, 5, 6},
      VecU8x16{4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10},
      VecU8x16{8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14},
  };

  // Note: the masks are inverted, i.e., they are ordered as {nth, (n-1)th, (n-2)th, ..., 0th}.
  static constexpr VecU32x4 BYTE_ALIGN_MASK = {3, 2, 1, 0};
  static constexpr VecU8x16 AND_MASK = {255, 1, 0, 0, 255, 1, 0, 0, 255, 1, 0, 0, 255, 1, 0, 0};

  template <size_t ITER, uint8_t DANGLING_BITS, typename CallbackFn>
  inline void decompress_iteration(VecU8x16 batch_lane, CallbackFn callback) {
    auto shuffle_input = [&] {
      static_assert(ITER < 3, "Cannot do more than 3 iterations per lane.");
#if GCC_COMPILER
      return __builtin_shuffle(batch_lane, SHUFFLE_MASKS[ITER]);
#else
      // clang-format off
      switch (ITER) {
        case 0: return __builtin_shufflevector(batch_lane, batch_lane, 0, 1, 2, 3, 1, 2, 3, 5, 2, 3, 4, 5, 3, 4, 5, 6);
        case 1: return __builtin_shufflevector(batch_lane, batch_lane, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
        case 2: return __builtin_shufflevector(batch_lane, batch_lane, 8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14);
        default: __builtin_unreachable();
          // clang-format on
      }
#endif
    };

    DEBUG_DO(std::cout << "load:  "; print_lane(&batch_lane););

    VecU8x16 lane = shuffle_input();
    DEBUG_DO(std::cout << "a16#" << ITER << ": "; print_lane(&lane););

    lane = reinterpret_cast<VecU32x4&>(lane) << BYTE_ALIGN_MASK;
    DEBUG_DO(std::cout << "a4 #" << ITER << ": "; print_lane(&lane););

    // There are 4 values per iteration, the first is shifted by 0 bits, the second by 1, the third by 2, and the fourth
    // by 3. So we need to always shift by 3. In the next iteration, the first is shifted by 4 bits, and so on. So we
    // need to shift by 4 * `iter` to include this shift.
    constexpr int32_t shift = 3 + (ITER * 4) + DANGLING_BITS;

    lane = reinterpret_cast<VecU32x4&>(lane) >> shift;
    DEBUG_DO(std::cout << "bit#" << ITER << ": "; print_lane(&lane););

    lane = lane & AND_MASK;
    DEBUG_DO(std::cout << "and#" << ITER << ": "; print_lane(&lane););

    callback(lane);
  }

  template <typename CallbackFn>
  void decompressor(const void* __restrict input_compressed, size_t num_tuples, CallbackFn callback) {
    const size_t num_batches = num_tuples / VALUES_PER_BATCH;

    const auto* compressed_data = static_cast<const uint8_t*>(input_compressed);

    assert(num_batches % 2 == 0);
    for (size_t batch = 0; batch < num_batches; batch += 2) {
      {
        const size_t offset = ((batch + 0) * BITS_PER_BATCH) / 8;
        auto batch_lane = *reinterpret_cast<const UnalignedVecU8x16*>(compressed_data + offset);
        decompress_iteration<0, 0>(batch_lane, callback);
        decompress_iteration<1, 0>(batch_lane, callback);
        decompress_iteration<2, 0>(batch_lane, callback);
      }
      {
        const size_t offset = ((batch + 1) * BITS_PER_BATCH) / 8;
        auto batch_lane = *reinterpret_cast<const UnalignedVecU8x16*>(compressed_data + offset);
        // In odd runs, we may have dangling bits at beginning of batch.
        decompress_iteration<0, DANGLING_BITS_PER_BATCH>(batch_lane, callback);
        decompress_iteration<1, DANGLING_BITS_PER_BATCH>(batch_lane, callback);
        decompress_iteration<2, DANGLING_BITS_PER_BATCH>(batch_lane, callback);
      }
    }
  }

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    auto store_fn = [&](VecU32x4 decompressed_values) {
      *reinterpret_cast<VecU32x4*>(output) = decompressed_values;
      output += VALUES_PER_ITERATION;
    };

    decompressor(input, num_tuples, store_fn);
  }
};

struct vector_512_scan {
  static constexpr size_t VALUES_PER_BATCH = (16 * 3) + 8;
  static constexpr size_t BYTES_PER_BATCH = (VALUES_PER_BATCH * COMPRESS_BITS) / 8;

  static constexpr size_t VECTOR_SIZE_IN_BYTES = 64;

  using VecU8x64 = VecT<uint8_t, VECTOR_SIZE_IN_BYTES>;
  using VecU16x32 = VecT<uint16_t, VECTOR_SIZE_IN_BYTES>;
  using VecU32x16 = VecT<uint32_t, VECTOR_SIZE_IN_BYTES>;
  using UnalignedVecU16x32 UNALIGNED = VecT<uint32_t, VECTOR_SIZE_IN_BYTES>;

  // For our half-lane output.
  using VecU32x8 = VecT<uint32_t, VECTOR_SIZE_IN_BYTES / 2>;
  using UnalignedVecU32x8 UNALIGNED = VecT<uint32_t, VECTOR_SIZE_IN_BYTES / 2>;

  // clang-format off
  static constexpr VecU16x32 LANE_SHUFFLE_MASKS[4] = {
    VecU16x32{0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13},
    VecU16x32{9, 10, 11, 12, 13, 14, 15, 16, 11, 12, 13, 14, 15, 16, 17, 18, 13, 14, 15, 16, 17, 18, 19, 20, 15, 16, 17, 18, 19, 20, 21, 22},
    VecU16x32{18, 19, 20, 21, 22, 23, 24, 25, 20, 21, 22, 23, 24, 25, 26, 27, 22, 23, 24, 25, 26, 27, 28, 29, 24, 25, 26, 27, 28, 29, 30, 31},
    VecU16x32{27, 28, 29, 30, 31, 32, 33, 34, 29, 30, 31, 32, 33, 34, 35, 36, 31, 32, 33, 34, 35, 36, 37, 38, 33, 34, 35, 36, 37, 38, 39, 40},
  };

  // Note: ordered from nth, n-1th, ..., 0th byte. But input is also n down to 0, so this is from "right" to "left" in
  // our print helper function. This is different from the AVX512 version, where we select for each 16 Byte chunk. That
  // is why it is 0-6 for the values there. Here we select from the global byte ordering, so we need all 64 Bytes.
  static constexpr VecU8x64 SHUFFLE_MASK = { 0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6,
                                            16, 17, 18, 19, 17, 18, 19, 20, 18, 19, 20, 21, 19, 20, 21, 22,
                                            32, 33, 34, 35, 33, 34, 35, 36, 34, 35, 36, 37, 35, 36, 37, 38,
                                            48, 49, 50, 51, 49, 50, 51, 52, 50, 51, 52, 53, 51, 52, 53, 54};
  // clang-format on

  static constexpr VecU32x16 SHIFT_MASK = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  static constexpr uint32_t AND_MASK = (1 << COMPRESS_BITS) - 1;

  template <size_t ITER>
  inline VecU32x16 decompress(VecU16x32 batch_lane) {
    auto shuffle_input = [&] {
      static_assert(ITER < 4, "Cannot do more than 4 iterations per lane.");
#if GCC_COMPILER
      return __builtin_shuffle(batch_lane, LANE_SHUFFLE_MASKS[ITER]);
#else
      // clang-format off
      switch (ITER) {
        case 0: return __builtin_shufflevector(batch_lane, batch_lane, 0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 12, 13);
        case 1: return __builtin_shufflevector(batch_lane, batch_lane, 9, 10, 11, 12, 13, 14, 15, 16, 11, 12, 13, 14, 15, 16, 17, 18, 13, 14, 15, 16, 17, 18, 19, 20, 15, 16, 17, 18, 19, 20, 21, 22);
        case 2: return __builtin_shufflevector(batch_lane, batch_lane, 18, 19, 20, 21, 22, 23, 24, 25, 20, 21, 22, 23, 24, 25, 26, 27, 22, 23, 24, 25, 26, 27, 28, 29, 24, 25, 26, 27, 28, 29, 30, 31);
        case 3: return __builtin_shufflevector(batch_lane, batch_lane, 27, 28, 29, 30, 31, 32, 33, 34, 29, 30, 31, 32, 33, 34, 35, 36, 31, 32, 33, 34, 35, 36, 37, 38, 33, 34, 35, 36, 37, 38, 39, 40);
        default: __builtin_unreachable();
          // clang-format on
      }
#endif
    };

    auto shuffle_lane = [&](VecU8x64& lane) {
#if GCC_COMPILER
      return __builtin_shuffle(lane, SHUFFLE_MASK);
#else
      // clang-format off
      return __builtin_shufflevector(lane, lane,
                                     0,  1,  2,  3,  1,  2,  3,  4,  2,  3,  4,  5,  3,  4,  5,  6,
                                     16, 17, 18, 19, 17, 18, 19, 20, 18, 19, 20, 21, 19, 20, 21, 22,
                                     32, 33, 34, 35, 33, 34, 35, 36, 34, 35, 36, 37, 35, 36, 37, 38,
                                     48, 49, 50, 51, 49, 50, 51, 52, 50, 51, 52, 53, 51, 52, 53, 54);
      // clang-format on
#endif
    };

    DEBUG_DO(std::cout << "load: "; print_lane(&batch_lane););

    VecU16x32 lane = shuffle_input();
    DEBUG_DO(std::cout << "a16 : "; print_lane(&lane););

    auto lane2 = shuffle_lane(reinterpret_cast<VecU8x64&>(lane));
    DEBUG_DO(std::cout << "a4  : "; print_lane(&lane2););

    auto lane3 = reinterpret_cast<VecU32x16&>(lane2) >> SHIFT_MASK;
    DEBUG_DO(std::cout << "bit : "; print_lane(&lane3););

    auto lane4 = lane3 & AND_MASK;
    DEBUG_DO(std::cout << "and : "; print_lane(&lane4););

    return lane4;
  }

  template <typename CallbackFn>
  inline void decompress_batch(VecU16x32 batch_lane, CallbackFn callback) {
    callback(decompress<0>(batch_lane));
    callback(decompress<1>(batch_lane));
    callback(decompress<2>(batch_lane));

    // Last iteration are only 8 values.
    auto final_lane = decompress<3>(batch_lane);
    callback(reinterpret_cast<VecU32x8&>(final_lane));
  }

  template <typename CallbackFn>
  void decompressor(const void* input_compressed, size_t num_tuples, CallbackFn callback) {
    const size_t num_batches = num_tuples / VALUES_PER_BATCH;
    const auto* compressed_data = static_cast<const uint8_t*>(input_compressed);

    for (size_t batch = 0; batch < num_batches; ++batch) {
      const size_t offset = batch * BYTES_PER_BATCH;
      auto batch_lane = *reinterpret_cast<const UnalignedVecU16x32*>(compressed_data + offset);
      decompress_batch(batch_lane, callback);
    }
  }

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    auto store_fn = [&]<typename Lane>(Lane decompressed_values) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
      if constexpr (std::is_same_v<Lane, VecU32x16>) {
        *reinterpret_cast<UnalignedVecU16x32*>(output) = decompressed_values;
        output += 16;
      } else if constexpr (std::is_same_v<Lane, VecU32x8>) {
        *reinterpret_cast<UnalignedVecU32x8*>(output) = decompressed_values;
        output += 8;
      }
#pragma GCC diagnostic pop
    };

    decompressor(input, num_tuples, store_fn);
  }
};

#define BM_ARGS Repetitions(1)->Unit(benchmark::kMicrosecond)

#if defined(__aarch64__)
BENCHMARK(BM_scanning<neon_scan>)->BM_ARGS;
#endif

#if defined(__x86_64__)
BENCHMARK(BM_scanning<x86_128_scan>)->BM_ARGS;
#endif

#if defined(AVX512_AVAILABLE)
BENCHMARK(BM_scanning<x86_512_scan>)->BM_ARGS;
#endif

BENCHMARK(BM_scanning<naive_scalar_scan>)->BM_ARGS;
BENCHMARK(BM_scanning<autovec_scalar_scan>)->BM_ARGS;
BENCHMARK(BM_scanning<vector_128_scan>)->BM_ARGS;
BENCHMARK(BM_scanning<vector_512_scan>)->BM_ARGS;

BENCHMARK_MAIN();