#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "benchmark/benchmark.h"
#include "common.hpp"
#include "simd.hpp"

// 128-bit vec needs 12 values, 512-bit vec needs 56 values.
// For autovec, processing 32-element blocks seems reasonable.
static constexpr uint64_t NUM_BASE_TUPLES = std::lcm(12, std::lcm(56, 32));

static constexpr uint64_t NUM_TARGET_TUPLES = 1'000'000;
static constexpr uint64_t SCALE_FACTOR = NUM_TARGET_TUPLES / NUM_BASE_TUPLES;

static constexpr uint64_t NUM_TUPLES = NUM_BASE_TUPLES * SCALE_FACTOR;
static constexpr size_t COMPRESS_BITS = 9;

#define BM_ARGS Unit(benchmark::kMicrosecond)

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

  for (const uint32_t input_number : input) {
    const uint64_t val = input_number & MASK;
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
  constexpr size_t NUM_BYTES = sizeof(SimdLane);
  if (as_bits) {
    return print_bits_right_to_left(lane, NUM_BYTES, os);
  }
  return print_bytes_right_to_left(lane, NUM_BYTES, os);
}

}  // namespace

template <typename ScanFn>
void BM_scanning(benchmark::State& state) {
  ScanFn scan_fn{};

  // Seed rng for same benchmark runs.
  std::mt19937 rng{std::random_device{}()};

  std::vector<uint32_t> tuples(NUM_TUPLES);
  for (size_t i = 0; i < NUM_TUPLES; ++i) {
    tuples[i] = rng() & ((1u << COMPRESS_BITS) - 1);
    //    tuples[i] = 42;  // for debugging
  }

  CompressedColumn compressed_column = compress_input(tuples);
  DecompressedColumn decompressed_column{NUM_TUPLES + 64};

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

  state.counters["PerValue"] = benchmark::Counter(static_cast<double>(state.iterations()) * NUM_TUPLES,
                                                  benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

///////////////////////
///      NAIVE      ///
///////////////////////
struct naive_scan {
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

BENCHMARK(BM_scanning<naive_scan>)->BM_ARGS;

///////////////////////
///    AUTOVEC      ///
///////////////////////
struct autovec_scan {
  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    static_assert(std::endian::native == std::endian::little,
                  "big-endian systems need extra logic to handle uint64_t boundaries correctly.");
    const auto* __restrict input_bytes = reinterpret_cast<const std::byte*>(input);

    constexpr size_t BATCH_SIZE_NUMBERS = 32;
    static_assert(9 * BATCH_SIZE_NUMBERS % 8 == 0, "Autovec approach needs a batch to always start byte-aligned");
    constexpr size_t BATCH_SIZE_BYTES = BATCH_SIZE_NUMBERS * 9 / 8;

    const size_t num_batches = num_tuples / BATCH_SIZE_NUMBERS;

    for (size_t batch = 0; batch < num_batches; ++batch) {
      const std::byte* __restrict batch_begin = input_bytes + batch * BATCH_SIZE_BYTES;

      for (size_t number_in_batch = 0; number_in_batch < BATCH_SIZE_NUMBERS; ++number_in_batch) {
        const size_t start_bit_in_batch = 9 * number_in_batch;
        const size_t byte_to_start_copying_from_in_batch = start_bit_in_batch / 8;
        const size_t leading_garbage_bits = start_bit_in_batch - 8 * byte_to_start_copying_from_in_batch;

        uint32_t value = 0;
        std::memcpy(&value, batch_begin + byte_to_start_copying_from_in_batch, sizeof(value));

        value = value >> leading_garbage_bits;
        value = value & ((1 << COMPRESS_BITS) - 1);

        output[batch * BATCH_SIZE_NUMBERS + number_in_batch] = value;
      }
    }
  }
};

BENCHMARK(BM_scanning<autovec_scan>)->BM_ARGS;

///////////////////////
///     VECTOR      ///
///////////////////////

/*
The vector variants all follow this approach:

* Load register-width-many-bytes of input data into vector register. This may contain up to 7 leading garbage bits, and
  thus have (regwidth)/9 or (regwidth - 7) / 9 many complete elements.
* Iterate over the values in the input vector. Each iteration outputs at max regwidth/32 elements. We need (regwidth /
  9) / (regwidth/32) = 32/9 = 3.5 iterations per input line.
* We want to use aligned stores for our output, so we write 3 full vectors and discard the remaining half row.

An iteration is repeated 3 times and performs these steps:
 1. Widen elements from 9 bit to 32 bit:
   1. Shuffle consecutive 32 bits from the input into 32-bit "lanes", each lane starting with the first byte to contain
      a new 9-bit element

   2. Eliminate leading garbage bits by right-shifting the 32-bit lanes by incrementing numbers.
     * For this, we have a shift vector that is {0, 1, 2, 3, ...}, which aligns all values
       (since lane N+1 has 1 more leading garbage bit than lane N)
     * The first lane may not yet be perfectly aligned, but have N leading garbage bits. In this case, we
       additionally need to right-shift each lane by N.
       * Overall, each individual value needs to be shifted by a MOD-8 value, and the initial garbage is also a MOD-8
         value, so we have a maximum shift of 2*7=14 bits. This is fine, since we're in 32-bit elements, so even with 14
         leading garbage bits we have 18 bits left, so no slicing happens.
       * With vector sizes 256 and above, using the aligned store approach, this doesn't happen, as an output register
         contains multiples of 8 values, so we always write 8N values, so we also always process a byte-aligned amount
         of input (8N * 9 == 0 mod 8).

   3. Eliminate trailing garbage bits by AND-masking with a constant (keep 9 least significant bits per lane).

 2. Write the output
*/

template <size_t vector_width_bits>
static constexpr auto shuffle_table_input_elements_to_lanes() {
  // Given 9-bit packed input integers, provide the shuffles required to shuffle the input elements into 32 bit lanes,
  // with leading / trailing garbage bits. 4 shuffles are required since we expand 9-bit numbers to 32-bit numbers, so a
  // vector register can only contain 9/32 = 0.28 of the input numbers, so we have to shuffle 4 times to process all
  // input numbers.
  std::array<std::array<uint8_t, vector_width_bits / 8>, 4> result{};

  constexpr size_t INPUT_ELEMENTS_PER_VECTOR = vector_width_bits / 9;
  constexpr size_t OUTPUT_ELEMENTS_PER_VECTOR = vector_width_bits / 32;

  for (size_t input_element = 0; input_element < INPUT_ELEMENTS_PER_VECTOR; ++input_element) {
    const size_t bits_before = 9 * input_element;
    const size_t bytes_before = bits_before / 8;

    const size_t output_array = input_element / OUTPUT_ELEMENTS_PER_VECTOR;
    const size_t output_element = input_element % OUTPUT_ELEMENTS_PER_VECTOR;

    unsigned char* write_ptr = result[output_array].data() + 4 * output_element;
    std::iota(write_ptr, write_ptr + 4, static_cast<uint8_t>(bytes_before));
  }

  return result;
}

template <size_t vector_width_bits>
static constexpr auto shift_values_for_lanes() {
  // Given integers shuffled into lanes, get shift offsets per lane to shift out leading garbage bits (assuming the
  // first element has 0 leading garbage bits)

  std::array<uint32_t, vector_width_bits / 8 / sizeof(uint32_t)> result{};
  std::iota(result.begin(), result.end(), 0);
  for (auto& el : result) {
    el %= 8;
  }
  return result;
}

template <size_t vector_width_bits>
struct vector_scan {
  static constexpr size_t VECTOR_WIDTH_BYTES = vector_width_bits / 8;
  static constexpr size_t INPUT_ELEMENTS_PER_VECTOR = vector_width_bits / 9;
  static constexpr size_t OUTPUT_ELEMENTS_PER_VECTOR = vector_width_bits / 32;

  using ByteVecT = typename simd::GccVec<unsigned char, VECTOR_WIDTH_BYTES>::T;
  using Uint32VecT = typename simd::GccVec<uint32_t, VECTOR_WIDTH_BYTES>::T;

  alignas(VECTOR_WIDTH_BYTES) static constexpr std::array SHUFFLE_TO_LANES =
      shuffle_table_input_elements_to_lanes<vector_width_bits>();
  alignas(VECTOR_WIDTH_BYTES) static constexpr std::array LANE_SHIFT_VALUES =
      shift_values_for_lanes<vector_width_bits>();

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    static_assert(std::endian::native == std::endian::little,
                  "big-endian systems need extra logic to handle uint64_t boundaries correctly.");

    constexpr size_t ADDITIONAL_GARBAGE_BITS_PER_SUBVECTOR = (9 * OUTPUT_ELEMENTS_PER_VECTOR) % 8;
    constexpr size_t ADDITIONAL_GARBAGE_BITS_PER_BATCH = (3 * ADDITIONAL_GARBAGE_BITS_PER_SUBVECTOR) % 8;

    // lowest power of two N with (N * ADDITIONAL_GARBAGE_BITS_PER_BATCH) % 8 == 0
    constexpr size_t BATCHES_PER_ITERATION = [&]() -> size_t {
      if constexpr (ADDITIONAL_GARBAGE_BITS_PER_BATCH != 0) {
        static_assert(8 % ADDITIONAL_GARBAGE_BITS_PER_BATCH == 0, "Computation would be wrong here");
        return 8 / ADDITIONAL_GARBAGE_BITS_PER_BATCH;
      }
      return 1;
    }();

    static_assert((ADDITIONAL_GARBAGE_BITS_PER_BATCH * BATCHES_PER_ITERATION) % 8 == 0, "Iteration end not aligned");
    // We write 3 full output vectors and discards the remaining ~half output values.
    constexpr size_t OUTPUT_ELEMENTS_PER_ITERATION = BATCHES_PER_ITERATION * 3 * OUTPUT_ELEMENTS_PER_VECTOR;
    static_assert(NUM_TUPLES % OUTPUT_ELEMENTS_PER_ITERATION == 0, "Would require loop epilogue");

    const size_t iterations = num_tuples / OUTPUT_ELEMENTS_PER_ITERATION;

    const auto* read_ptr = reinterpret_cast<const std::byte*>(input);

    // This approach works fine for 128 and 256 vector width, but it compiles to terrible code on 512-bit platforms
    // < cannonlake. May be related that without AVX512VBMI2 (>=icelake), the byte-level shuffle is not supported,
    // so we have to fall back to cross-lane-16bit-shuffle followed by in-lane-8-bit-shuffle with x86_512. This was
    // also the approach of the old vector-512 implementation.
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
      for (size_t batch = 0; batch < BATCHES_PER_ITERATION; ++batch) {
        auto input_vec = simd::load_unaligned<ByteVecT>(read_ptr);
        read_ptr += 3 * (OUTPUT_ELEMENTS_PER_VECTOR)*9 / 8;

#if CLANG_COMPILER
        // We want the shuffle mask to be known at compile-time on NEON, so we force unroll here.
        // If we don't force the unroll, it does not detect this. For 512 bits, the unrolled loop is too big, so the
        // vectorizer fails, and we get horrible code --> only unroll for smaller vectors.
        constexpr int UNROLL_COUNT = vector_width_bits >= 512 ? 1 : 3;
        #pragma clang loop unroll_count(UNROLL_COUNT)
#endif
        for (size_t i = 0; i < 3; ++i) {
          auto shuffle_mask = simd::load<ByteVecT>(SHUFFLE_TO_LANES[i].data());
          Uint32VecT lanes = simd::shuffle_vector(input_vec, shuffle_mask);

          auto shift_values = simd::load<Uint32VecT>(LANE_SHIFT_VALUES.data());
          lanes >>= shift_values;
          lanes >>= batch * ADDITIONAL_GARBAGE_BITS_PER_BATCH + ((i * ADDITIONAL_GARBAGE_BITS_PER_SUBVECTOR) % 8);

          lanes &= ((1 << COMPRESS_BITS) - 1);

          simd::store(output, lanes);
          output += OUTPUT_ELEMENTS_PER_VECTOR;
        }
      }

      if constexpr (BATCHES_PER_ITERATION != 1) {
        // If we have multiple batches per iteration, we do that because a single iteration builds up leading garbage
        // bits, so the batches build up a full byte of garbage.
        read_ptr++;
      }
    }
  }
};
BENCHMARK(BM_scanning<vector_scan<128>>)->BM_ARGS;
BENCHMARK(BM_scanning<vector_scan<256>>)->BM_ARGS;
BENCHMARK(BM_scanning<vector_scan<512>>)->BM_ARGS;

///////////////////////
///      NEON       ///
///////////////////////
#if defined(__aarch64__)
struct neon_scan {
  static constexpr size_t OUTPUT_ELEMENTS_PER_VECTOR = 4;  // 16B / 4B-per-element
  static constexpr size_t OUTPUT_ELEMENTS_PER_ITERATION = 3ul * 2 * OUTPUT_ELEMENTS_PER_VECTOR;

  static constexpr uint32x4_t BYTE_ALIGN_MASK = {3, 2, 1, 0};
  static constexpr uint8x16_t AND_MASK = {255, 1, 0, 0, 255, 1, 0, 0, 255, 1, 0, 0, 255, 1, 0, 0};

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    const auto* read_ptr = reinterpret_cast<const uint8_t*>(input);

    auto shuffle_input = [](uint8x16_t vec, size_t iter) {
      assert(iter < 3 && "Cannot do more than 3 iterations per lane.");
      // clang-format off
      switch (iter) {
        case 0: return __builtin_shufflevector(vec, vec, 0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6);
        case 1: return __builtin_shufflevector(vec, vec, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10);
        case 2: return __builtin_shufflevector(vec, vec, 8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14);
        default: __builtin_unreachable();
      }
      // clang-format on
    };

    const size_t num_iterations = num_tuples / OUTPUT_ELEMENTS_PER_ITERATION;
    for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
      uint8x16_t input_vec = vld1q_u8(read_ptr);

      for (int i = 0; i < 3; ++i) {
        uint8x16_t vec = shuffle_input(input_vec, i);
        vec = vshlq_u32(vreinterpretq_s32_u8(vec), BYTE_ALIGN_MASK);
        vec = vshlq_s32(vreinterpretq_u32_u8(vec), vmovq_n_s32(-(3 + (i * 4))));
        vec = vandq_u8(vec, AND_MASK);
        vst1q_u32(output, vec);
        output += OUTPUT_ELEMENTS_PER_VECTOR;
      }

      // For the first 128 bits, we will output 3*4=12 elements, processing 12*9=108 bits=13.5 bytes.
      // For the second vector, we have to read beginning at the "dangling" half byte.
      input_vec = vld1q_u8(read_ptr + 13);

      for (int i = 0; i < 3; ++i) {
        uint8x16_t vec = shuffle_input(input_vec, i);
        vec = vshlq_u32(vreinterpretq_s32_u8(vec), BYTE_ALIGN_MASK);
        vec = vshlq_s32(vreinterpretq_u32_u8(vec), vmovq_n_s32(-(3 + (i * 4) + 4)));
        vec = vandq_u8(vec, AND_MASK);
        vst1q_u32(output, vec);
        output += OUTPUT_ELEMENTS_PER_VECTOR;
      }

      read_ptr += OUTPUT_ELEMENTS_PER_ITERATION * 9 / 8;
    }
  }
};

BENCHMARK(BM_scanning<neon_scan>)->BM_ARGS;
#endif

///////////////////////
///       x86       ///
///////////////////////
#if defined(__x86_64__)

struct x86_128_scan {
  static constexpr size_t OUTPUT_ELEMENTS_PER_VECTOR = 4;  // 16B / 4B-per-element
  static constexpr size_t OUTPUT_ELEMENTS_PER_ITERATION = 3ul * 2 * OUTPUT_ELEMENTS_PER_VECTOR;

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    const std::array shuffle_masks = {
        _mm_set_epi8(6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0),
        _mm_set_epi8(10, 9, 8, 7, 9, 8, 7, 6, 8, 7, 6, 5, 7, 6, 5, 4),
        _mm_set_epi8(14, 13, 12, 11, 13, 12, 11, 10, 12, 11, 10, 9, 11, 10, 9, 8),
    };

    const __m128i byte_align_mask = _mm_set_epi32(1 << 0, 1 << 1, 1 << 2, 1 << 3);
    const __m128i and_mask = _mm_set1_epi32((1 << COMPRESS_BITS) - 1);

    const auto* read_ptr = reinterpret_cast<const std::byte*>(input);

    const size_t num_iterations = num_tuples / OUTPUT_ELEMENTS_PER_ITERATION;
    for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
      __m128i input_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(read_ptr));

      for (int i = 0; i < 3; ++i) {
        __m128i lane = _mm_shuffle_epi8(input_vec, shuffle_masks[i]);
        lane = _mm_mullo_epi32(lane, byte_align_mask);  // _mm_sllv_epi32 requires AVX2
        lane = _mm_srli_epi32(lane, 3 + (i * 4));
        lane = _mm_and_si128(lane, and_mask);
        _mm_store_si128(reinterpret_cast<__m128i*>(output), lane);
        output += OUTPUT_ELEMENTS_PER_VECTOR;
      }

      // For the first 128 bits, we will output 3*4=12 elements, processing 12*9=108 bits=13.5 bytes.
      // For the second vector, we have to read beginning at the the "dangling" half byte.
      input_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(read_ptr + 13));

      for (int i = 0; i < 3; ++i) {
        __m128i lane = _mm_shuffle_epi8(input_vec, shuffle_masks[i]);
        lane = _mm_mullo_epi32(lane, byte_align_mask);
        lane = _mm_srli_epi32(lane, 3 + (i * 4) + 4);
        lane = _mm_and_si128(lane, and_mask);
        _mm_store_si128(reinterpret_cast<__m128i*>(output), lane);
        output += OUTPUT_ELEMENTS_PER_VECTOR;
      }

      read_ptr += OUTPUT_ELEMENTS_PER_ITERATION * 9 / 8;
    }
  }
};

BENCHMARK(BM_scanning<x86_128_scan>)->BM_ARGS;

struct x86_avx2_scan {
  static constexpr size_t OUTPUT_ELEMENTS_PER_128_VECTOR = 4;  // 16B / 4B-per-element
  static constexpr size_t OUTPUT_ELEMENTS_PER_ITERATION = 3ul * 2 * OUTPUT_ELEMENTS_PER_128_VECTOR;

  alignas(16) static constexpr std::array SHUFFLE_TO_LANES = shuffle_table_input_elements_to_lanes<128>();
  alignas(16) static constexpr std::array LANE_SHIFT_VALUES = shift_values_for_lanes<128>();

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    const size_t iterations = num_tuples / OUTPUT_ELEMENTS_PER_ITERATION;

    const auto* read_ptr = reinterpret_cast<const std::byte*>(input);

    const __m128i shift_values = _mm_load_si128(reinterpret_cast<const __m128i*>(LANE_SHIFT_VALUES.data()));
    const __m128i and_mask = _mm_set1_epi32((1 << COMPRESS_BITS) - 1);

    for (size_t iteration = 0; iteration < iterations; ++iteration) {
      // With AVX2, we can't shuffle across 128bit lanes, so we fall back to using two 128-bit registers.
      __m128i input_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(read_ptr));

      for (size_t i = 0; i < 3; ++i) {
        const __m128i shuffle_mask = _mm_load_si128(reinterpret_cast<const __m128i*>(SHUFFLE_TO_LANES[i].data()));
        const __m128i lanes = _mm_shuffle_epi8(input_vec, shuffle_mask);
        __m128i shifted_lanes = _mm_srlv_epi32(lanes, shift_values);
        shifted_lanes = _mm_srli_epi32(shifted_lanes, static_cast<int>(i * 4) % 8);
        const __m128i result = _mm_and_si128(shifted_lanes, and_mask);
        _mm_store_si128(reinterpret_cast<__m128i*>(output), result);
        output += OUTPUT_ELEMENTS_PER_128_VECTOR;
      }

      // For the first 128 bits, we will output 3*4=12 elements, processing 12*9=108 bits=13.5 bytes.
      // For the second vector, we have to read beginning at the "dangling" half byte.
      input_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(read_ptr + 13));

      for (size_t i = 0; i < 3; ++i) {
        const __m128i shuffle_mask = _mm_load_si128(reinterpret_cast<const __m128i*>(SHUFFLE_TO_LANES[i].data()));
        const __m128i lanes = _mm_shuffle_epi8(input_vec, shuffle_mask);
        __m128i shifted_lanes = _mm_srlv_epi32(lanes, shift_values);
        shifted_lanes = _mm_srli_epi32(shifted_lanes, 4 + static_cast<int>(i * 4) % 8);
        const __m128i result = _mm_and_si128(shifted_lanes, and_mask);
        _mm_store_si128(reinterpret_cast<__m128i*>(output), result);
        output += OUTPUT_ELEMENTS_PER_128_VECTOR;
      }

      read_ptr += OUTPUT_ELEMENTS_PER_ITERATION * 9 / 8;
    }
  }
};
BENCHMARK(BM_scanning<x86_avx2_scan>)->BM_ARGS;

struct x86_pdep_scan {
  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    // approach as found in
    // https://github.com/facebookincubator/velox/blob/3d820e92a399e60c867990b6ac94e6f518e8d9af/velox/dwio/common/BitPackDecoder.h#L315-L347
    const auto* __restrict input_bytes = reinterpret_cast<const std::byte*>(input);

    constexpr uint64_t PDEP_STORE_MASK = 0x000001ff000001ff;

    // process 8 values (8 * 9 bits = 72bit = 9 Bytes) at once
    static_assert(NUM_TUPLES % 8 == 0);
    for (size_t tuple_index = 0; tuple_index < num_tuples; tuple_index += 8) {
      const size_t read_start_byte = tuple_index * 9 / 8;

      uint64_t lower_8_bytes = 0;
      std::memcpy(&lower_8_bytes, input_bytes + read_start_byte, sizeof(lower_8_bytes));

      uint64_t upper_byte = 0;
      std::memcpy(&upper_byte, input_bytes + read_start_byte + 8, 1);

      std::array<uint64_t, 4> decompressed_values{};
      decompressed_values[0] = _pdep_u64(lower_8_bytes >> (0 * 9), PDEP_STORE_MASK);
      decompressed_values[1] = _pdep_u64(lower_8_bytes >> (2 * 9), PDEP_STORE_MASK);
      decompressed_values[2] = _pdep_u64(lower_8_bytes >> (4 * 9), PDEP_STORE_MASK);
      // 6*9=54, so we have 64-54=10 bits left in lower_8_bytes, and we need to "append" the 8 bits of the next byte
      decompressed_values[3] = _pdep_u64(lower_8_bytes >> (6 * 9) | (upper_byte << (64 - 6 * 9)), PDEP_STORE_MASK);

      std::memcpy(output + tuple_index, decompressed_values.data(), sizeof(decompressed_values));
    }
  }
};

BENCHMARK(BM_scanning<x86_pdep_scan>)->BM_ARGS;
#endif

#if AVX512_AVAILABLE
struct x86_512_scan {
  static constexpr size_t OUTPUT_ELEMENTS_PER_VECTOR = 512 / 32;
  static constexpr size_t OUTPUT_ELEMENTS_PER_ITERATION = 3ul * OUTPUT_ELEMENTS_PER_VECTOR;

  void operator()(const uint64_t* __restrict input, uint32_t* __restrict output, size_t num_tuples) {
    // clang-format off
    const std::array lane_shuffle_masks = {
      _mm512_set_epi16(13, 12, 11, 10, 9, 8, 7, 6, 11, 10, 9, 8, 7, 6, 5, 4, 9, 8, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2, 1, 0),
      _mm512_set_epi16(22, 21, 20, 19, 18, 17, 16, 15, 20, 19, 18, 17, 16, 15, 14, 13, 18, 17, 16, 15, 14, 13, 12, 11, 16, 15, 14, 13, 12, 11, 10, 9),
      _mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 29, 28, 27, 26, 25, 24, 23, 22, 27, 26, 25, 24, 23, 22, 21, 20, 25, 24, 23, 22, 21, 20, 19, 18),
      _mm512_set_epi16(40, 39, 38, 37, 36, 35, 34, 33, 38, 37, 36, 35, 34, 33, 32, 31, 36, 35, 34, 33, 32, 31, 30, 29, 34, 33, 32, 31, 30, 29, 28, 27)
    };

    const __m512i shuffle_mask = _mm512_set_epi8(6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0,
                                                 6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0,
                                                 6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0,
                                                 6, 5, 4, 3, 5, 4, 3, 2, 4, 3, 2, 1, 3, 2, 1, 0);
    // clang-format on

    const __m512i shift_mask = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i and_mask = _mm512_set1_epi32((1u << COMPRESS_BITS) - 1);

    const size_t num_iterations = num_tuples / OUTPUT_ELEMENTS_PER_ITERATION;
    const auto* read_ptr = reinterpret_cast<const std::byte*>(input);

    for (size_t iteration = 0; iteration < num_iterations; ++iteration) {
      const __m512i input_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(read_ptr));
      read_ptr += OUTPUT_ELEMENTS_PER_ITERATION * 9 / 8;
      for (int i = 0; i < 3; ++i) {
        __m512i lane = _mm512_permutexvar_epi16(lane_shuffle_masks[i], input_vec);
        lane = _mm512_shuffle_epi8(lane, shuffle_mask);
        lane = _mm512_srlv_epi32(lane, shift_mask);
        lane = _mm512_and_epi32(lane, and_mask);
        _mm512_store_si512(reinterpret_cast<__m512i*>(output), lane);
        output += OUTPUT_ELEMENTS_PER_VECTOR;
      }
    }
  }
};

BENCHMARK(BM_scanning<x86_512_scan>)->BM_ARGS;
#endif

BENCHMARK_MAIN();
