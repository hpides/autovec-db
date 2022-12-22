#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "common.hpp"
#include "benchmark/benchmark.h"

#define BM_ARGS Repetitions(1)->Arg(27)

// TODO(lawben): Check which number here makes sense. We need: #keys / (#vector-lanes / 8B key) registers.
//                 --> 128 keys = 16 zmm | 32 ymm | 64 xmm registers
//                 -->  64 keys =  8 zmm | 16 ymm | 32 xmm registers
//               This will impact if the compiler can unroll the loop or not.
static constexpr uint64_t NUM_KEYS = 64;

// Not yet consistently used.
using KeyT = uint64_t;

using HashArray = AlignedArray<KeyT, NUM_KEYS, 64>;

// Constant taken from https://github.com/rurban/smhasher
constexpr static uint64_t MULTIPLY_CONSTANT = 0x75f17d6b3588f843ull;

uint64_t calculate_hash(uint64_t key, uint64_t required_bits) {
  return (key * MULTIPLY_CONSTANT) >> (64 - required_bits);
}

/*
 Some instruction sets can't multiply vectors of 64-bit ints and truncate the result elements to 64 bits (AVX2, NEON).
 There, we multiply in the 32bit domain, using this logic to compute A * B with A, B being 64 bit numbers

 A * B = (A_Hi * 2^32 + A_Lo) * (B_Hi * 2^32 + B_Lo)
 = (A_Hi * B_Hi * 2^32 * 2^32) + (A_Hi * 2^32 * B_Lo) + (A_Lo * B_Hi * 2^32) + (A_Lo * B_Lo)

  (A_Hi * B_Hi * 2^64) can be discarded, 64 lowest-value bits are 0.
  ->
 = (A_Hi * B_Lo * 2^32) + (A_Lo * B_Hi * 2^32) + (A_Lo * B_Lo)
 = ((A_Hi * B_Lo) + (A_Lo * B_Hi)) * 2^32 + (A_Lo * B_Lo)

 This is also what Agner Fog's vector library uses if no native instruction is available:
 https://github.com/vectorclass/version2/blob/master/vectori128.h#L4062-L4081
*/

template <typename HashFn>
void BM_hashing(benchmark::State& state) {
  HashFn hash_fn{};
  uint64_t required_bits = state.range(0);

  // Seed rng for same benchmark runs.
  std::mt19937_64 rng{345873456};

  HashArray keys_to_hash;
  HashArray correct_hash_values;
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    keys_to_hash[i] = rng();
    correct_hash_values[i] = calculate_hash(keys_to_hash[i], required_bits);
  }

  HashArray hashes{};

  // Do one sanity check that we get the correct results.
  hash_fn(keys_to_hash, required_bits, &hashes);
  if (hashes != correct_hash_values) {
    throw std::runtime_error{"Bad hash calculation"};
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(keys_to_hash.data());
    hash_fn(keys_to_hash, required_bits, &hashes);
    benchmark::DoNotOptimize(hashes);
  }
}

#if defined(__aarch64__)
#include <arm_neon.h>

/** Doing this in NEON is not very useful, as we can neither do a vector multiply nor a variable right shift. */
struct neon_hash {
  void operator()(const HashArray& keys_to_hash, uint64_t required_bits, HashArray* __restrict result) {
    // TODO(Richard): Unify with x86_128 implementation
    static_assert(NUM_KEYS % 2 == 0);
    using VecArray = std::array<uint64x2_t, NUM_KEYS / 2>;

    auto& hashes = reinterpret_cast<VecArray&>(*result);

    // We need this "hack" here, as NEON requires the right shift value to be a compile-time constant. So we shift left
    // with a negative value.
    int64_t shift_by = -(64 - static_cast<int64_t>(required_bits));
    uint64x2_t shift_value = vmovq_n_u64(shift_by);

    for (size_t i = 0; i < NUM_KEYS / 2; ++i) {
      const size_t offset = i * 2;
      alignas(16) std::array<uint64_t, 2> keys{keys_to_hash[offset] * MULTIPLY_CONSTANT,
                                               keys_to_hash[offset + 1] * MULTIPLY_CONSTANT};
      uint64x2_t multiplied_keys = vld1q_u64(keys.data());
      hashes[i] = vshlq_u64(multiplied_keys, shift_value);
    }
  }
};

#elif defined(__x86_64__)

#include <immintrin.h>
struct x86_128_hash {
  using VecT = __m128i;
  static constexpr size_t KEYS_PER_ITERATION = sizeof(VecT) / sizeof(KeyT);
  static_assert(NUM_KEYS % KEYS_PER_ITERATION == 0);

  __m128i multiply(__m128i a, __m128i b) {
    // logic same as in https://github.com/vectorclass/version2/blob/master/vectori128.h#L4062-L4081
    // 4x32: 0, 0, 0, 0
    __m128i zero = _mm_setzero_si128();

    // 4x32: b0Hi, b0Lo, b1Hi, b1Lo
    __m128i b_hi_lo_swapped = _mm_shuffle_epi32(b, 0xB1);

    // 4x32: a0Lo * b0Hi, a0Hi * b0Lo, a1Lo * b1Hi, a1Hi * b1Lo
    __m128i product_hi_lo_pairs = _mm_mullo_epi32(a, b_hi_lo_swapped);

    // 4x32: a0Lo * b0Hi + a0Hi * b0Lo, a1Lo * b1Hi + a1Hi * b1Lo, 0, 0
    __m128i hi_lo_pair_product_sums = _mm_hadd_epi32(product_hi_lo_pairs, zero);

    // 4x32: 0, a0Lo * b0Hi + a0Hi * b0Lo, 0, a1Lo * b1Hi + a1Hi * b1Lo
    __m128i product_hi_lo_pairs_shuffled = _mm_shuffle_epi32(hi_lo_pair_product_sums, 0x73);

    // 2x64: a0Lo * b0Lo, a1Lo * b1Lo
    __m128i product_lo_lo = _mm_mul_epu32(a, b);

    // 2x64: a0Lo * b0Lo + (a0Lo * b0Hi + a0Hi * b0Lo) << 32, a1Lo * b1Lo + (a1Lo * b1Hi + a1Hi * b1Lo) << 32
    return _mm_add_epi64(product_lo_lo, product_hi_lo_pairs_shuffled);
  }

  void operator()(const HashArray& keys_to_hash, uint64_t required_bits, HashArray* __restrict result) {
    auto* typed_input_ptr = reinterpret_cast<const VecT*>(keys_to_hash.data());
    auto* typed_output_ptr = reinterpret_cast<VecT*>(result->data());

    VecT factor_vec = _mm_set1_epi64x(MULTIPLY_CONSTANT);
    uint64_t shift = 64 - required_bits;

    for (size_t i = 0; i < NUM_KEYS / KEYS_PER_ITERATION; ++i) {
      auto multiplied = multiply(typed_input_ptr[i], factor_vec);
      auto shifted = _mm_srli_epi64(multiplied, shift);
      typed_output_ptr[i] = shifted;
    }
  }
};
BENCHMARK(BM_hashing<x86_128_hash>)->BM_ARGS;

#if defined(AVX512_AVAILABLE)
struct x86_512_hash {
  using VecT = __m512i;
  static constexpr size_t KEYS_PER_ITERATION = sizeof(VecT) / sizeof(KeyT);
  static_assert(NUM_KEYS % KEYS_PER_ITERATION == 0);

  void operator()(const HashArray& keys_to_hash, uint64_t required_bits, HashArray* __restrict result) {
    auto* typed_input_ptr = reinterpret_cast<const VecT*>(keys_to_hash.data());
    auto* typed_output_ptr = reinterpret_cast<VecT*>(result->data());

    VecT factor_vec = _mm512_set1_epi64(MULTIPLY_CONSTANT);
    uint64_t shift = 64 - required_bits;

    for (size_t i = 0; i < NUM_KEYS / KEYS_PER_ITERATION; ++i) {
      auto multiplied = _mm512_mullo_epi64(typed_input_ptr[i], factor_vec);
      auto shifted = _mm512_srli_epi64(multiplied, shift);
      typed_output_ptr[i] = shifted;
    }
  }
};
BENCHMARK(BM_hashing<x86_512_hash>)->BM_ARGS;
#endif

#endif

struct
#if GCC_COMPILER
__attribute__((optimize("no-tree-vectorize")))
#endif
naive_scalar_hash {
  void operator()(const HashArray& keys_to_hash, uint64_t required_bits, HashArray* __restrict result) {
#pragma clang loop vectorize(disable)
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      (*result)[i] = calculate_hash(keys_to_hash[i], required_bits);
    }
  }
};

struct autovec_scalar_hash {
  void operator()(const HashArray& keys_to_hash, uint64_t required_bits, HashArray* __restrict result) {
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      (*result)[i] = calculate_hash(keys_to_hash[i], required_bits);
    }
  }
};

template <size_t VECTOR_BITS>
struct vector_hash {
  static constexpr size_t VECTOR_BYTES = VECTOR_BITS / 8;
  static constexpr size_t NUM_VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(uint64_t);

  using VecT __attribute__((vector_size(VECTOR_BYTES))) = uint64_t;
  static_assert(sizeof(VecT) == VECTOR_BYTES);

  using VecArray = std::array<VecT, NUM_KEYS / NUM_VECTOR_ELEMENTS>;

  void operator()(const HashArray& keys_to_hash, uint64_t required_bits, HashArray* __restrict result) {
    const auto& vec_keys = reinterpret_cast<const VecArray&>(keys_to_hash);

    auto& hashes = reinterpret_cast<VecArray&>(*result);

    uint64_t shift = 64 - required_bits;
    for (size_t i = 0; i < NUM_KEYS / NUM_VECTOR_ELEMENTS; ++i) {
      hashes[i] = (vec_keys[i] * MULTIPLY_CONSTANT) >> shift;
    }
  }
};

#if defined(__aarch64__)
BENCHMARK(BM_hashing<neon_hash>)->BM_ARGS;
#endif

BENCHMARK(BM_hashing<naive_scalar_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<autovec_scalar_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_hash<64>>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_hash<128>>)->BM_ARGS;

// TODO: figure out why these are wrong or why they are so fast!
BENCHMARK(BM_hashing<vector_hash<256>>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_hash<512>>)->BM_ARGS;

BENCHMARK_MAIN();
