#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"
#include "simd.hpp"

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

  benchmark::DoNotOptimize(keys_to_hash.data());

  // Do one sanity check that we get the correct results.
  if (hash_fn(keys_to_hash, required_bits) != correct_hash_values) {
    throw std::runtime_error{"Bad hash calculation"};
  }

  for (auto _ : state) {
    const HashArray hashes = hash_fn(keys_to_hash, required_bits);
    benchmark::DoNotOptimize(hashes);
  }
}

#if defined(__aarch64__)
struct neon_hash {
  uint64x2_t multiply(uint64x2_t a, uint64x2_t b) {
    // b0Hi, b0Lo, b1Hi, b1Lo
    uint32x4_t b_hi_lo_swapped = vrev64q_u32(b);

    // a0Lo * b0Hi, a0Hi * b0Lo, a1Lo * b1Hi, a1Hi * b1Lo
    uint32x4_t product_hi_lo_pairs = vmulq_u32(a, b_hi_lo_swapped);

    // (a0Lo * b0Hi + a0Hi * b0Lo), (a1Lo * b1Hi + a1Hi * b1Lo)
    uint64x2_t hi_lo_pair_product_sums = vpaddlq_u32(product_hi_lo_pairs);

    // (a0Lo * b0Hi + a0Hi * b0Lo) << 32, (a1Lo * b1Hi + a1Hi * b1Lo) << 32
    uint64x2_t hi_lo_pair_product_sums_shifted = vshlq_n_u64(hi_lo_pair_product_sums, 32);

    // a0Lo, a1Lo
    uint32x2_t a_lo = vmovn_u64(a);

    // b0Lo, b1Lo
    uint32x2_t b_lo = vmovn_u64(b);

    // a0Lo * b0Lo + (a0Lo * b0Hi + a0Hi * b0Lo) << 32, a1Lo * b1Lo + (a1Lo * b1Hi + a1Hi * b1Lo) << 32
    return vmlal_u32(hi_lo_pair_product_sums_shifted, b_lo, a_lo);
  }

  using VecT = uint64x2_t;
  static constexpr size_t KEYS_PER_ITERATION = sizeof(VecT) / sizeof(KeyT);
  static_assert(NUM_KEYS % KEYS_PER_ITERATION == 0);

  HashArray operator()(const HashArray& keys_to_hash, uint64_t required_bits) {
    HashArray result;
    auto* typed_input_ptr = reinterpret_cast<const VecT*>(keys_to_hash.data());
    auto* typed_output_ptr = reinterpret_cast<VecT*>(result.data());

    VecT factor_vec = vdupq_n_u64(MULTIPLY_CONSTANT);
    uint64_t shift = 64 - required_bits;
    // right shift is not available with run-time values, so we left-shift by a negative offset
    VecT shift_vec = vdupq_n_u64(-shift);

    for (size_t i = 0; i < NUM_KEYS / KEYS_PER_ITERATION; ++i) {
      auto multiplied = multiply(typed_input_ptr[i], factor_vec);
      auto shifted = vshlq_u64(multiplied, shift_vec);
      typed_output_ptr[i] = shifted;
    }

    return result;
  }
};
BENCHMARK(BM_hashing<neon_hash>)->BM_ARGS;
#endif

#if defined(__x86_64__)
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

  HashArray operator()(const HashArray& keys_to_hash, uint64_t required_bits) {
    HashArray result;
    auto* typed_input_ptr = reinterpret_cast<const VecT*>(keys_to_hash.data());
    auto* typed_output_ptr = reinterpret_cast<VecT*>(result.data());

    VecT factor_vec = _mm_set1_epi64x(MULTIPLY_CONSTANT);
    uint64_t shift = 64 - required_bits;

    for (size_t i = 0; i < NUM_KEYS / KEYS_PER_ITERATION; ++i) {
      auto multiplied = multiply(typed_input_ptr[i], factor_vec);
      auto shifted = _mm_srli_epi64(multiplied, shift);
      typed_output_ptr[i] = shifted;
    }

    return result;
  }
};
BENCHMARK(BM_hashing<x86_128_hash>)->BM_ARGS;

struct x86_256_hash {
  using VecT = __m256i;
  static constexpr size_t KEYS_PER_ITERATION = sizeof(VecT) / sizeof(KeyT);
  static_assert(NUM_KEYS % KEYS_PER_ITERATION == 0);

  __m256i multiply(__m256i a, __m256i b) {
    // logic same as in https://github.com/vectorclass/version2/blob/master/vectori256.h#L3358-L3365
    // 8x32: 0, 0, 0, 0, 0, 0, 0, 0
    __m256i zero = _mm256_setzero_si256();

    // 8x32: b0Hi, b0Lo, b1Hi, b1Lo, b2Hi, b2Lo, b3Hi, b3Lo
    __m256i b_hi_lo_swapped = _mm256_shuffle_epi32(b, 0xB1);

    // 8x32: a0Lo * b0Hi, a0Hi * b0Lo, a1Lo * b1Hi, a1Hi * b1Lo, a2Lo * b2Hi, a2Hi * b2Lo, a3Lo * b3Hi, a3Hi * b3Lo
    __m256i product_hi_lo_pairs = _mm256_mullo_epi32(a, b_hi_lo_swapped);

    // 4x32: a0Lo b0Hi + a0Hi b0Lo, a1Lo b1Hi + a1Hi b1Lo, 0, 0, a2Lo b2Hi + a2Hi b2Lo, a3Lo b3Hi + a3Hi b3Lo, 0, 0
    __m256i hi_lo_pair_product_sums = _mm256_hadd_epi32(product_hi_lo_pairs, zero);

    // 4x32: 0, a0Lo b0Hi + a0Hi b0Lo, 0, a1Lo b1Hi + a1Hi b1Lo, 0, a2Lo b2Hi + a2Hi b2Lo, 0, a3Lo b3Hi + a3Hi b3Lo
    __m256i product_hi_lo_pairs_shuffled = _mm256_shuffle_epi32(hi_lo_pair_product_sums, 0x73);

    // 2x64: a0Lo * b0Lo, a1Lo * b1Lo, a2Lo * b2Lo, b3Lo * b3Lo
    __m256i product_lo_lo = _mm256_mul_epu32(a, b);

    // 4x64: a0Lo b0Lo + (a0Lo b0Hi + a0Hi b0Lo) << 32, a1Lo b1Lo + (a1Lo b1Hi + a1Hi b1Lo) << 32, ...
    return _mm256_add_epi64(product_lo_lo, product_hi_lo_pairs_shuffled);
  }

  HashArray operator()(const HashArray& keys_to_hash, uint64_t required_bits) {
    HashArray result;
    auto* typed_input_ptr = reinterpret_cast<const VecT*>(keys_to_hash.data());
    auto* typed_output_ptr = reinterpret_cast<VecT*>(result.data());

    VecT factor_vec = _mm256_set1_epi64x(MULTIPLY_CONSTANT);
    uint64_t shift = 64 - required_bits;

    for (size_t i = 0; i < NUM_KEYS / KEYS_PER_ITERATION; ++i) {
      auto multiplied = multiply(typed_input_ptr[i], factor_vec);
      auto shifted = _mm256_srli_epi64(multiplied, shift);
      typed_output_ptr[i] = shifted;
    }

    return result;
  }
};
BENCHMARK(BM_hashing<x86_256_hash>)->BM_ARGS;
#endif

#if AVX512_AVAILABLE
struct x86_512_hash {
  using VecT = __m512i;
  static constexpr size_t KEYS_PER_ITERATION = sizeof(VecT) / sizeof(KeyT);
  static_assert(NUM_KEYS % KEYS_PER_ITERATION == 0);

  HashArray operator()(const HashArray& keys_to_hash, uint64_t required_bits) {
    HashArray result;

    auto* typed_input_ptr = reinterpret_cast<const VecT*>(keys_to_hash.data());
    auto* typed_output_ptr = reinterpret_cast<VecT*>(result.data());

    VecT factor_vec = _mm512_set1_epi64(MULTIPLY_CONSTANT);
    uint64_t shift = 64 - required_bits;

    for (size_t i = 0; i < NUM_KEYS / KEYS_PER_ITERATION; ++i) {
      auto multiplied = _mm512_mullo_epi64(typed_input_ptr[i], factor_vec);
      auto shifted = _mm512_srli_epi64(multiplied, shift);
      typed_output_ptr[i] = shifted;
    }

    return result;
  }
};
BENCHMARK(BM_hashing<x86_512_hash>)->BM_ARGS;
#endif

struct naive_scalar_hash {
#if GCC_COMPILER
  __attribute__((optimize("no-tree-vectorize")))
#endif
  HashArray
  operator()(const HashArray& keys_to_hash, uint64_t required_bits) {
    HashArray result;
#if CLANG_COMPILER
#pragma clang loop vectorize(disable)
#endif
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      result[i] = calculate_hash(keys_to_hash[i], required_bits);
    }

    return result;
  }
};

struct autovec_scalar_hash {
  HashArray operator()(const HashArray& keys_to_hash, uint64_t required_bits) {
    HashArray result;
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      result[i] = calculate_hash(keys_to_hash[i], required_bits);
    }
    return result;
  }
};

template <size_t VECTOR_BITS>
struct vector_hash {
  static constexpr size_t VECTOR_BYTES = VECTOR_BITS / 8;
  static constexpr size_t NUM_VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(uint64_t);

  using VecT = typename simd::GccVec<uint64_t, VECTOR_BYTES>::T;
  static_assert(sizeof(VecT) == VECTOR_BYTES);

  HashArray operator()(const HashArray& keys_to_hash, uint64_t required_bits) {
    HashArray result;
    const VecT* vec_keys = reinterpret_cast<const VecT*>(keys_to_hash.data());

    auto hashes = reinterpret_cast<VecT*>(result.data());

    uint64_t shift = 64 - required_bits;
    for (size_t i = 0; i < NUM_KEYS / NUM_VECTOR_ELEMENTS; ++i) {
      hashes[i] = (vec_keys[i] * MULTIPLY_CONSTANT) >> shift;
    }
    return result;
  }
};

BENCHMARK(BM_hashing<naive_scalar_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<autovec_scalar_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_hash<64>>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_hash<128>>)->BM_ARGS;

// TODO: figure out why these are so fast
// Richard: On x86, clang generates different code for the 64-bit multiplication
// that is ~20% faster. I'd guess it's due to microarchitecture insight?
// Should we give this as a win to clang? If we were using agner fogs library,
// we wouldn't get that performance.
BENCHMARK(BM_hashing<vector_hash<256>>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_hash<512>>)->BM_ARGS;

BENCHMARK_MAIN();
