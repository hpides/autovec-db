#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"

#define BM_ARGS Repetitions(1)->Arg(27)

// TODO(lawben): Check which number here makes sense. We need: #keys / (#vector-lanes / 8B key) registers.
//                 --> 128 keys = 16 zmm | 32 ymm | 64 xmm registers
//                 -->  64 keys =  8 zmm | 16 ymm | 32 xmm registers
//               This will impact if the compiler can unroll the loop or not.
static constexpr uint64_t NUM_KEYS = 64;

template <typename Type, size_t LENGTH, size_t ALIGN = 64>
using BaseAlignedArray __attribute__((aligned(ALIGN))) = std::array<Type, LENGTH>;

using AlignedArray = BaseAlignedArray<uint64_t, NUM_KEYS>;

// Constant taken from https://github.com/rurban/smhasher
constexpr static uint64_t MULTIPLY_CONSTANT = 0x75f17d6b3588f843ull;

uint64_t calculate_hash(uint64_t key, uint64_t required_bits) {
  return (key * MULTIPLY_CONSTANT) >> (64 - required_bits);
}

template <typename HashFn>
void BM_hashing(benchmark::State& state) {
  HashFn hash_fn{};
  uint64_t required_bits = state.range(0);

  // Seed rng for same benchmark runs.
  std::mt19937_64 rng{345873456};

  AlignedArray keys_to_hash;
  AlignedArray correct_hash_values;
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    keys_to_hash[i] = rng();
    correct_hash_values[i] = calculate_hash(keys_to_hash[i], required_bits);
  }

  AlignedArray hashes{};

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
  void operator()(const AlignedArray& keys_to_hash, uint64_t required_bits, AlignedArray* __restrict result) {
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
struct x86_128_hash {
  void operator()(const AlignedArray& keys_to_hash, uint64_t required_bits, AlignedArray* __restrict result) {
    // TODO
    (void)keys_to_hash;
    (void)required_bits;
    (void)result;
  }
};
BENCHMARK(BM_hashing<x86_128_hash>)->BM_ARGS;

#if defined(AVX512_AVAILABLE)
struct x86_512_hash {
  void operator()(const AlignedArray& keys_to_hash, uint64_t required_bits, AlignedArray* __restrict result) {
    // TODO
  }
};
BENCHMARK(BM_hashing<x86_512_hash>)->BM_ARGS;
#endif

#endif

struct naive_scalar_hash {
  void operator()(const AlignedArray& keys_to_hash, uint64_t required_bits, AlignedArray* __restrict result) {
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      (*result)[i] = calculate_hash(keys_to_hash[i], required_bits);
    }
  }
};

struct autovec_scalar_hash {
  void operator()(const AlignedArray& keys_to_hash, uint64_t required_bits, AlignedArray* __restrict result) {
    // TODO(Richard): Actually, the naive code is already perfectly vectorizable. Check out generated assembly.
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      (*result)[i] = keys_to_hash[i] * MULTIPLY_CONSTANT;
    }

    uint64_t shift = 64 - required_bits;
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      (*result)[i] = (*result)[i] >> shift;
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

  void operator()(const AlignedArray& keys_to_hash, uint64_t required_bits, AlignedArray* __restrict result) {
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
