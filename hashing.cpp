#include <array>
#include <cstdint>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"

#define BM_ARGS UseRealTime()->Repetitions(10);

static constexpr uint64_t NUM_KEYS = 128;

struct alignas(512) AlignedArray {
  std::array<uint64_t, NUM_KEYS> data{};
};

// Constant taken from https://github.com/rurban/smhasher
constexpr static uint64_t MULTIPLY_CONSTANT = 0x75f17d6b3588f843ull;

/** We assume a basic multiply-shift hash, based on the constants used in SMHasher and 22 to 27 Bits needed in the hash
 * table. We add this range to avoid the compiler optimizing the subtraction here. We perform a multiplication
 * followed by a right-shift by (64 - shift). */
uint64_t calculate_hash(uint64_t key, uint64_t shift) {
  return (key * MULTIPLY_CONSTANT) >> (64 - shift);
}

template <typename HashFn>
void BM_hashing(benchmark::State& state) {
  HashFn hash_fn{};

  // Seed rng for same benchmark runs.
  std::mt19937_64 rng{345873456};

  AlignedArray keys_to_hash{};
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    keys_to_hash.data[i] = rng();
  }

  for (auto _ : state) {
    AlignedArray hashes = hash_fn(keys_to_hash, 27);
    benchmark::DoNotOptimize(hashes);
  }
}

#if defined(__aarch64__)
#include <arm_neon.h>

struct neon_hash {
  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    // TODO
    return AlignedArray{};
  }
};

BENCHMARK(BM_hashing<neon_hash>)->BM_ARGS;

#elif defined(__x86_64__)
struct x86_128_hash {
  KeyArray operator()(const KeyArray& keys_to_hash, uint64_t shift) {
    // TODO
    return AlignedArray{};
  }
};

struct x86_512_hash {
  KeyArray operator()(const KeyArray& keys_to_hash, uint64_t shift) {
    // TODO
    return AlignedArray{};
  }
};

BENCHMARK(BM_hashing<x86_128_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<x86_512_hash>)->BM_ARGS;
#endif

struct naive_scalar_hash {
  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    AlignedArray hashes{};
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      hashes.data[i] = calculate_hash(keys_to_hash.data[i], shift);
    }
    return hashes;
  }
};

struct autovec_scalar_hash {
  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    // TODO: Check this.
    AlignedArray multiplied_values{};
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      multiplied_values.data[i] = keys_to_hash.data[i] * MULTIPLY_CONSTANT;
    }

    AlignedArray shifted_values{};
    uint64_t actual_shift = 64 - shift;
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      shifted_values.data[i] = multiplied_values.data[i] >> actual_shift;
    }

    return shifted_values;
  }
};

struct vector_128_hash {
  using vec64x2 = uint64_t __attribute__((vector_size(16)));
  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    // TODO
    return AlignedArray{};
  }
};

struct vector_512_hash {
  using vec64x8 = uint64_t __attribute__((vector_size(64)));
  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    // TODO
    return AlignedArray{};
  }
};

BENCHMARK(BM_hashing<naive_scalar_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<autovec_scalar_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_128_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_512_hash>)->BM_ARGS;

BENCHMARK_MAIN();
