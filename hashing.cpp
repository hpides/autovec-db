#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"

#define BM_ARGS UseRealTime()->Repetitions(5)->Arg(27);

// TODO(lawben): Check which number here makes sense. We need: #keys / (#vector-lanes / 8B key) registers.
//                 --> 128 keys = 16 zmm | 32 ymm | 64 xmm registers
//                 -->  64 keys =  8 zmm | 16 ymm | 32 xmm registers
//               This will impact if the compiler can unroll the loop or not.
static constexpr uint64_t NUM_KEYS = 64;

struct alignas(512) AlignedArray {
  // We want to use an empty custom constructor here to avoid zeroing the array when creating an AlignedArray.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init,modernize-use-equals-default)
  AlignedArray(){};

  std::array<uint64_t, NUM_KEYS> data;
};

// Constant taken from https://github.com/rurban/smhasher
// constexpr static uint64_t MULTIPLY_CONSTANT = 0x75f17d6b3588f843ull;
constexpr static uint64_t MULTIPLY_CONSTANT = 2;

/** We assume a basic multiply-shift hash, based on the constants used in SMHasher and 27 Bits needed in the hash table.
 * We add this range to avoid the compiler optimizing the subtraction here. We perform a multiplication followed by a
 * right-shift by (64 - shift). */
uint64_t calculate_hash(uint64_t key, uint64_t shift) { return (key * MULTIPLY_CONSTANT) >> (64 - shift); }

template <typename HashFn>
void BM_hashing(benchmark::State& state) {
  HashFn hash_fn{};
  uint64_t shift = state.range(0);

  // Seed rng for same benchmark runs.
  std::mt19937_64 rng{345873456};

  AlignedArray keys_to_hash;
  AlignedArray check_keys;
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    keys_to_hash.data[i] = rng();
    check_keys.data[i] = calculate_hash(keys_to_hash.data[i], shift);
  }

  AlignedArray hashes{};
  // Fill with 0s here as we will most likely reuse the same memory after each run, and we use uninitialized array to
  // avoid zeroing explicitly in each iteration.
  std::fill(hashes.data.begin(), hashes.data.end(), 0);

  // Do one sanity check that we get the correct results.
  hashes = hash_fn(keys_to_hash, shift);
  if (hashes.data != check_keys.data) {
    throw std::runtime_error{"Bad hash calculation"};
  }

  for (auto _ : state) {
    hashes = hash_fn(keys_to_hash, shift);
    benchmark::DoNotOptimize(hashes);
  }
}

#if defined(__aarch64__)
#include <arm_neon.h>

/** Doing this in NEON is not very useful, as we can neither do a vector multiply nor a variable right shift. */
struct neon_hash {
  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    struct alignas(alignof(AlignedArray)) VecArray {
      std::array<uint64x2_t, NUM_KEYS / 2> data;
    };

    AlignedArray actual_hashes;
    const auto& vec_keys = reinterpret_cast<const VecArray&>(keys_to_hash);
    auto& hashes = reinterpret_cast<VecArray&>(actual_hashes);

    // We need this "hack" here, as NEON requires the right shift value to be a compile-time constant. So we shift left
    // with a negative value.
    int64_t actual_shift = -(64 - static_cast<int64_t>(shift));
    uint64x2_t shift_value = vmovq_n_u64(actual_shift);

    for (size_t i = 0; i < NUM_KEYS / 2; ++i) {
      const size_t offset = i * 2;
      alignas(16) std::array<uint64_t, 2> keys{keys_to_hash.data[offset] * MULTIPLY_CONSTANT,
                                               keys_to_hash.data[offset + 1] * MULTIPLY_CONSTANT};
      uint64x2_t multiplied_keys = vld1q_u64(keys.data());
      hashes.data[i] = vshlq_u64(multiplied_keys, shift_value);
    }

    return actual_hashes;
  }
};

#elif defined(__x86_64__)
struct x86_128_hash {
  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    // TODO
    return AlignedArray{};
  }
};

struct x86_512_hash {
  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    // TODO
    return AlignedArray{};
  }
};
#endif

struct naive_scalar_hash {
  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    AlignedArray hashes;
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      hashes.data[i] = calculate_hash(keys_to_hash.data[i], shift);
    }
    return hashes;
  }
};

struct autovec_scalar_hash {
  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    AlignedArray hashes;
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      hashes.data[i] = keys_to_hash.data[i] * MULTIPLY_CONSTANT;
    }

    uint64_t actual_shift = 64 - shift;
    for (size_t i = 0; i < NUM_KEYS; ++i) {
      hashes.data[i] = hashes.data[i] >> actual_shift;
    }

    return hashes;
  }
};

template <size_t VECTOR_BITS>
struct vector_hash {
  static constexpr size_t VECTOR_BYTES = VECTOR_BITS / 8;
  static constexpr size_t NUM_VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(uint64_t);

  using VecT __attribute__((vector_size(VECTOR_BYTES))) = uint64_t;
  static_assert(sizeof(VecT) == VECTOR_BYTES);

  struct alignas(alignof(AlignedArray)) VecArray {
    std::array<VecT, NUM_KEYS / NUM_VECTOR_ELEMENTS> data;
  };

  AlignedArray operator()(const AlignedArray& keys_to_hash, uint64_t shift) {
    const auto& vec_keys = reinterpret_cast<const VecArray&>(keys_to_hash);

    AlignedArray actual_hashes;
    auto& hashes = reinterpret_cast<VecArray&>(actual_hashes);

    uint64_t actual_shift = 64 - shift;
    for (size_t i = 0; i < NUM_KEYS / NUM_VECTOR_ELEMENTS; ++i) {
      hashes.data[i] = (vec_keys.data[i] * MULTIPLY_CONSTANT) >> actual_shift;
    }

    return actual_hashes;
  }
};

#if defined(__aarch64__)
BENCHMARK(BM_hashing<neon_hash>)->BM_ARGS;
#endif

#if defined(__x86_64__)
BENCHMARK(BM_hashing<x86_128_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<x86_512_hash>)->BM_ARGS;
#endif

BENCHMARK(BM_hashing<naive_scalar_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<autovec_scalar_hash>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_hash<64>>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_hash<128>>)->BM_ARGS;

// TODO: figure out why these or why they are so fast!
BENCHMARK(BM_hashing<vector_hash<256>>)->BM_ARGS;
BENCHMARK(BM_hashing<vector_hash<512>>)->BM_ARGS;

BENCHMARK_MAIN();
