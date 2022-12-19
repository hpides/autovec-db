#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"

#define BM_ARGS UseRealTime()->Repetitions(5)->Arg(27);

static constexpr size_t COMPRESS_BITS = 9;

namespace {

std::vector<uint64_t> compress_input(const std::vector<uint32_t>& input) {
  constexpr uint64_t U64_BITS = 64;
  constexpr uint64_t MASK = (1 << COMPRESS_BITS) - 1;

  size_t needed_bits = COMPRESS_BITS * input.size();
  size_t array_size = std::ceil(static_cast<double>(needed_bits) / U64_BITS);
  std::vector<uint64_t> compressed_data(array_size);
  uint64_t* buffer = compressed_data.data();

  uint64_t bits_left = U64_BITS;
  size_t idx_ = 0;

  for (uint32_t i : input) {
    uint64_t val = i & MASK;
    buffer[idx_] |= val << (U64_BITS - bits_left);

    if (bits_left < COMPRESS_BITS) {
      buffer[++idx_] |= val >> bits_left;
      bits_left += U64_BITS;
    }
    bits_left -= COMPRESS_BITS;
    if (bits_left == 0) {
      bits_left = U64_BITS;
      idx_++;
    }
  }

  return compressed_data;
}

}  // namespace

static constexpr uint64_t NUM_KEYS = 10'000'000;
using Column = AlignedArray<uint32_t, NUM_KEYS, 512>;

template <typename HashFn>
void BM_scanning(benchmark::State& state) {
  HashFn hash_fn{};
  uint64_t shift = state.range(0);

  // Seed rng for same benchmark runs.
  std::mt19937_64 rng{82323457236434673ul};

  std::unique_ptr<Column> keys_to_scan;
  for (size_t i = 0; i < NUM_KEYS; ++i) {
    keys_to_scan.data[i] = rng();
    check_keys.data[i] = calculate_scan(keys_to_scan.data[i], shift);
  }

  for (auto _ : state) {
    hashes = hash_fn(keys_to_scan, shift);
    benchmark::DoNotOptimize(hashes);
  }
}

#if defined(__aarch64__)
#include <arm_neon.h>

/** Doing this in NEON is not very useful, as we can neither do a vector multiply nor a variable right shift. */
struct neon_scan {
  AlignedArray operator()(const AlignedArray& keys_to_scan, uint64_t shift) {
  }
};

#elif defined(__x86_64__)
struct x86_128_scan {
  AlignedArray operator()(const AlignedArray& keys_to_scan, uint64_t shift) {
    // TODO
    return AlignedArray{};
  }
};

struct x86_512_scan {
  AlignedArray operator()(const AlignedArray& keys_to_scan, uint64_t shift) {
    // TODO
    return AlignedArray{};
  }
};
#endif

struct naive_scalar_scan {
  void operator()(const AlignedArray& keys_to_scan, uint64_t shift) {
  }
};

struct autovec_scalar_scan {
  AlignedArray operator()(const AlignedArray& keys_to_scan, uint64_t shift) {
  }
};



#if defined(__aarch64__)
BENCHMARK(BM_scanning<neon_scan>)->BM_ARGS;
#endif

#if defined(__x86_64__)
BENCHMARK(BM_scaning<x86_128_scan>)->BM_ARGS;
BENCHMARK(BM_scaning<x86_512_scan>)->BM_ARGS;
#endif

BENCHMARK(BM_scanning<naive_scalar_scan>)->BM_ARGS;
BENCHMARK(BM_scanning<autovec_scalar_scan>)->BM_ARGS;

BENCHMARK_MAIN();
