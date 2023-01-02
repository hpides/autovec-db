#include <array>
#include <bit>
#include <cstdint>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"

static constexpr uint64_t NUM_ENTRIES = 15;
static constexpr uint64_t NO_MATCH = std::numeric_limits<uint64_t>::max();

struct Entry {
  uint64_t key;
  uint64_t value;
};

struct HashBucket {
  alignas(16) std::array<uint8_t, NUM_ENTRIES> fingerprints;
  std::array<Entry, NUM_ENTRIES> entries;
};

static_assert(sizeof(HashBucket) == 256, "Hash Bucket should be 256 Byte for this benchmark");

#define BM_ARGS Repetitions(1)

template <typename FindFn>
void BM_hash_bucket_get(benchmark::State& state) {
  FindFn find_fn{};
  HashBucket bucket{};
  bucket.fingerprints = {3, 8, 12, 10, 11, 6, 15, 1, 5, 14, 13, 2, 9, 4, 7};
  bucket.entries = {Entry{30, 30},   Entry{80, 80},   Entry{120, 120}, Entry{100, 100}, Entry{110, 110},
                    Entry{60, 60},   Entry{150, 150}, Entry{10, 10},   Entry{50, 50},   Entry{140, 140},
                    Entry{130, 130}, Entry{20, 20},   Entry{90, 90},   Entry{40, 40},   Entry{70, 70}};

  // Seed rng for same benchmark runs.
  std::mt19937 rng{345873456};

  std::array<uint8_t, NUM_ENTRIES> lookup_fps = bucket.fingerprints;
  std::shuffle(lookup_fps.begin(), lookup_fps.end(), rng);

  std::array<uint64_t, NUM_ENTRIES> lookup_keys{};
  for (size_t i = 0; i < NUM_ENTRIES; ++i) {
    lookup_keys[i] = lookup_fps[i] * 10;
  }

  for (auto _ : state) {
    for (size_t i = 0; i < NUM_ENTRIES; ++i) {
      benchmark::DoNotOptimize(&bucket);
      benchmark::DoNotOptimize(lookup_keys.data());
      benchmark::DoNotOptimize(lookup_fps.data());

      uint64_t value = find_fn(bucket, lookup_keys[i], lookup_fps[i]);
      assert(value != NO_MATCH);

      benchmark::DoNotOptimize(value);
    }
  }
}

// We need to pass an offset here, as the higher matches need to check the second half of the entries array.
uint64_t find_key_match(HashBucket& bucket, uint64_t key, uint64_t matches, size_t entry_offset) {
  while (matches != 0) {
    // The comparison returns 00000000 for a mismatch, so we need to divide by 8 to get the actual number of 0's.
    uint32_t trailing_zeros = std::countr_zero(matches);
    uint16_t match_pos = entry_offset + (trailing_zeros / 8);

    // We give this a likely hint, as we expect the number of fingerprint collisions to be low. So on average, we
    // want this to be the happy path and immediately return if possible.
    if (bucket.entries[match_pos].key == key) [[likely]] {
      return bucket.entries[match_pos].value;
    }

    // We want to remove all 1's that we just matched. So we set all 8 bits that we just matched and invert the
    // number for a clean 11111111...00000000...11111111 mask.
    matches &= ~(255ul << trailing_zeros);
  }
  return NO_MATCH;
}

#if defined(__aarch64__)
#include <arm_neon.h>

struct neon_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    uint8_t* fingerprints = bucket.fingerprints.data();

    // Load the fingerprints into a SIMD register.
    uint8x16_t fp_vector = vld1q_u8(fingerprints);

    // Broadcast the fingerprint to compare against into a SIMD register. We only use 15 values, so the last one is 0.
    vec8x16_t lookup_fp = vmovq_n_u8(fingerprint);
    lookup_fp[15] = 0;

    // Compare fingerprints.
    auto matching_fingerprints = reinterpret_cast<__uint128_t>(vceqq_u8(fp_vector, lookup_fp));

    // We could do this with a single movemask on x86, but ARM NEON does not support this. So we split our range into
    // two values that we check after each other. The extraction here is a no-op, as the __uint128_t result is stored in
    // two 64 bit registers anyway. This is only a logical conversion.
    uint64_t low_matches = *reinterpret_cast<uint64_t*>(&matching_fingerprints);
    uint64_t high_matches = *(reinterpret_cast<uint64_t*>(&matching_fingerprints) + 1);

    uint64_t low_match = find_key_match(bucket, key, low_matches, 0);
    if (low_match != NO_MATCH) {
      return low_match;
    }

    return find_key_match(bucket, key, high_matches, 8);
  }
};

BENCHMARK(BM_hash_bucket_get<neon_find>)->BM_ARGS;

#elif defined(__x86_64__)
struct x86_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    // TODO
    uint8_t* fingerprints = bucket.fingerprints.data();
    (void)fingerprints;
    (void)key;
    (void)fingerprint;
    return 0;
  }
};

BENCHMARK(BM_hash_bucket_get<x86_find>)->BM_ARGS;
#endif

struct naive_scalar_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    for (size_t i = 0; i < NUM_ENTRIES; ++i) {
      if (bucket.fingerprints[i] == fingerprint && bucket.entries[i].key == key) {
        return bucket.entries[i].value;
      }
    }
    return NO_MATCH;
  }
};

struct autovec_scalar_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    // TODO: This is not the solution we want yet, it's just a dummy WIP state.
    alignas(16) std::array<bool, 16> matches{false};
    for (size_t i = 0; i < NUM_ENTRIES; ++i) {
      matches[i] = bucket.fingerprints[i] == fingerprint;
    }

    for (size_t i = 0; i < NUM_ENTRIES; ++i) {
      if (matches[i] && bucket.entries[i].key == key) {
        return bucket.entries[i].value;
      }
    }
    return NO_MATCH;
  }
};

struct vector_find {
  using vec8x16 = uint8_t __attribute__((vector_size(16)));
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    uint8_t* fingerprints = bucket.fingerprints.data();

    // Load the fingerprints into a SIMD register.
    vec8x16 fp_vector = *reinterpret_cast<vec8x16*>(fingerprints);

    // Broadcast the fingerprint to compare against into a SIMD register. We only use 15 values, so the last one is 0.
    vec8x16 lookup_fp = broadcast<vec8x16>(fingerprint);
    lookup_fp[15] = 0;

    // Compare fingerprints.
    auto matching_fingerprints = reinterpret_cast<__uint128_t>(fp_vector == lookup_fp);

    uint64_t low_matches = static_cast<uint64_t>(matching_fingerprints);
    uint64_t high_matches = static_cast<uint64_t>(matching_fingerprints >> 64);

    uint64_t low_match = find_key_match(bucket, key, low_matches, 0);
    if (low_match != NO_MATCH) {
      return low_match;
    }

    return find_key_match(bucket, key, high_matches, 8);
  }
};

BENCHMARK(BM_hash_bucket_get<naive_scalar_find>)->BM_ARGS;
BENCHMARK(BM_hash_bucket_get<autovec_scalar_find>)->BM_ARGS;
BENCHMARK(BM_hash_bucket_get<vector_find>)->BM_ARGS;

BENCHMARK_MAIN();
