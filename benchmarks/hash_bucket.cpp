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
  // TODO: The real hashmap has to handle potentially deleted entries (tombstones), bucket overflow
  // TODO: For F14, all "valid" fingerprints have the MSB set to distinguish from "no value stored" (I think?)
  // TODO: F14 has 14 elements after 2 bytes of metadata. Is that our guideline?
  std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<> index_distribution(0, NUM_ENTRIES - 1);
  std::ranges::generate(bucket.fingerprints, [&]() { return rng(); });
  std::ranges::generate(bucket.entries, [&]() { return Entry{rng(), rng()}; });

  std::array<size_t, NUM_ENTRIES> lookup_indices;
  std::ranges::generate(lookup_indices, [&]() {
    if (rng() % 2 == 0) {
      return index_distribution(rng);  // With ~50% probability, we look up a random existing element.
    } else {
      return -1;  // otherwise, we attempt to lookup a non-existing key
    }
  });

  std::array<uint8_t, NUM_ENTRIES> lookup_fps;
  std::ranges::transform(lookup_indices, lookup_fps.begin(), [&](size_t index) {
    if (index == -1ull) {
      return static_cast<uint8_t>(rng() | 128);  // can collide, this is realistic
    } else {
      return bucket.fingerprints[index];
    }
  });

  std::array<uint64_t, NUM_ENTRIES> lookup_keys{};
  std::ranges::transform(lookup_indices, lookup_fps.begin(), [&](size_t index) {
    if (index == -1ull) {
      return rng();  // In theory, this can also collide. We don't want that to happen. In practice, it doesn't happen.
    } else {
      return bucket.entries[index].key;
    }
  });

  // "leak" pointers so that successive DoNotOptimize calls invalidate it.
  benchmark::DoNotOptimize(lookup_keys.data());
  benchmark::DoNotOptimize(lookup_fps.data());

  for (auto _ : state) {
    for (size_t i = 0; i < NUM_ENTRIES; ++i) {
      const uint64_t value = find_fn(bucket, lookup_keys[i], lookup_fps[i]);
      assert((lookup_indices[i] == -1ull && value == NO_MATCH) ||
             (lookup_indices[i] != -1ull && value == bucket.entries[lookup_indices[i]].value));

      benchmark::DoNotOptimize(value);
    }
  }
}

// Offset is required to process both 64bit halves of the 128 bit fingerprint comparison result
inline uint64_t key_matches_from_fingerprint_matches_byte(HashBucket& bucket, uint64_t key,
                                                          uint64_t fingerprint_matches, size_t entry_offset) {
  // TODO: Call once, pass in 16B (as uint128_t?). We want this to be inlined anyway, could also be a reference.
  while (fingerprint_matches != 0) {
    uint32_t trailing_zeros = std::countr_zero(fingerprint_matches);
    // 1B = 8bit per truthness-value
    uint16_t match_pos = entry_offset + (trailing_zeros / 8);

    // We expect fingerprint collisions to be unlikely
    if (bucket.entries[match_pos].key == key) [[likely]] {
      return bucket.entries[match_pos].value;
    }

    // Clear all bits in the 0b11111111-byte
    fingerprint_matches &= ~(255ul << trailing_zeros);
  }
  return NO_MATCH;
}

#if defined(__aarch64__)
struct neon_find_bytes {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    uint8x16_t fp_vector = vld1q_u8(bucket.fingerprints.data());

    // Broadcast the fingerprint to compare against.
    uint8x16_t lookup_fp = vmovq_n_u8(fingerprint);
    // The fingerprints-array only has 15 elements, so we use 0 as a never-matching lookup value for the last element
    // TODO: It is not guaranteed that the value stored in the padding is not actually 0. So, this could very well
    // give us a match. We would then access keys out-of-bounds.
    // Generally, we're deep inside UB land here.
    // Autovectorization will likely not do this.
    // TODO: Guarantee that fingerprint can never be 0 is somewhat unrealistic?
    lookup_fp[15] = 0;

    uint8x16_t fingerprint_matches = vceqq_u8(fp_vector, lookup_fp);

    uint64_t low_fp_matches = reinterpret_cast<uint64_t*>(matching_fingerprints)[0];
    uint64_t found_value = key_matches_from_fingerprint_matches_byte(bucket, key, low_matches, 0);
    if (found_value != NO_MATCH) {
      return found_value;
    }

    uint64_t high_fp_matches = reinterpret_cast<uint64_t*>(&matching_fingerprints)[1];
    return key_matches_from_fingerprint_matches_byte(bucket, key, high_matches, 8);
  }
};
BENCHMARK(BM_hash_bucket_get<neon_find_bytes>)->BM_ARGS;

#elif defined(__x86_64__)
struct x86_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    // TODO
    uint8_t* fingerprints = bucket.fingerprints.data();
    (void)fingerprints;
    (void)key;
    (void)fingerprint;
    return 13;
  }
};

BENCHMARK(BM_hash_bucket_get<x86_find>)->BM_ARGS;
#endif

struct naive_scalar_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    // Somewhat of a baseline (not really, since entries are stored as key-value pairs. I'd guess for this to be
    // auto-vectorizable, it would have to be struct-of-array layout (all keys, then all values).
    for (size_t i = 0; i < NUM_ENTRIES; ++i) {
      if (bucket.fingerprints[i] == fingerprint && bucket.entries[i].key == key) {
        return bucket.entries[i].value;
      }
    }
    return NO_MATCH;
  }
};

struct naive_scalar_key_only_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t /*fingerprint*/) {
    for (size_t i = 0; i < NUM_ENTRIES; ++i) {
      if (bucket.entries[i].key == key) {
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
  using vec8x16 = GccVec<uint8_t, 16>::T;
  using vec64x2 = GccVec<uint64_t, 16>::T;

  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    vec8x16 fp_vector = *reinterpret_cast<vec8x16*>(bucket.fingerprints.data());

    vec8x16 lookup_fp = broadcast<vec8x16>(fingerprint);
    vec8x16 matching_fingerprints = fp_vector == lookup_fp;

    uint64_t low_matches = reinterpret_cast<uint64_t*>(&matching_fingerprints)[0];
    uint64_t low_match = key_matches_from_fingerprint_matches_byte(bucket, key, low_matches, 0);
    if (low_match != NO_MATCH) {
      return low_match;
    }

    uint64_t high_matches = reinterpret_cast<uint64_t*>(&matching_fingerprints)[1];
    return key_matches_from_fingerprint_matches_byte(bucket, key, high_matches, 8);
  }
};

BENCHMARK(BM_hash_bucket_get<naive_scalar_key_only_find>)->BM_ARGS;
BENCHMARK(BM_hash_bucket_get<naive_scalar_find>)->BM_ARGS;
BENCHMARK(BM_hash_bucket_get<autovec_scalar_find>)->BM_ARGS;
BENCHMARK(BM_hash_bucket_get<vector_find>)->BM_ARGS;

BENCHMARK_MAIN();
