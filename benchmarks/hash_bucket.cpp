#include <stddef.h>

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
  // Valid fingerprints have their most significant bit set. MSB=0 indicates that there is no entry stored at this slot.
  alignas(16) std::array<uint8_t, NUM_ENTRIES + 1> fingerprints;
  std::array<Entry, NUM_ENTRIES> entries;
};

static_assert(sizeof(HashBucket) == 256, "Hash Bucket should be 256 Byte for this benchmark");

#define BM_ARGS Repetitions(10)->MinTime(0.1)->ReportAggregatesOnly()

template <typename FindFn>
void BM_hash_bucket_get(benchmark::State& state) {
  std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<> index_distribution(0, NUM_ENTRIES - 1);
  std::uniform_int_distribution<uint64_t> existing_key_distribution(0, 1e19);
  std::uniform_int_distribution<uint64_t> non_existing_key_distribution(1e19 + 1);

  HashBucket bucket{};
  std::ranges::generate(bucket.fingerprints, [&]() { return rng() | 128; });
  bucket.fingerprints[NUM_ENTRIES] = 0;
  std::ranges::generate(bucket.entries, [&]() { return Entry{existing_key_distribution(rng), rng()}; });

  std::array<size_t, NUM_ENTRIES> lookup_indices;
  std::ranges::generate(lookup_indices, [&]() {
    if (rng() % 2 == 0) {
      // TODO: If this gives us too much deviation on the benchmarking results, we might want to have less random
      // indices here. Currently, I get a stddev of 5% to 15% on vector_find
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
  std::ranges::transform(lookup_indices, lookup_keys.begin(), [&](size_t index) {
    if (index == -1ull) {
      return non_existing_key_distribution(rng);
    } else {
      return bucket.entries[index].key;
    }
  });

  // "leak" pointers so that successive DoNotOptimize calls invalidate it.
  benchmark::DoNotOptimize(lookup_keys.data());
  benchmark::DoNotOptimize(lookup_fps.data());

  FindFn find_fn{};
  for (auto _ : state) {
    for (size_t i = 0; i < NUM_ENTRIES; ++i) {
      const uint64_t value = find_fn(bucket, lookup_keys[i], lookup_fps[i]);
      assert((lookup_indices[i] == -1ull && value == NO_MATCH) ||
             (lookup_indices[i] != -1ull && value == bucket.entries[lookup_indices[i]].value));

      benchmark::DoNotOptimize(value);
    }
  }
}

inline uint64_t key_matches_from_fingerprint_matches_byte(HashBucket& bucket, uint64_t key,
                                                          __uint128_t fingerprint_matches) {
  while (fingerprint_matches != 0) {
    int trailing_zeros = std::countr_zero(fingerprint_matches);
    // 1B = 8bit per truthness-value
    int match_pos = (trailing_zeros / 8);

    // We expect fingerprint collisions to be unlikely
    if (bucket.entries[match_pos].key == key) [[likely]] {
      return bucket.entries[match_pos].value;
    }

    // Clear all bits in the 0b11111111-byte
    fingerprint_matches &= ~(static_cast<__uint128_t>(255) << trailing_zeros);
  }
  return NO_MATCH;
}

inline uint64_t key_matches_from_fingerprint_matches_bit(HashBucket& bucket, uint64_t key,
                                                         uint16_t fingerprint_matches) {
  while (fingerprint_matches != 0) {
    int trailing_zeros = std::countr_zero(fingerprint_matches);
    int match_pos = trailing_zeros;

    // We expect fingerprint collisions to be unlikely
    if (bucket.entries[match_pos].key == key) [[likely]] {
      return bucket.entries[match_pos].value;
    }

    // Clear the fingerprint match bit
    fingerprint_matches &= ~(1 << trailing_zeros);
  }
  return NO_MATCH;
}

#if defined(__aarch64__)
struct neon_bytemask_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    uint8x16_t fp_vector = vld1q_u8(bucket.fingerprints.data());

    // Broadcast the fingerprint to compare against.
    uint8x16_t lookup_fp = vmovq_n_u8(fingerprint);
    uint8x16_t fingerprint_matches = vceqq_u8(fp_vector, lookup_fp);

    return key_matches_from_fingerprint_matches_byte(bucket, key, reinterpret_cast<__uint128_t>(fingerprint_matches));
  }
};
BENCHMARK(BM_hash_bucket_get<neon_bytemask_find>)->BM_ARGS;
#endif

#if defined(__x86_64__)
struct x86_bytemask_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    __m128i* fp_vector = reinterpret_cast<__m128i*>(bucket.fingerprints.data());
    __m128i lookup_fp = _mm_set1_epi8(fingerprint);
    __m128i fingerprint_matches = _mm_cmpeq_epi8(*fp_vector, lookup_fp);

    return key_matches_from_fingerprint_matches_byte(bucket, key, reinterpret_cast<__uint128_t>(fingerprint_matches));
  }
};
BENCHMARK(BM_hash_bucket_get<x86_bytemask_find>)->BM_ARGS;

struct x86_bitmask_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    __m128i* fp_vector = reinterpret_cast<__m128i*>(bucket.fingerprints.data());
    __m128i lookup_fp = _mm_set1_epi8(fingerprint);
    __m128i fingerprint_matches = _mm_cmpeq_epi8(*fp_vector, lookup_fp);
    uint16_t fingerprint_matches_bits = _mm_movemask_epi8(fingerprint_matches);

    return key_matches_from_fingerprint_matches_bit(bucket, key, fingerprint_matches_bits);
  }
};
BENCHMARK(BM_hash_bucket_get<x86_bitmask_find>)->BM_ARGS;
#endif

#if defined(AVX512_AVAILABLE)
struct x86_avx512_bitmask_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    __m128i* fp_vector = reinterpret_cast<__m128i*>(bucket.fingerprints.data());
    __m128i lookup_fp = _mm_set1_epi8(fingerprint);
    uint16_t fingerprint_matches = _mm_cmpeq_epi8_mask(*fp_vector, lookup_fp);

    return key_matches_from_fingerprint_matches_bit(bucket, key, fingerprint_matches);
  }
};
BENCHMARK(BM_hash_bucket_get<x86_avx512_bitmask_find>)->BM_ARGS;
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
BENCHMARK(BM_hash_bucket_get<naive_scalar_find>)->BM_ARGS;

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
BENCHMARK(BM_hash_bucket_get<naive_scalar_key_only_find>)->BM_ARGS;

struct autovec_scalar_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    // TODO: This code is okay-ish C++ for autovectorization.
    // However, clang is currently broken when extracting values from the 16-byte char array as a bigger int
    // (https://github.com/llvm/llvm-project/issues/59937)
    // and GCC doesn't autovectorize the match-from-fingerprints logic well
    // playground: https://godbolt.org/z/Eoezxh9E3
    uint8_t* __restrict fingerprint_data = std::assume_aligned<16>(bucket.fingerprints.data());

    alignas(16) std::array<uint8_t, 16> matches;
    for (size_t i = 0; i < 16; ++i) {
      matches[i] = fingerprint_data[i] == fingerprint ? 0xff : 0;
    }

    return key_matches_from_fingerprint_matches_byte(bucket, key, reinterpret_cast<__uint128_t&>(matches[0]));
  }
};
BENCHMARK(BM_hash_bucket_get<autovec_scalar_find>)->BM_ARGS;

struct vector_bytemask_find {
  using vec8x16 = GccVec<uint8_t, 16>::T;
  using vec64x2 = GccVec<uint64_t, 16>::T;

  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    vec8x16 fp_vector = *reinterpret_cast<vec8x16*>(bucket.fingerprints.data());

    vec8x16 lookup_fp = broadcast<vec8x16>(fingerprint);
    vec8x16 matching_fingerprints = fp_vector == lookup_fp;

    return key_matches_from_fingerprint_matches_byte(bucket, key, reinterpret_cast<__uint128_t>(matching_fingerprints));
  }
};
BENCHMARK(BM_hash_bucket_get<vector_bytemask_find>)->BM_ARGS;

#if CLANG_COMPILER
struct vector_bitmask_find {
  using vec8x16 = GccVec<uint8_t, 16>::T;
  using vec64x2 = GccVec<uint64_t, 16>::T;
  using vec1x16 = ClangBitmask<16>::T;

  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    vec8x16 fp_vector = *reinterpret_cast<vec8x16*>(bucket.fingerprints.data());
    vec8x16 lookup_fp = broadcast<vec8x16>(fingerprint);

    vec1x16 matching_fingerprints_bits_vec = __builtin_convertvector(fp_vector == lookup_fp, vec1x16);
    uint16_t matching_fingerprints_bits = reinterpret_cast<uint16_t&>(matching_fingerprints_bits_vec);

    return key_matches_from_fingerprint_matches_bit(bucket, key, matching_fingerprints_bits);
  }
};
BENCHMARK(BM_hash_bucket_get<vector_bitmask_find>)->BM_ARGS;
#endif

BENCHMARK_MAIN();
