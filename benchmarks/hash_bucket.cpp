#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"
#include "simd.hpp"

static constexpr uint64_t NUM_ENTRIES = 15;
static constexpr uint64_t NUM_LOOKUPS_PER_ITERATION = 1024;
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

#define BM_ARGS Unit(benchmark::kNanosecond)

template <typename FindFn>
void BM_hash_bucket_get(benchmark::State& state) {
  std::mt19937 rng{std::random_device{}()};
  std::uniform_int_distribution<> index_distribution(0, NUM_ENTRIES - 1);
  // 2^64 is ~1.84e19, so we split the key range into existing and non-existing keys at 1e19
  std::uniform_int_distribution<uint64_t> existing_key_distribution(0, 1e19);
  std::uniform_int_distribution<uint64_t> non_existing_key_distribution(1e19 + 1);

  HashBucket bucket{};
  std::generate(bucket.fingerprints.begin(), bucket.fingerprints.end(), [&]() { return rng() | 128u; });
  bucket.fingerprints[NUM_ENTRIES] = 0;
  std::generate(bucket.entries.begin(), bucket.entries.end(), [&]() {
    return Entry{existing_key_distribution(rng), rng()};
  });

  std::array<size_t, NUM_LOOKUPS_PER_ITERATION> lookup_indices{};
  std::generate(lookup_indices.begin(), lookup_indices.end(), [&]() {
    if (rng() % 2 == 0) {
      return index_distribution(rng);  // With ~50% probability, we look up a random existing element.
    }
    return -1;  // otherwise, we attempt to lookup a non-existing key
  });

  std::array<uint8_t, NUM_LOOKUPS_PER_ITERATION> lookup_fps{};
  std::transform(lookup_indices.begin(), lookup_indices.end(), lookup_fps.begin(), [&](size_t index) {
    if (index == -1ull) {
      return static_cast<uint8_t>(rng() | 128u);  // can collide, this is realistic
    }
    return bucket.fingerprints[index];
  });

  std::array<uint64_t, NUM_LOOKUPS_PER_ITERATION> lookup_keys{};
  std::transform(lookup_indices.begin(), lookup_indices.end(), lookup_keys.begin(), [&](size_t index) {
    if (index == -1ull) {
      return non_existing_key_distribution(rng);
    }
    return bucket.entries[index].key;
  });

  // "leak" pointers so that successive DoNotOptimize calls invalidate it.
  benchmark::DoNotOptimize(lookup_keys.data());
  benchmark::DoNotOptimize(lookup_fps.data());

  FindFn find_fn{};
  for (auto _ : state) {
    for (size_t i = 0; i < NUM_LOOKUPS_PER_ITERATION; ++i) {
      const uint64_t value = find_fn(bucket, lookup_keys[i], lookup_fps[i]);
      assert((lookup_indices[i] == -1ull && value == NO_MATCH) ||
             (lookup_indices[i] != -1ull && value == bucket.entries[lookup_indices[i]].value));

      benchmark::DoNotOptimize(value);
    }
  }

  state.counters["PerLookup"] = benchmark::Counter(static_cast<double>(state.iterations()) * NUM_LOOKUPS_PER_ITERATION,
                                                   benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

inline uint64_t key_matches_from_fingerprint_matches_byte(const HashBucket& bucket, uint64_t key,
                                                          __uint128_t fingerprint_matches) {
  if (fingerprint_matches == 0) {
    return NO_MATCH;
  }

  auto process_half = [&](uint64_t half, size_t bit_offset) {
    while (half != 0) {
      const int trailing_zeros = std::countr_zero(half);

      // We know for a fact that trailing_zeros is a multiple of 8, but clang 15 doesn't properly optimize, and still
      // produces shift-right 3, shift-left 4 to clear the lower 3 bits. Thus, we compute the address manually without
      // the division.
      static_assert(sizeof(bucket.entries[0]) % 8 == 0);
      const auto& bucket_entry =
          *reinterpret_cast<const Entry*>(reinterpret_cast<const std::byte*>(bucket.entries.data()) +
                                          (trailing_zeros + bit_offset) * (sizeof(Entry) / 8));

      if (bucket_entry.key == key) [[likely]] {
        return bucket_entry.value;
      }

      half &= ~(255ul << static_cast<unsigned int>(trailing_zeros));
    }
    return NO_MATCH;
  };

  auto result = process_half(static_cast<uint64_t>(fingerprint_matches), 0);
  if (result != NO_MATCH) {
    return result;
  }
  return process_half(static_cast<uint64_t>(fingerprint_matches >> 64), 64);
}

inline uint64_t key_matches_from_fingerprint_matches_bit(HashBucket& bucket, uint64_t key,
                                                         uint16_t fingerprint_matches) {
  while (fingerprint_matches != 0) {
    const int trailing_zeros = std::countr_zero(fingerprint_matches);
    const int match_pos = trailing_zeros;

    // We expect fingerprint collisions to be unlikely
    if (bucket.entries[match_pos].key == key) [[likely]] {
      return bucket.entries[match_pos].value;
    }

    // Clear the fingerprint match bit (which is now the least significant set bit)
    fingerprint_matches &= fingerprint_matches - 1u;
  }
  return NO_MATCH;
}

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

    alignas(16) std::array<uint8_t, 16> matches{};
    for (size_t i = 0; i < 16; ++i) {
      matches[i] = fingerprint_data[i] == fingerprint ? 0xff : 0;
    }

    return key_matches_from_fingerprint_matches_byte(bucket, key, reinterpret_cast<__uint128_t&>(matches[0]));
  }
};
BENCHMARK(BM_hash_bucket_get<autovec_scalar_find>)->BM_ARGS;

struct vector_bytemask_find {
  using uint8x16 = simd::GccVec<uint8_t, 16>::T;

  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    const auto fp_vector = *reinterpret_cast<uint8x16*>(bucket.fingerprints.data());
    const uint8x16 matching_fingerprints = fp_vector == fingerprint;

    return key_matches_from_fingerprint_matches_byte(bucket, key, reinterpret_cast<__uint128_t>(matching_fingerprints));
  }
};
BENCHMARK(BM_hash_bucket_get<vector_bytemask_find>)->BM_ARGS;

#if CLANG_COMPILER
struct vector_bitmask_find {
  using uint8x16 = simd::GccVec<uint8_t, 16>::T;
  using boolx16 = simd::ClangBitmask<16>::T;

  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    const auto fp_vector = *reinterpret_cast<uint8x16*>(bucket.fingerprints.data());
    const boolx16 matching_fingerprints_bits_vec = __builtin_convertvector(fp_vector == fingerprint, boolx16);
    const uint16_t matching_fingerprints_bits = reinterpret_cast<const uint16_t&>(matching_fingerprints_bits_vec);

    return key_matches_from_fingerprint_matches_bit(bucket, key, matching_fingerprints_bits);
  }
};
BENCHMARK(BM_hash_bucket_get<vector_bitmask_find>)->BM_ARGS;
#endif

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

struct neon_bitmask_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    uint8x16_t fp_vector = vld1q_u8(bucket.fingerprints.data());

    // Broadcast the fingerprint to compare against.
    uint8x16_t lookup_fp = vmovq_n_u8(fingerprint);
    uint8x16_t fingerprint_matches = vceqq_u8(fp_vector, lookup_fp);

    constexpr uint8x16_t bit_mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint16_t matches = 0;
    uint8x16_t masked_matches = vandq_u8(fingerprint_matches, bit_mask);
    matches |= static_cast<uint16_t>(vaddv_u8(vget_low_u8(masked_matches)));
    matches |= static_cast<uint16_t>(vaddv_u8(vget_high_u8(masked_matches))) << 8;

    return key_matches_from_fingerprint_matches_bit(bucket, key, matches);
  }
};
BENCHMARK(BM_hash_bucket_get<neon_bitmask_find>)->BM_ARGS;
#endif

#if defined(__x86_64__)
struct x86_bytemask_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    auto* fp_vector = reinterpret_cast<__m128i*>(bucket.fingerprints.data());
    const __m128i lookup_fp = _mm_set1_epi8(static_cast<int8_t>(fingerprint));
    const __m128i fingerprint_matches = _mm_cmpeq_epi8(*fp_vector, lookup_fp);

    return key_matches_from_fingerprint_matches_byte(bucket, key, reinterpret_cast<__uint128_t>(fingerprint_matches));
  }
};
BENCHMARK(BM_hash_bucket_get<x86_bytemask_find>)->BM_ARGS;

struct x86_bitmask_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    const auto* fp_vector = reinterpret_cast<__m128i*>(bucket.fingerprints.data());
    const __m128i lookup_fp = _mm_set1_epi8(static_cast<int8_t>(fingerprint));
    const __m128i fingerprint_matches = _mm_cmpeq_epi8(*fp_vector, lookup_fp);
    const uint16_t fingerprint_matches_bits = _mm_movemask_epi8(fingerprint_matches);

    return key_matches_from_fingerprint_matches_bit(bucket, key, fingerprint_matches_bits);
  }
};
BENCHMARK(BM_hash_bucket_get<x86_bitmask_find>)->BM_ARGS;
#endif

#if AVX512_AVAILABLE
struct x86_avx512_bitmask_find {
  uint64_t operator()(HashBucket& bucket, uint64_t key, uint8_t fingerprint) {
    const __m128i* fp_vector = reinterpret_cast<__m128i*>(bucket.fingerprints.data());
    const __m128i lookup_fp = _mm_set1_epi8(static_cast<int8_t>(fingerprint));
    const uint16_t fingerprint_matches = _mm_cmpeq_epi8_mask(*fp_vector, lookup_fp);

    return key_matches_from_fingerprint_matches_bit(bucket, key, fingerprint_matches);
  }
};
BENCHMARK(BM_hash_bucket_get<x86_avx512_bitmask_find>)->BM_ARGS;
#endif

BENCHMARK_MAIN();
