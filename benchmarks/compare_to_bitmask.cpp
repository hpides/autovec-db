#include <algorithm>
#include <bitset>
#include <cassert>
#include <iostream>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"

using InputT = AlignedArray<uint8_t, 64, 64>;

template <typename ElementT_, typename MaskT_ = uint64_t>
struct naive_scalar_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;

  MaskT operator()(const InputT& input1, const InputT& input2) {
    const ElementT* __restrict input1_typed = reinterpret_cast<const ElementT*>(input1.data());
    const ElementT* __restrict input2_typed = reinterpret_cast<const ElementT*>(input2.data());

    static_assert(sizeof(InputT) % sizeof(ElementT) == 0);
    constexpr size_t iterations = sizeof(InputT) / sizeof(ElementT);

    MaskT result = 0;
    for (size_t i = 0; i < iterations; ++i) {
      if (input1_typed[i] == input2_typed[i]) {
        result |= 1ull << i;
      }
    }
    return result;
  }
};

#if CLANG_COMPILER
template <size_t VECTOR_BITS, typename ElementT_, typename MaskT_ = uint64_t>
struct clang_vector_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;

  static constexpr size_t VECTOR_BYTES = VECTOR_BITS / 8;
  static constexpr size_t NUM_VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(ElementT);

  using VecT = typename GccVec<ElementT, VECTOR_BYTES>::T;
  using MaskVecT = typename ClangBitmask<NUM_VECTOR_ELEMENTS>::T;

  using SingleComparisonResultT = typename UnsignedInt<sizeof(MaskVecT)>::T;

  MaskT operator()(const InputT& input1, const InputT& input2) {
    if constexpr (sizeof(VecT) == sizeof(InputT)) {
      const VecT* __restrict input1_vec = reinterpret_cast<const VecT*>(input1.data());
      const VecT* __restrict input2_vec = reinterpret_cast<const VecT*>(input2.data());
      const MaskVecT result = __builtin_convertvector(*input1_vec == *input2_vec, MaskVecT);
      return reinterpret_cast<const MaskT&>(result);
    } else {
      static_assert(sizeof(InputT) % sizeof(VecT) == 0);
      constexpr size_t iterations = sizeof(InputT) / sizeof(VecT);

      const VecT* __restrict input1_vec = reinterpret_cast<const VecT*>(input1.data());
      const VecT* __restrict input2_vec = reinterpret_cast<const VecT*>(input2.data());

      MaskT result = 0;
      for (size_t i = 0; i < iterations; ++i) {
        MaskVecT subresult_vec = __builtin_convertvector(*(input1_vec + i) == *(input2_vec + i), MaskVecT);
        MaskT subresult = reinterpret_cast<SingleComparisonResultT&>(subresult_vec);
        assert(static_cast<unsigned int>(std::countl_zero(subresult)) >= 8 * sizeof(MaskT) - NUM_VECTOR_ELEMENTS);
        const size_t offset = i * NUM_VECTOR_ELEMENTS;
        result |= subresult << offset;
      }
      return result;
    }
  }
};
#endif

template <typename ElementT_, typename MaskT_ = uint64_t>
struct bitset_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;

  MaskT operator()(const InputT& input1, const InputT& input2) {
    const ElementT* __restrict input1_typed = reinterpret_cast<const ElementT*>(input1.data());
    const ElementT* __restrict input2_typed = reinterpret_cast<const ElementT*>(input2.data());

    static_assert(sizeof(InputT) % sizeof(ElementT) == 0);
    constexpr size_t iterations = sizeof(InputT) / sizeof(ElementT);

    std::bitset<sizeof(MaskT) * 8> result;
    for (size_t i = 0; i < iterations; ++i) {
      result[i] = input1_typed[i] == input2_typed[i];
    }
    return result.to_ullong();
  }
};

template <size_t VECTOR_BITS, typename ElementT_, typename MaskT_ = uint64_t>
struct gcc_vector_bitmask {
  // TODO
};

template <typename ElementT_, typename MaskT_ = uint64_t>
struct neon_bitmask {
  // TODO
};

#if defined(__x86_64__)

#include <immintrin.h>

template <typename ElementT_, typename MaskT_ = uint64_t>
struct x86_128_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;
  using VecT = __m128i;
  static constexpr size_t NUM_VECTOR_ELEMENTS = sizeof(VecT) / sizeof(ElementT);

  MaskT operator()(const InputT& input1, const InputT& input2) {
    const auto* __restrict input1_typed = reinterpret_cast<const VecT*>(input1.data());
    const auto* __restrict input2_typed = reinterpret_cast<const VecT*>(input2.data());

    static_assert(sizeof(InputT) % sizeof(VecT) == 0);
    constexpr size_t iterations = sizeof(InputT) / sizeof(VecT);

    MaskT result = 0;
    for (size_t i = 0; i < iterations; ++i) {
      VecT vector_compare_result = [&]() {
        if constexpr (sizeof(ElementT) == 1) {
          return _mm_cmpeq_epi8(input1_typed[i], input2_typed[i]);
        } else if constexpr (sizeof(ElementT) == 2) {
          return _mm_cmpeq_epi16(input1_typed[i], input2_typed[i]);
        } else if constexpr (sizeof(ElementT) == 4) {
          return _mm_cmpeq_epi32(input1_typed[i], input2_typed[i]);
        } else if constexpr (sizeof(ElementT) == 8) {
          return _mm_cmpeq_epi64(input1_typed[i], input2_typed[i]);
        }
      }();

      MaskT subresult = [&]() {
        if constexpr (sizeof(ElementT) == 1) {
          return _mm_movemask_epi8(vector_compare_result);
        } else if constexpr (sizeof(ElementT) == 2) {
          // Moves all upper halves of the 2B elements to the bytes (0, 8), clears upper bytes.
          __m128i upper_byte_shuffle_mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 15, 13, 11, 9, 7, 5, 3, 1);
          __m128i upper_bytes_shuffled = _mm_shuffle_epi8(vector_compare_result, upper_byte_shuffle_mask);

          // Moves all lower halves of the 2B elements to the bytes (0, 8), clears upper bytes.
          __m128i lower_byte_shuffle_mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);
          __m128i lower_bytes_shuffled = _mm_shuffle_epi8(vector_compare_result, lower_byte_shuffle_mask);

          // AND them together to get 2B-value equality
          __m128i int16_equality_vec = _mm_and_si128(upper_bytes_shuffled, lower_bytes_shuffled);

          // extract to integer bitmask
          return _mm_movemask_epi8(int16_equality_vec);

        } else if constexpr (sizeof(ElementT) == 4) {
          return _mm_movemask_ps(reinterpret_cast<__m128>(vector_compare_result));
        } else if constexpr (sizeof(ElementT) == 8) {
          return _mm_movemask_pd(reinterpret_cast<__m128d>(vector_compare_result));
        }
      }();

      assert(static_cast<unsigned int>(std::countl_zero(subresult)) >= 8 * sizeof(MaskT) - NUM_VECTOR_ELEMENTS);

      const size_t offset = i * NUM_VECTOR_ELEMENTS;
      result |= subresult << offset;
    }
    return result;
  }
};

#if defined(AVX512_AVAILABLE)
template <typename ElementT_, typename MaskT_ = uint64_t>
struct x86_512_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;

  using VecT = __m512i;

  static_assert(sizeof(VecT) == sizeof(InputT));

  MaskT operator()(const InputT& input1, const InputT& input2) {
    const auto* __restrict input1_typed = reinterpret_cast<const VecT*>(input1.data());
    const auto* __restrict input2_typed = reinterpret_cast<const VecT*>(input2.data());

    if constexpr (sizeof(ElementT) == 1) {
      return _mm512_cmpeq_epi8_mask(*input1_typed, *input2_typed);
    } else if constexpr (sizeof(ElementT) == 2) {
      return _mm512_cmpeq_epi16_mask(*input1_typed, *input2_typed);
    } else if constexpr (sizeof(ElementT) == 4) {
      return _mm512_cmpeq_epi32_mask(*input1_typed, *input2_typed);
    } else if constexpr (sizeof(ElementT) == 8) {
      return _mm512_cmpeq_epi64_mask(*input1_typed, *input2_typed);
    }
  }
};
#endif

#endif

template <typename BenchFunc>
void BM_compare_to_bitmask(benchmark::State& state) {
  InputT input1;
  InputT input2;

  std::mt19937_64 rng{std::random_device{}()};
  std::ranges::generate(input1.arr, std::ref(rng));
  input2 = input1;
  for (auto& value : input2.arr) {
    if (rng() % 2 == 0) {
      value = rng();
    }
  }

  BenchFunc bench_fn{};

  auto naive_result = naive_scalar_bitmask<typename BenchFunc::ElementT, typename BenchFunc::MaskT>{}(input1, input2);
  auto specialized_result = bench_fn(input1, input2);
  if (naive_result != specialized_result) {
    throw std::runtime_error("Bad mask computation");
  }

  benchmark::DoNotOptimize(input1.data());
  benchmark::DoNotOptimize(input2.data());

  for (auto _ : state) {
    const auto result = bench_fn(input1, input2);
    benchmark::DoNotOptimize(result);
  }
}

#define BM_ARGS Repetitions(1)

BENCHMARK(BM_compare_to_bitmask<naive_scalar_bitmask<uint8_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<naive_scalar_bitmask<uint16_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<naive_scalar_bitmask<uint32_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<naive_scalar_bitmask<uint64_t>>)->BM_ARGS;

BENCHMARK(BM_compare_to_bitmask<bitset_bitmask<uint8_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<bitset_bitmask<uint16_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<bitset_bitmask<uint32_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<bitset_bitmask<uint64_t>>)->BM_ARGS;

#if CLANG_COMPILER
BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<128, uint8_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<128, uint16_t>>)->BM_ARGS;
// TODO: Currently broken. Sub-Byte bool-vectors seem to be broken in many ways.
// BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<128, uint32_t>>)->BM_ARGS;

BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<256, uint8_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<256, uint16_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<256, uint32_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<256, uint64_t>>)->BM_ARGS;

BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<512, uint8_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<512, uint16_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<512, uint32_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<clang_vector_bitmask<512, uint64_t>>)->BM_ARGS;
#endif

#if defined(__x86_64__)
BENCHMARK(BM_compare_to_bitmask<x86_128_bitmask<uint8_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<x86_128_bitmask<uint16_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<x86_128_bitmask<uint32_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<x86_128_bitmask<uint64_t>>)->BM_ARGS;

#if defined(AVX512_AVAILABLE)
BENCHMARK(BM_compare_to_bitmask<x86_512_bitmask<uint8_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<x86_512_bitmask<uint16_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<x86_512_bitmask<uint32_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<x86_512_bitmask<uint64_t>>)->BM_ARGS;
#endif

#endif

BENCHMARK_MAIN();
