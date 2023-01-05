#include <algorithm>
#include <bitset>
#include <cassert>
#include <iostream>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"

template <typename InputT, typename ElementT>
using DefaultMaskT = typename UnsignedInt<(sizeof(InputT) / sizeof(ElementT) + 7) / 8>::T;

template <typename InputT_, typename ElementT_, typename MaskT_ = DefaultMaskT<InputT_, ElementT_>>
struct naive_scalar_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;
  using InputT = InputT_;

  MaskT operator()(const InputT& input1, const InputT& input2) {
    const auto* __restrict input1_typed = reinterpret_cast<const ElementT*>(input1.data());
    const auto* __restrict input2_typed = reinterpret_cast<const ElementT*>(input2.data());

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
template <size_t VECTOR_BITS, typename InputT_, typename ElementT_, typename MaskT_ = DefaultMaskT<InputT_, ElementT_>>
struct clang_vector_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;
  using InputT = InputT_;

  static constexpr size_t VECTOR_BYTES = VECTOR_BITS / 8;
  static constexpr size_t NUM_VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(ElementT);

  using VecT = typename GccVec<ElementT, VECTOR_BYTES>::T;
  using MaskVecT = typename ClangBitmask<NUM_VECTOR_ELEMENTS>::T;

  using SingleComparisonResultT = typename UnsignedInt<sizeof(MaskVecT)>::T;

  MaskT operator()(const InputT& input1, const InputT& input2) {
    if constexpr (sizeof(VecT) == sizeof(InputT)) {
      const auto* __restrict input1_vec = reinterpret_cast<const VecT*>(input1.data());
      const auto* __restrict input2_vec = reinterpret_cast<const VecT*>(input2.data());
      const MaskVecT result_vec = __builtin_convertvector(*input1_vec == *input2_vec, MaskVecT);

      MaskT result = reinterpret_cast<const MaskT&>(result_vec);
      if constexpr (NUM_VECTOR_ELEMENTS != 8 * sizeof(result)) {
        result &= (1 << NUM_VECTOR_ELEMENTS) - 1;
      }
      return result;
    } else {
      static_assert(sizeof(InputT) % sizeof(VecT) == 0);
      constexpr size_t iterations = sizeof(InputT) / sizeof(VecT);

      const auto* __restrict input1_vec = reinterpret_cast<const VecT*>(input1.data());
      const auto* __restrict input2_vec = reinterpret_cast<const VecT*>(input2.data());

      MaskT result = 0;
      for (size_t i = 0; i < iterations; ++i) {
        MaskVecT subresult_vec = __builtin_convertvector(input1_vec[i] == input2_vec[i], MaskVecT);
        MaskT subresult = reinterpret_cast<SingleComparisonResultT&>(subresult_vec);
        if constexpr (NUM_VECTOR_ELEMENTS != 8 * sizeof(SingleComparisonResultT)) {
          subresult &= (1 << NUM_VECTOR_ELEMENTS) - 1;
        }
        assert(static_cast<unsigned int>(std::countl_zero(subresult)) >= 8 * sizeof(MaskT) - NUM_VECTOR_ELEMENTS);
        const size_t offset = i * NUM_VECTOR_ELEMENTS;
        result |= subresult << offset;
      }
      return result;
    }
  }
};

template <size_t VECTOR_BITS>
struct sized_clang_vector_bitmask {
  template <typename InputT_, typename ElementT_>
  using Benchmark = clang_vector_bitmask<VECTOR_BITS, InputT_, ElementT_>;
};

#endif

template <typename InputT_, typename ElementT_, typename MaskT_ = DefaultMaskT<InputT_, ElementT_>>
struct bitset_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;
  using InputT = InputT_;

  MaskT operator()(const InputT& input1, const InputT& input2) {
    const auto* __restrict input1_typed = reinterpret_cast<const ElementT*>(input1.data());
    const auto* __restrict input2_typed = reinterpret_cast<const ElementT*>(input2.data());

    static_assert(sizeof(InputT) % sizeof(ElementT) == 0);
    constexpr size_t iterations = sizeof(InputT) / sizeof(ElementT);

    std::bitset<sizeof(MaskT) * 8> result;
    for (size_t i = 0; i < iterations; ++i) {
      result[i] = input1_typed[i] == input2_typed[i];
    }
    return result.to_ullong();
  }
};

template <size_t VECTOR_BITS, typename InputT_, typename ElementT_, typename MaskT_ = DefaultMaskT<InputT_, ElementT_>>
struct gcc_vector_bitmask {
  // TODO
};

#if defined(__aarch64__)
#include <arm_neon.h>

template <typename InputT_, typename ElementT_, typename MaskT_ = DefaultMaskT<InputT_, ElementT_>>
struct neon_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;
  using InputT = InputT_;

  static constexpr size_t NUM_VECTOR_BYTES = 16;
  static constexpr size_t NUM_VECTOR_ELEMENTS = NUM_VECTOR_BYTES / sizeof(ElementT);

  MaskT operator()(const InputT& input1, const InputT& input2) {
    auto get_sub_mask = [&](auto in1, auto in2) {
      if constexpr (sizeof(ElementT) == 1) {
        static_assert(sizeof(MaskT) >= 2, "Need at least 16 bits for uint8.");
        using VecT = uint8x16_t;
        constexpr VecT mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
        MaskT result = 0;
        VecT matches = vceqq_u8(reinterpret_cast<const VecT&>(in1), reinterpret_cast<const VecT&>(in2));
        VecT masked_matches = vandq_u8(matches, mask);
        result |= vaddv_u8(vget_low_u8(masked_matches));
        result |= static_cast<MaskT>(vaddv_u8(vget_high_u8(masked_matches))) << 8;
        return result;
      } else if constexpr (sizeof(ElementT) == 2) {
        using VecT = uint16x8_t;
        constexpr VecT mask = {1, 2, 4, 8, 16, 32, 64, 128};
        auto matches = vceqq_u16(reinterpret_cast<const VecT&>(in1), reinterpret_cast<const VecT&>(in2));
        return vaddvq_u16(vandq_u16(matches, mask));
      } else if constexpr (sizeof(ElementT) == 4) {
        using VecT = uint32x4_t;
        constexpr VecT mask = {1, 2, 4, 8};
        auto matches = vceqq_u32(reinterpret_cast<const VecT&>(in1), reinterpret_cast<const VecT&>(in2));
        return vaddvq_u32(vandq_u32(matches, mask));
      } else if constexpr (sizeof(ElementT) == 8) {
        using VecT = uint64x2_t;
        constexpr VecT mask = {1, 2};
        auto matches = vceqq_u64(reinterpret_cast<const VecT&>(in1), reinterpret_cast<const VecT&>(in2));
        return vaddvq_u64(vandq_u64(matches, mask));
      }
    };

    if constexpr (sizeof(InputT) == NUM_VECTOR_BYTES) {
      return get_sub_mask(input1, input2);
    } else {
      static_assert(sizeof(InputT) % NUM_VECTOR_BYTES == 0);
      const auto* __restrict input1_vec = reinterpret_cast<const uint8x16_t*>(input1.data());
      const auto* __restrict input2_vec = reinterpret_cast<const uint8x16_t*>(input2.data());

      constexpr size_t iterations = sizeof(InputT) / NUM_VECTOR_BYTES;
      MaskT result = 0;
      for (size_t i = 0; i < iterations; ++i) {
        MaskT sub_mask = get_sub_mask(input1_vec[i], input2_vec[i]);
        const size_t offset = i * NUM_VECTOR_ELEMENTS;
        result |= sub_mask << offset;
      }
      return result;
    }
  }
};

#endif

#if defined(__x86_64__)

#include <immintrin.h>

template <typename InputT_, typename ElementT_, typename MaskT_ = DefaultMaskT<InputT_, ElementT_>>
struct x86_128_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;
  using InputT = InputT_;

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
template <typename InputT_, typename ElementT_, typename MaskT_ = DefaultMaskT<InputT_, ElementT_>>
struct x86_512_bitmask {
  using ElementT = ElementT_;
  using MaskT = MaskT_;
  using InputT = InputT_;

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
  using InputT = typename BenchFunc::InputT;
  InputT input1;
  InputT input2;

  std::mt19937_64 rng{std::random_device{}()};
  for (auto& value : input1.arr) {
    value = rng();
  }

  input2 = input1;
  for (auto& value : input2.arr) {
    if (rng() % 2 == 0) {
      value = rng();
    }
  }

  BenchFunc bench_fn{};
  constexpr size_t NUM_MASK_BITS = sizeof(typename BenchFunc::MaskT) * 8;

  auto naive_result =
      naive_scalar_bitmask<InputT, typename BenchFunc::ElementT, typename BenchFunc::MaskT>{}(input1, input2);
  TRACE_DO(std::cout << "Naive: " << std::bitset<NUM_MASK_BITS>(naive_result) << std::endl;);
  auto specialized_result = bench_fn(input1, input2);

  if (naive_result != specialized_result) {
    throw std::runtime_error("Bad mask computation. Expected: " + std::bitset<NUM_MASK_BITS>(naive_result).to_string() +
                             " but got: " + std::bitset<NUM_MASK_BITS>(specialized_result).to_string());
  }

  benchmark::DoNotOptimize(input1.data());
  benchmark::DoNotOptimize(input2.data());

  for (auto _ : state) {
    const auto result = bench_fn(input1, input2);
    benchmark::DoNotOptimize(result);
  }
}

using Input16Byte = AlignedArray<uint8_t, 16, 16>;
using Input64Byte = AlignedArray<uint8_t, 64, 64>;

#define BM_ARGS Unit(benchmark::kNanosecond)

#define BENCHMARK_WITH_INPUT(bm, input)                           \
  BENCHMARK(BM_compare_to_bitmask<bm<input, uint8_t>>)->BM_ARGS;  \
  BENCHMARK(BM_compare_to_bitmask<bm<input, uint16_t>>)->BM_ARGS; \
  BENCHMARK(BM_compare_to_bitmask<bm<input, uint32_t>>)->BM_ARGS; \
  BENCHMARK(BM_compare_to_bitmask<bm<input, uint64_t>>)->BM_ARGS

/////////////////////////
///   16 Byte Input   ///
/////////////////////////
BENCHMARK_WITH_INPUT(naive_scalar_bitmask, Input16Byte);
BENCHMARK_WITH_INPUT(bitset_bitmask, Input16Byte);

#if CLANG_COMPILER
BENCHMARK_WITH_INPUT(sized_clang_vector_bitmask<128>::Benchmark, Input16Byte);
#endif

#if defined(__aarch64__)
BENCHMARK_WITH_INPUT(neon_bitmask, Input16Byte);
#endif

#if defined(__x86_64__)
BENCHMARK_WITH_INPUT(x86_128_bitmask, Input16Byte);
#endif

/////////////////////////
///   64 Byte Input   ///
/////////////////////////
BENCHMARK_WITH_INPUT(naive_scalar_bitmask, Input64Byte);
BENCHMARK_WITH_INPUT(bitset_bitmask, Input64Byte);

#if CLANG_COMPILER
BENCHMARK(BM_compare_to_bitmask<sized_clang_vector_bitmask<256>::Benchmark<Input64Byte, uint8_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<sized_clang_vector_bitmask<256>::Benchmark<Input64Byte, uint16_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<sized_clang_vector_bitmask<256>::Benchmark<Input64Byte, uint32_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<sized_clang_vector_bitmask<256>::Benchmark<Input64Byte, uint64_t>>)->BM_ARGS;

BENCHMARK(BM_compare_to_bitmask<sized_clang_vector_bitmask<512>::Benchmark<Input64Byte, uint8_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<sized_clang_vector_bitmask<512>::Benchmark<Input64Byte, uint16_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<sized_clang_vector_bitmask<512>::Benchmark<Input64Byte, uint32_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<sized_clang_vector_bitmask<512>::Benchmark<Input64Byte, uint64_t>>)->BM_ARGS;
#endif

#if defined(__aarch64__)
BENCHMARK_WITH_INPUT(neon_bitmask, Input64Byte);
#endif

#if defined(__x86_64__)
BENCHMARK_WITH_INPUT(x86_128_bitmask, Input64Byte);
#endif

#if defined(AVX512_AVAILABLE)
BENCHMARK(BM_compare_to_bitmask<x86_512_bitmask<Input64Byte, uint8_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<x86_512_bitmask<Input64Byte, uint16_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<x86_512_bitmask<Input64Byte, uint32_t>>)->BM_ARGS;
BENCHMARK(BM_compare_to_bitmask<x86_512_bitmask<Input64Byte, uint64_t>>)->BM_ARGS;
#endif

BENCHMARK_MAIN();
