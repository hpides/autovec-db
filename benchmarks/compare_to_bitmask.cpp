#include <algorithm>
#include <bitset>
#include <cassert>
#include <iostream>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"

template <typename SubresultFunc, typename InputT>
auto create_bitmask_using_subresults(const InputT& input1, const InputT& input2) {
  SubresultFunc subresult_func;
  constexpr size_t VECTOR_BYTES = sizeof(typename SubresultFunc::InputT);

  const auto* __restrict input1_vec = reinterpret_cast<const typename SubresultFunc::InputT*>(input1.data());
  const auto* __restrict input2_vec = reinterpret_cast<const typename SubresultFunc::InputT*>(input2.data());

  if constexpr (sizeof(InputT) == VECTOR_BYTES) {
    return subresult_func(*input1_vec, *input2_vec);
  } else {
    static_assert(sizeof(InputT) % VECTOR_BYTES == 0);
    constexpr size_t iterations = sizeof(InputT) / VECTOR_BYTES;

    using MaskT = decltype(subresult_func(input1_vec[0], input2_vec[0]));
    constexpr size_t NUM_VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(typename InputT::DataT);

    MaskT result = 0;
    for (size_t i = 0; i < iterations; ++i) {
      MaskT sub_mask = subresult_func(input1_vec[i], input2_vec[i]);

      // sub mask can only have its NUM_VECTOR_ELEMENTS least significant bits set.
      assert(static_cast<unsigned int>(std::countl_zero(sub_mask)) >= 8 * sizeof(MaskT) - NUM_VECTOR_ELEMENTS);

      const size_t offset = i * NUM_VECTOR_ELEMENTS;
      result |= sub_mask << offset;
    }
    return result;
  }
}

template <typename InputT>
using DefaultMaskT = typename UnsignedInt<(sizeof(InputT) / sizeof(typename InputT::DataT) + 7) / 8>::T;

template <typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
#if GCC_COMPILER
  __attribute__((optimize("no-tree-vectorize")))
#endif
struct naive_scalar_bitmask {
  using MaskT = MaskT_;
  using InputT = InputT_;

  MaskT operator()(const InputT& input1, const InputT& input2) {
    const auto* __restrict input1_typed = input1.data();
    const auto* __restrict input2_typed = input2.data();

    constexpr size_t iterations = sizeof(InputT) / sizeof(typename InputT::DataT);

    MaskT result = 0;
#if CLANG_COMPILER
#pragma clang loop vectorize(disable)
#endif
    for (size_t i = 0; i < iterations; ++i) {
      if (input1_typed[i] == input2_typed[i]) {
        result |= 1ull << i;
      }
    }
    return result;
  }
};

template <typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct autovec_scalar_bitmask {
  using MaskT = MaskT_;
  using InputT = InputT_;

  // same code as for naive scalar, just with vectorization enabled
  // As far as we know, there is no good way to get this autovectorized
  // see https://stackoverflow.com/questions/75030873
  MaskT operator()(const InputT& input1, const InputT& input2) {
    const auto* __restrict input1_typed = input1.data();
    const auto* __restrict input2_typed = input2.data();

    constexpr size_t iterations = sizeof(InputT) / sizeof(typename InputT::DataT);

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
template <size_t VECTOR_BITS, typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct clang_vector_bitmask {
  using MaskT = MaskT_;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;

  static constexpr size_t VECTOR_BYTES = VECTOR_BITS / 8;
  static constexpr size_t NUM_VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(ElementT);

  using VecT = typename GccVec<ElementT, VECTOR_BYTES>::T;
  using MaskVecT = typename ClangBitmask<NUM_VECTOR_ELEMENTS>::T;

  using SingleComparisonResultT = typename UnsignedInt<sizeof(MaskVecT)>::T;

  struct GetSubresulMask {
    using InputT = VecT;

    MaskT operator()(const InputT& subinput1, const InputT& subinput2) {
      MaskVecT subresult_vec = __builtin_convertvector(subinput1 == subinput2, MaskVecT);
      MaskT subresult = reinterpret_cast<SingleComparisonResultT&>(subresult_vec);

      if constexpr (NUM_VECTOR_ELEMENTS != 8 * sizeof(SingleComparisonResultT)) {
        // TODO: Clang codegen isn't really good for 128 vectors with 8x8B / 16x4B input, and the masking here doesn't
        // seem to help.
        subresult &= (1 << NUM_VECTOR_ELEMENTS) - 1;
      }

      return subresult;
    }
  };

  MaskT operator()(const InputT& input1, const InputT& input2) {
    return create_bitmask_using_subresults<GetSubresulMask>(input1, input2);
  }
};

template <size_t VECTOR_BITS>
struct sized_clang_vector_bitmask {
  template <typename InputT_>
  using Benchmark = clang_vector_bitmask<VECTOR_BITS, InputT_>;
};
#endif

template <typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct bitset_bitmask {
  using MaskT = MaskT_;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;

  MaskT operator()(const InputT& input1, const InputT& input2) {
    const auto* __restrict input1_typed = input1.data();
    const auto* __restrict input2_typed = input2.data();

    constexpr size_t iterations = sizeof(InputT) / sizeof(ElementT);

    std::bitset<sizeof(MaskT) * 8> result;
    for (size_t i = 0; i < iterations; ++i) {
      result[i] = input1_typed[i] == input2_typed[i];
    }
    return result.to_ullong();
  }
};

template <size_t VECTOR_BITS, typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct gcc_vector_bitmask {
  // TODO
};

#if defined(__aarch64__)
template <typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct neon_bitmask {
  using MaskT = MaskT_;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;

  using VecT = NeonVecT<sizeof(ElementT)>::T;

  struct GetSubresulMask {
    using InputT = VecT;

    MaskT operator()(const InputT& subinput1, const InputT& subinput2) {
      if constexpr (sizeof(ElementT) == 1) {
        constexpr VecT mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
        MaskT result = 0;
        VecT matches = vceqq_u8(subinput1, subinput2);
        VecT masked_matches = vandq_u8(matches, mask);
        result |= vaddv_u8(vget_low_u8(masked_matches));
        result |= static_cast<MaskT>(vaddv_u8(vget_high_u8(masked_matches))) << 8;
        return result;

      } else if constexpr (sizeof(ElementT) == 2) {
        constexpr VecT mask = {1, 2, 4, 8, 16, 32, 64, 128};
        return vaddvq_u16(vandq_u16(vceqq_u16(subinput1, subinput2), mask));

      } else if constexpr (sizeof(ElementT) == 4) {
        constexpr VecT mask = {1, 2, 4, 8};
        return vaddvq_u32(vandq_u32(vceqq_u32(subinput1, subinput2), mask));

      } else if constexpr (sizeof(ElementT) == 8) {
        constexpr VecT mask = {1, 2};
        return vaddvq_u64(vandq_u64(vceqq_u64(subinput1, subinput2), mask));
      }
    }
  };

  MaskT operator()(const InputT& input1, const InputT& input2) {
    return create_bitmask_using_subresults<GetSubresulMask>(input1, input2);
  }
};
#endif

#if defined(__x86_64__)
template <typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct x86_128_bitmask {
  using MaskT = MaskT_;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;

  using VecT = __m128i;
  static constexpr size_t NUM_VECTOR_ELEMENTS = sizeof(VecT) / sizeof(ElementT);

  struct GetSubresulMask {
    using InputT = VecT;

    MaskT operator()(const InputT& subinput1, const InputT& subinput2) {
      VecT vector_compare_result = [&]() {
        if constexpr (sizeof(ElementT) == 1) {
          return _mm_cmpeq_epi8(subinput1, subinput2);
        } else if constexpr (sizeof(ElementT) == 2) {
          return _mm_cmpeq_epi16(subinput1, subinput2);
        } else if constexpr (sizeof(ElementT) == 4) {
          return _mm_cmpeq_epi32(subinput1, subinput2);
        } else if constexpr (sizeof(ElementT) == 8) {
          return _mm_cmpeq_epi64(subinput1, subinput2);
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
      return subresult;
    }
  };

  MaskT operator()(const InputT& input1, const InputT& input2) {
    return create_bitmask_using_subresults<GetSubresulMask>(input1, input2);
  }
};

#if defined(AVX512_AVAILABLE)
template <typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct x86_512_bitmask {
  using MaskT = MaskT_;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;

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

  auto naive_result = naive_scalar_bitmask<InputT, typename BenchFunc::MaskT>{}(input1, input2);
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

using Input_64B_as_64x1B = AlignedArray<uint8_t, 64, 64>;
using Input_64B_as_32x2B = AlignedArray<uint16_t, 32, 64>;
using Input_64B_as_16x4B = AlignedArray<uint32_t, 16, 64>;
using Input_64B_as_8x8B = AlignedArray<uint64_t, 8, 64>;

using Input_16B_as_16x1B = AlignedArray<uint8_t, 16, 16>;
using Input_16B_as_8x2B = AlignedArray<uint16_t, 8, 16>;
using Input_16B_as_4x4B = AlignedArray<uint32_t, 4, 16>;
using Input_16B_as_2x8B = AlignedArray<uint64_t, 2, 16>;

#define BM_ARGS Unit(benchmark::kNanosecond)

#define BENCHMARK_WITH_64B_INPUT(bm)                                 \
  BENCHMARK(BM_compare_to_bitmask<bm<Input_64B_as_64x1B>>)->BM_ARGS; \
  BENCHMARK(BM_compare_to_bitmask<bm<Input_64B_as_32x2B>>)->BM_ARGS; \
  BENCHMARK(BM_compare_to_bitmask<bm<Input_64B_as_16x4B>>)->BM_ARGS; \
  BENCHMARK(BM_compare_to_bitmask<bm<Input_64B_as_8x8B>>)->BM_ARGS

#define BENCHMARK_WITH_16B_INPUT(bm)                                 \
  BENCHMARK(BM_compare_to_bitmask<bm<Input_16B_as_16x1B>>)->BM_ARGS; \
  BENCHMARK(BM_compare_to_bitmask<bm<Input_16B_as_8x2B>>)->BM_ARGS;  \
  BENCHMARK(BM_compare_to_bitmask<bm<Input_16B_as_4x4B>>)->BM_ARGS;  \
  BENCHMARK(BM_compare_to_bitmask<bm<Input_16B_as_2x8B>>)->BM_ARGS

/////////////////////////
///   16 Byte Input   ///
/////////////////////////
BENCHMARK_WITH_16B_INPUT(naive_scalar_bitmask);
BENCHMARK_WITH_16B_INPUT(autovec_scalar_bitmask);
BENCHMARK_WITH_16B_INPUT(bitset_bitmask);

#if CLANG_COMPILER
BENCHMARK_WITH_16B_INPUT(sized_clang_vector_bitmask<128>::Benchmark);
#endif

#if defined(__aarch64__)
BENCHMARK_WITH_16B_INPUT(neon_bitmask);
#endif

#if defined(__x86_64__)
BENCHMARK_WITH_16B_INPUT(x86_128_bitmask);
#endif

/////////////////////////
///   64 Byte Input   ///
/////////////////////////
BENCHMARK_WITH_64B_INPUT(naive_scalar_bitmask);
BENCHMARK_WITH_64B_INPUT(autovec_scalar_bitmask);
BENCHMARK_WITH_64B_INPUT(bitset_bitmask);

#if CLANG_COMPILER
BENCHMARK_WITH_64B_INPUT(sized_clang_vector_bitmask<128>::Benchmark);
BENCHMARK_WITH_64B_INPUT(sized_clang_vector_bitmask<256>::Benchmark);
BENCHMARK_WITH_64B_INPUT(sized_clang_vector_bitmask<512>::Benchmark);
#endif

#if defined(__aarch64__)
BENCHMARK_WITH_64B_INPUT(neon_bitmask);
#endif

#if defined(__x86_64__)
BENCHMARK_WITH_64B_INPUT(x86_128_bitmask);
#endif

#if defined(AVX512_AVAILABLE)
BENCHMARK_WITH_64B_INPUT(x86_512_bitmask);
#endif

BENCHMARK_MAIN();
