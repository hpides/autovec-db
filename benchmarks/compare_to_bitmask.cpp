#include <bitset>
#include <cassert>
#include <iostream>
#include <random>

#include "benchmark/benchmark.h"
#include "common.hpp"
#include "simd.hpp"

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
struct naive_scalar_bitmask {
  using MaskT = MaskT_;
  using InputT = InputT_;

#if GCC_COMPILER
  __attribute__((optimize("no-tree-vectorize")))
#endif
  MaskT
  operator()(const InputT& input1, const InputT& input2) {
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

  using VecT = typename simd::GccVec<ElementT, VECTOR_BYTES>::T;
  using MaskVecT = typename simd::ClangBitmask<NUM_VECTOR_ELEMENTS>::T;

  using SingleComparisonResultT = typename UnsignedInt<sizeof(MaskVecT)>::T;

  struct GetSubresulMask {
    using InputT = VecT;

    MaskT operator()(const InputT& subinput1, const InputT& subinput2) {
      // Note from performance investigation: On skylake, this gives 50% of the handwritten performance for 8B
      // vector elements. Clang does not use "extract single-precision to mask" (movmskps) or "extract
      // double-precision to mask" (movmskpd) instructions here, even though they would be perfectly usable and do not
      // depend on the values actually being floats.  Instead, it combines the 8B truthiness values into 1B truthiness
      // values using 3 packs and one shift.
      //
      // This does not occur if avx512 is available, which can compare and store the result in a mask register in one
      // instruction, see https://godbolt.org/z/efe8d61Gz.
      // This could be considered a performance bug in clang
      MaskVecT subresult_vec = __builtin_convertvector(subinput1 == subinput2, MaskVecT);
      MaskT subresult = reinterpret_cast<SingleComparisonResultT&>(subresult_vec);

      if constexpr (NUM_VECTOR_ELEMENTS != 8 * sizeof(SingleComparisonResultT)) {
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
  using MaskT = MaskT_;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;

  static constexpr size_t VECTOR_BYTES = VECTOR_BITS / 8;
  static constexpr size_t NUM_VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(ElementT);

  using VecT = typename GccVec<ElementT, VECTOR_BYTES>::T;

  struct GetSubresulMask {
    using InputT = VecT;

    MaskT operator()(const InputT& subinput1, const InputT& subinput2) {
      // GCC does not have a bit-bool-vector, so we can't use the clang-logic (using convert_shufflevector).
      // We also didn't find any other good way to encode this conversion. This is one approach, but we're not happy
      // with it.
      //
      // For a real codebase, it would probably make sense to have a helper function "byte_vector_to_bit_mask" that uses
      // platform-specific intrinsics. However, this pessimizes optimizations of multiple vector operations: Clang can
      // optimize
      // __builtin_convertvector(a == b, boolvector) into a single instruction, the GCC approach wouldn't do that.
      auto subresult_bool_vec = subinput1 == subinput2;
      MaskT result = 0;
      for (size_t i = 0; i < NUM_VECTOR_ELEMENTS; ++i) {
        result |= subresult_bool_vec[i] ? 1ull << i : 0;
      }
      return result;
    }
  };

  MaskT operator()(const InputT& input1, const InputT& input2) {
    return create_bitmask_using_subresults<GetSubresulMask>(input1, input2);
  }
};
template <size_t VECTOR_BITS>
struct sized_gcc_vector_bitmask {
  template <typename InputT_>
  using Benchmark = gcc_vector_bitmask<VECTOR_BITS, InputT_>;
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
  // Implementations for AVX2 and SSE would be identical here
  // (AVX2 doesn't add any useful 128bit operations)
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
          // Weirdness: Up to AVX2 (including), we do not have "extract 2B truthiness to mask" instructions

          // The comparison result is two bytes, either 0xffff or 0x0000. We just need one byte out of that.
          // Move all lower halves of the 2B elements to the bytes (0, 8), clears upper bytes.
          __m128i lower_byte_shuffle_mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);
          __m128i lower_bytes_shuffled = _mm_shuffle_epi8(vector_compare_result, lower_byte_shuffle_mask);
          return _mm_movemask_epi8(lower_bytes_shuffled);
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

template <typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct x86_256_avx2_bitmask {
  // identical to x86_128, just wider.
  // performance analysis on skylake: This creates 2 vector loads, 2 vector compares, 2 mask extractions, 1 shift, 1 or
  // the clang-vector-512bit code generation is smarter and doesn't have to keep the outer loop. It creates 2 vector
  // loads, 2 vector compares, 1 vector pack, 1 vector permute, 1 mask extract. In total, this is one fewer instruction,
  // and we observe this being ~12% faster.
  using MaskT = MaskT_;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;

  using VecT = __m256i;
  static constexpr size_t NUM_VECTOR_ELEMENTS = sizeof(VecT) / sizeof(ElementT);

  struct GetSubresulMask {
    using InputT = VecT;

    MaskT operator()(const InputT& subinput1, const InputT& subinput2) {
      VecT vector_compare_result = [&]() {
        if constexpr (sizeof(ElementT) == 1) {
          return _mm256_cmpeq_epi8(subinput1, subinput2);
        } else if constexpr (sizeof(ElementT) == 2) {
          return _mm256_cmpeq_epi16(subinput1, subinput2);
        } else if constexpr (sizeof(ElementT) == 4) {
          return _mm256_cmpeq_epi32(subinput1, subinput2);
        } else if constexpr (sizeof(ElementT) == 8) {
          return _mm256_cmpeq_epi64(subinput1, subinput2);
        }
      }();

      MaskT subresult = [&]() {
        if constexpr (sizeof(ElementT) == 1) {
          // cast to unsigned is necessary because otherwise the widening will be sign-expanding, filling in 1-bits
          return static_cast<uint32_t>(_mm256_movemask_epi8(vector_compare_result));
        } else if constexpr (sizeof(ElementT) == 2) {
          // TODO: On i5-6200U, the clang-vector-variant is slightly faster (extract to XMM, 2 vpacksswb, 2 movmsk, 1
          // shift, 1 or).
          // avx2 shuffle is super weird: You can only shuffle within a lane, indices are within the current lane.
          __m256i lower_byte_shuffle_mask = _mm256_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0,
                                                            -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);
          __m256i lower_bytes_shuffled = _mm256_shuffle_epi8(vector_compare_result, lower_byte_shuffle_mask);
          // We now have: In each lane, the lowest 8 bytes contain our 8 result values.
          auto two_half_results = _mm256_movemask_epi8(lower_bytes_shuffled);
          // 0b00000000 11011001 00000000 11011111
          uint16_t result = two_half_results | (two_half_results >> 8);
          return result;
        } else if constexpr (sizeof(ElementT) == 4) {
          return _mm256_movemask_ps(reinterpret_cast<__m256>(vector_compare_result));
        } else if constexpr (sizeof(ElementT) == 8) {
          return _mm256_movemask_pd(reinterpret_cast<__m256d>(vector_compare_result));
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

#if AVX512_AVAILABLE
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
BENCHMARK_WITH_16B_INPUT(sized_gcc_vector_bitmask<128>::Benchmark);

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
BENCHMARK_WITH_64B_INPUT(sized_gcc_vector_bitmask<128>::Benchmark);
BENCHMARK_WITH_64B_INPUT(sized_gcc_vector_bitmask<256>::Benchmark);
BENCHMARK_WITH_64B_INPUT(sized_gcc_vector_bitmask<512>::Benchmark);

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
BENCHMARK_WITH_64B_INPUT(x86_256_avx2_bitmask);
#endif

#if AVX512_AVAILABLE
BENCHMARK_WITH_64B_INPUT(x86_512_bitmask);
#endif

BENCHMARK_MAIN();
