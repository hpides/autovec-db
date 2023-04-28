#include <bitset>
#include <cassert>
#include <limits>
#include <random>
#include <stdexcept>

#include "benchmark/benchmark.h"
#include "simd.hpp"

template <typename InputT>
using DefaultMaskT = typename UnsignedInt<(sizeof(InputT) / sizeof(typename InputT::DataT) + 7) / 8>::T;

template <typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct naive {
  using MaskT = MaskT_;
  using InputT = InputT_;

  MaskT operator()(const InputT& input1, const InputT& input2) {
    const auto* __restrict input1_typed = input1.data();
    const auto* __restrict input2_typed = input2.data();

    constexpr size_t ITERATIONS = sizeof(InputT) / sizeof(typename InputT::DataT);

    MaskT result = 0;
    for (size_t i = 0; i < ITERATIONS; ++i) {
      if (input1_typed[i] == input2_typed[i]) {
        result |= 1ull << i;
      }
    }
    return result;
  }
};

template <typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct clang {
  using MaskT = MaskT_;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;

  static constexpr size_t VECTOR_BITS = 128;
  static constexpr size_t VECTOR_BYTES = VECTOR_BITS / 8;
  static constexpr size_t NUM_VECTOR_ELEMENTS = VECTOR_BYTES / sizeof(ElementT);

  using VecT = typename simd::GccVec<ElementT, VECTOR_BYTES>::T;
  using MaskVecT = typename simd::ClangBitmask<NUM_VECTOR_ELEMENTS>::T;

  using SingleComparisonResultT = typename UnsignedInt<sizeof(MaskVecT)>::T;

  MaskT operator()(const InputT& input1, const InputT& input2) {
    const MaskVecT subresult_vec = __builtin_convertvector(
        reinterpret_cast<const VecT&>(input1) == reinterpret_cast<const VecT&>(input2), MaskVecT);
    return reinterpret_cast<const SingleComparisonResultT&>(subresult_vec);
  }
};

template <typename InputT_, typename MaskT_ = DefaultMaskT<InputT_>>
struct neon {
  using MaskT = MaskT_;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;

  using VecT = typename simd::NeonVecT<sizeof(ElementT)>::T;

  struct GetSubresultMask {
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
    return GetSubresultMask{}(reinterpret_cast<const VecT&>(input1), reinterpret_cast<const VecT&>(input2));
  }
};

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

template <typename InputT_>
struct llvm_current16x8_with_2addv {
  using MaskT = uint16_t;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;
  using VecT = typename simd::NeonVecT<sizeof(ElementT)>::T;
  static constexpr uint8x16_t mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};

  uint16_t operator()(const VecT& a, const VecT& b) {
    auto matches = vceqq_u8(a, b);
    auto masked_matches = vandq_u8(matches, mask);
    uint16_t result = 0;
    result |= vaddv_u8(vget_low_u8(masked_matches));
    result |= (vaddv_u8(vget_high_u8(masked_matches))) << 8;
    return result;
  }

  MaskT operator()(const InputT& input1, const InputT& input2) {
    return this->operator()(reinterpret_cast<const VecT&>(input1), reinterpret_cast<const VecT&>(input2));
  }
};

template <typename InputT_>
struct llvm_suggested16x8_with_zip_combine {
  using MaskT = uint16_t;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;
  using VecT = typename simd::NeonVecT<sizeof(ElementT)>::T;
  static constexpr uint8x16_t mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};

  uint16_t operator()(const VecT& a, const VecT& b) {
    auto matches = vceqq_u8(a, b);
    auto masked_matches = vandq_u8(matches, mask);
    // We need vzip_u8 here and not vzip1 because that would lose half of the values.
    // The return type is uint8x8x2_t, so we have to combine it.
    // This is then actually combined quite nicely to a vtbl instruction.
    // See method below, which has idential assembly.
    auto zipped = vzip_u8(vget_low_u8(masked_matches), vget_high_u8(masked_matches));
    auto combined = vcombine_u8(zipped.val[0], zipped.val[1]);
    return vaddvq_u16(combined);
  }

  MaskT operator()(const InputT& input1, const InputT& input2) {
    return this->operator()(reinterpret_cast<const VecT&>(input1), reinterpret_cast<const VecT&>(input2));
  }
};

template <typename InputT_>
struct llvm_suggested16x8_with_shuffle {
  using MaskT = uint16_t;
  using InputT = InputT_;
  using ElementT = typename InputT::DataT;
  using VecT = typename simd::NeonVecT<sizeof(ElementT)>::T;
  static constexpr uint8x16_t mask = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};

  uint16_t operator()(const VecT& a, const VecT& b) {
    auto matches = vceqq_u8(a, b);
    auto masked_matches = vandq_u8(matches, mask);
    auto x =
        __builtin_shufflevector(masked_matches, masked_matches, 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    return vaddvq_u16(x);
  }

  MaskT operator()(const InputT& input1, const InputT& input2) {
    return this->operator()(reinterpret_cast<const VecT&>(input1), reinterpret_cast<const VecT&>(input2));
  }
};

/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

template <typename BenchFunc>
void bm_bitmask(benchmark::State& state) {
  using InputT = typename BenchFunc::InputT;
  InputT input1;
  InputT input2;

  std::mt19937_64 rng{std::random_device{}()};
  for (auto& value : input1) {
    value = rng();
  }

  input2 = input1;
  for (auto& value : input2) {
    if (rng() % 2 == 0) {
      value = rng();
    }
  }

  BenchFunc bench_fn{};
  constexpr size_t NUM_MASK_BITS = sizeof(typename BenchFunc::MaskT) * 8;

  auto naive_result = naive<InputT, typename BenchFunc::MaskT>{}(input1, input2);
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

  state.counters["PerValue"] =
      benchmark::Counter(state.iterations() * input1.size(), benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

using v16xi8 = AlignedArray<uint8_t, 16, 16>;
using v8i16 = AlignedArray<uint16_t, 8, 16>;
using v4i32 = AlignedArray<uint32_t, 4, 16>;
using v2i64 = AlignedArray<uint64_t, 2, 16>;

#define BM_ARGS Unit(benchmark::kNanosecond)

// NOLINTBEGIN(bugprone-macro-parentheses): `bm` is a template here, we can't put it in parentheses
#define BENCHMARK_WITH_16B_INPUT(bm)                                 \
  BENCHMARK(bm_bitmask<bm<v16xi8>>)->BM_ARGS; \
  BENCHMARK(bm_bitmask<bm<v8i16>>)->BM_ARGS;  \
  BENCHMARK(bm_bitmask<bm<v4i32>>)->BM_ARGS;  \
  BENCHMARK(bm_bitmask<bm<v2i64>>)->BM_ARGS
// NOLINTEND(bugprone-macro-parentheses)

/////////////////////////
///   16 Byte Input   ///
/////////////////////////
BENCHMARK_WITH_16B_INPUT(naive);
BENCHMARK_WITH_16B_INPUT(neon);
BENCHMARK_WITH_16B_INPUT(clang);

BENCHMARK(bm_bitmask<llvm_current16x8_with_2addv<v16xi8>>)->BM_ARGS;
BENCHMARK(bm_bitmask<llvm_suggested16x8_with_shuffle<v16xi8>>)->BM_ARGS;
BENCHMARK(bm_bitmask<llvm_suggested16x8_with_zip_combine<v16xi8>>)->BM_ARGS;

BENCHMARK_MAIN();
