#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

#if (defined(__GNUC__) && !defined(__clang__))
#define GCC_COMPILER 1
#else
#define GCC_COMPILER 0
#endif

#if (defined(__clang__))
#define CLANG_COMPILER 1
#else
#define CLANG_COMPILER 0
#endif

#ifdef ENABLE_DEBUG_DO
#define DEBUG_DO(block) \
  do {                  \
    block               \
  } while (0)
#else
#define DEBUG_DO(block) (void)0
#endif

// This is used for stuff that you really only want to see when you are debugging actively. To use this, either pass
// -DENABLE_TRACE_DO or uncomment the #define line below.
// #define ENABLE_TRACE_DO
#ifdef ENABLE_TRACE_DO
#define TRACE_DO(block) \
  do {                  \
    block               \
  } while (0)
#else
#define TRACE_DO(block) (void)0
#endif

// We assume that _if_ a server has AVX512, it has everything we need.
// To check this, we use the AVX512 Foundation.
#if (defined(__AVX512F__))
#define AVX512_AVAILABLE 1
#else
#define AVX512_AVAILABLE 0
#endif

template <typename DataT_, size_t NUM_ENTRIES_, size_t ALIGN>
struct alignas(ALIGN) AlignedArray {
  using DataT = DataT_;
  static constexpr size_t NUM_ENTRIES = NUM_ENTRIES_;

  // We want to use an empty custom constructor here to avoid zeroing the array when creating an AlignedArray.
  AlignedArray() {}

  std::array<DataT, NUM_ENTRIES> arr;

  DataT* data() noexcept { return arr.data(); }
  const DataT* data() const noexcept { return arr.data(); }
  DataT& operator[](size_t i) noexcept { return arr[i]; }
  const DataT& operator[](size_t i) const noexcept { return arr[i]; }

  auto operator<=>(const AlignedArray& other) const = default;
};

template <typename T, size_t ALIGN>
struct AlignedData {
  explicit AlignedData(size_t num_entries) : data{static_cast<T*>(std::aligned_alloc(ALIGN, num_entries * sizeof(T)))} {
    if (data == nullptr) {
      throw std::runtime_error{"Could not allocate memory. " + std::string{std::strerror(errno)}};
    }
    std::memset(data, 0, num_entries * sizeof(T));
  }

  ~AlignedData() { free(data); }

  // We don't need any of these for the benchmarks.
  AlignedData(const AlignedData&) = delete;
  AlignedData& operator=(const AlignedData&) = delete;
  AlignedData& operator=(AlignedData&&) = delete;

  AlignedData(AlignedData&&) noexcept = default;

  // Docs for assume_aligned: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1007r3.pdf
  [[nodiscard]] T* aligned_data() { return std::assume_aligned<ALIGN>(data); }
  [[nodiscard]] const T* aligned_data() const { return std::assume_aligned<ALIGN>(data); }

 private:
  T* data;
};

// Helpers to allow finding the corresponding uintX_t type from a size at compile time
// clang-format off
template <size_t BYTES> struct UnsignedInt;
// clang reports bit-vectors of less than 8 bits to have sizeof() == 0, see https://github.com/llvm/llvm-project/issues/59788
template <> struct UnsignedInt<0> { using T = uint8_t; };
template <> struct UnsignedInt<1> { using T = uint8_t; };
template <> struct UnsignedInt<2> { using T = uint16_t; };
template <> struct UnsignedInt<4> { using T = uint32_t; };
template <> struct UnsignedInt<8> { using T = uint64_t; };
// clang-format on

#if defined(__aarch64__)
#include <arm_neon.h>
// clang-format off
template <size_t ELEMENT_BYTES> struct NeonVecT;
template <> struct NeonVecT<1> { using T = uint8x16_t; };
template <> struct NeonVecT<2> { using T = uint16x8_t; };
template <> struct NeonVecT<4> { using T = uint32x4_t; };
template <> struct NeonVecT<8> { using T = uint64x2_t; };
// clang-format on
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#endif
