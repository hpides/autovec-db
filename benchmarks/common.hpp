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

  AlignedArray() = default;

  auto begin() noexcept { return array_.begin(); }
  auto end() noexcept { return array_.end(); }

  [[nodiscard]] DataT* data() noexcept { return array_.data(); }
  [[nodiscard]] const DataT* data() const noexcept { return array_.data(); }
  [[nodiscard]] DataT& operator[](size_t index) noexcept { return array_[index]; }
  [[nodiscard]] const DataT& operator[](size_t index) const noexcept { return array_[index]; }

  [[nodiscard]] size_t size() const noexcept { return array_.size(); }

  auto operator<=>(const AlignedArray& other) const = default;

 private:
  std::array<DataT, NUM_ENTRIES> array_;
};

template <typename T, size_t ALIGN>
struct AlignedData {
  explicit AlignedData(size_t num_entries)
      : data_{static_cast<T*>(std::aligned_alloc(ALIGN, num_entries * sizeof(T)))} {
    if (data_ == nullptr) {
      // NOLINTNEXTLINE(concurrency-mt-unsafe): Our benchmarks run single-threaded
      throw std::runtime_error{"Could not allocate memory. " + std::string{std::strerror(errno)}};
    }
    std::memset(data_, 0, num_entries * sizeof(T));
  }

  ~AlignedData() { free(data_); }  // NOLINT(cppcoreguidelines-no-malloc): This _is_ the RAII class

  // We don't need any of these for the benchmarks.
  AlignedData(const AlignedData&) = delete;
  AlignedData& operator=(const AlignedData&) = delete;
  AlignedData& operator=(AlignedData&&) = delete;

  AlignedData(AlignedData&&) noexcept = default;

  // Docs for assume_aligned: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1007r3.pdf
  [[nodiscard]] T* aligned_data() { return std::assume_aligned<ALIGN>(data_); }
  [[nodiscard]] const T* aligned_data() const { return std::assume_aligned<ALIGN>(data_); }

 private:
  T* data_;
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
