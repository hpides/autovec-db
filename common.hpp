#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#if (defined(__GNUC__) && !defined(__clang__))
#define GCC_COMPILER 1
#else
#define GCC_COMPILER 0
#endif

#ifdef NDEBUG
#define DEBUG_DO(block) (void)0
#else
#define DEBUG_DO(block) \
  do {                  \
    block               \
  } while (0)
#endif

// We assume that _if_ a server has AVX512, it has everything we need.
// To check this, we use the AVX512 Foundation.
#if (defined(__AVX512F__))
#define AVX512_AVAILABLE
#endif

template <typename DataT, size_t NUM_ENTRIES, size_t ALIGN>
struct alignas(ALIGN) AlignedArray {
  // We want to use an empty custom constructor here to avoid zeroing the array when creating an AlignedArray.
  AlignedArray() {}

  std::array<DataT, NUM_ENTRIES> arr;
  DataT* data() noexcept { return arr.data(); }
  const DataT* data() const noexcept { return arr.data(); }
  DataT& operator[](size_t i) noexcept { return arr[i]; }
  const DataT& operator[](size_t i) const noexcept { return arr[i]; }

  auto operator<=>(const AlignedArray<DataT, NUM_ENTRIES, ALIGN>& other) const = default;
};
