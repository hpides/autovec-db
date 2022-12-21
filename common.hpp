#pragma once

#include <array>
#include <cstdint>
#include <cstddef>

#if (defined(__GNUC__) && !defined(__clang__))
#define GCC_COMPILER 1
#else
#define GCC_COMPILER 0
#endif

#ifdef NDEBUG
#define DEBUG_DO(block) (void) 0;
#else
#define DEBUG_DO(block) do { block } while(0)
#endif


#if(__AVX512VBMI2__ == 1)
#define AVX512_AVAILABLE
#endif

template <typename DataT, size_t NUM_ENTRIES, size_t ALIGN>
struct alignas(ALIGN) AlignedArray {
  // We want to use an empty custom constructor here to avoid zeroing the array when creating an AlignedArray.
  AlignedArray(){}

  std::array<uint64_t, NUM_ENTRIES> data;
};
