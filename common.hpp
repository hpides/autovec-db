#pragma once

#include <array>

template <typename DataT, size_t NUM_ENTRIES, size_t ALIGN>
struct alignas(ALIGN) AlignedArray {
  // We want to use an empty custom constructor here to avoid zeroing the array when creating an AlignedArray.
  AlignedArray(){};

  std::array<uint64_t, NUM_ENTRIES> data;
};
