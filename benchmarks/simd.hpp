#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>

#include "common.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace simd {

template <typename ElementT_, size_t VECTOR_SIZE_IN_BYTES, size_t ALIGNMENT = VECTOR_SIZE_IN_BYTES>
struct GccVec {
  using ElementT = ElementT_;
  using T __attribute__((vector_size(VECTOR_SIZE_IN_BYTES), aligned(ALIGNMENT))) = ElementT;
  using UnalignedT __attribute__((vector_size(VECTOR_SIZE_IN_BYTES), aligned(1))) = ElementT;
};

#if CLANG_COMPILER
template <size_t NUM_BITS>
struct ClangBitmask {
  using T __attribute__((ext_vector_type(NUM_BITS))) = bool;
};
#endif
#if defined(__aarch64__)
// clang-format off
template <size_t ELEMENT_BYTES> struct NeonVecT;
template <> struct NeonVecT<1> { using T = uint8x16_t; };
template <> struct NeonVecT<2> { using T = uint16x8_t; };
template <> struct NeonVecT<4> { using T = uint32x4_t; };
template <> struct NeonVecT<8> { using T = uint64x2_t; };
// clang-format on
#endif

// Produces e.g. {1, 2, 4, 8} for (uint64x4) or {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128} for uint8x16
template <typename VecT>
VecT positional_single_bit_mask() {
  // We'd want this to be constexpr, but we can't, because the vector types can't be used in a constexpr way
  // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=101651
  VecT result{};
  using ElementT = std::decay_t<decltype(result[0])>;
  static_assert(std::is_unsigned_v<ElementT>);

  const size_t vector_elements = sizeof(VecT) / sizeof(ElementT);

  ElementT current_value = 1;
  for (size_t i = 0; i < vector_elements; ++i) {
    result[i] = current_value;
    if (current_value <= std::numeric_limits<ElementT>::max() / 2) {
      current_value *= 2;
    } else {
      current_value = 1;
    }
  }

  return result;
}

/*
 * NAMESPACE DETAIL
 */
namespace detail {

#if defined(__aarch64__)

template <typename VectorT, typename MaskT>
inline VectorT neon_shuffle_vector(VectorT vec, MaskT mask) {
  static_assert(sizeof(VectorT) == 16, "Can only do NEON shuffle on 16-Byte NEON vectors.");
  constexpr uint8_t ELEMENT_SIZE = sizeof(decltype(vec[0]));

  auto vec_mask = reinterpret_cast<uint8x16_t>(__builtin_convertvector(mask, VectorT));

  // All 64 Bit vectors have the same flow, but require the instruction without `q` in it.
  if constexpr (ELEMENT_SIZE == 1) {
    // Case: uint8x16_t. This can be represented with one instruction.
    return vqtbl1q_u8(vec, vec_mask);
  } else if constexpr (ELEMENT_SIZE == 2) {
    // Case: uint16x8_t
    // Takes every even-numbered element from the input vectors. As we pass the same vector twice, we pick byte
    // 0/2/4/... from `mask` twice, duplicate it in `converted_mask`.
    uint8x16_t converted_mask = vtrn1q_u8(mask, vec_mask);
    converted_mask *= ELEMENT_SIZE;
    converted_mask += uint8x16_t{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    return vqtbl1q_u8(vec, converted_mask);
  } else if constexpr (ELEMENT_SIZE == 4) {
    // Case: uint32x4_t

    // Alternative 1:
    //     uint8x16_t converted_mask = {
    //         static_cast<uint8_t>(mask[0]), static_cast<uint8_t>(mask[0]), static_cast<uint8_t>(mask[0]),
    //         static_cast<uint8_t>(mask[0]), static_cast<uint8_t>(mask[1]), static_cast<uint8_t>(mask[1]),
    //         static_cast<uint8_t>(mask[1]), static_cast<uint8_t>(mask[1]), static_cast<uint8_t>(mask[2]),
    //         static_cast<uint8_t>(mask[2]), static_cast<uint8_t>(mask[2]), static_cast<uint8_t>(mask[2]),
    //         static_cast<uint8_t>(mask[3]), static_cast<uint8_t>(mask[3]), static_cast<uint8_t>(mask[3]),
    //         static_cast<uint8_t>(mask[3])};

    // Alternative 2: (this is better than #1)
    //     uint32x4_t converted_mask_u32 = mask << 8;
    //     converted_mask_u32 |= mask;
    //     converted_mask_u32 <<= 8;
    //     converted_mask_u32 |= mask;
    //     converted_mask_u32 <<= 8;
    //     converted_mask_u32 |= mask;
    // uint8x16_t converted_mask = vreinterpretq_u8_u32(converted_mask_u32);

    // Alternative 3: (this is better than #2 and #1)
    uint8x16_t converted_mask = vreinterpretq_u8_u32(vreinterpretq_u32_u8(vec_mask) * 0x01010101);

    // Finalize is the same for all alternatives.
    converted_mask *= ELEMENT_SIZE;
    converted_mask += uint8x16_t{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    return vqtbl1q_u8(vec, converted_mask);
  } else if constexpr (ELEMENT_SIZE == 8) {
    // Case: uint64x2_t

    // Alternative 1:
    // uint8x16_t converted_mask = {
    //     static_cast<uint8_t>(mask[0]), static_cast<uint8_t>(mask[0]),
    //     static_cast<uint8_t>(mask[0]), static_cast<uint8_t>(mask[0]),
    //     static_cast<uint8_t>(mask[0]), static_cast<uint8_t>(mask[0]),
    //     static_cast<uint8_t>(mask[0]), static_cast<uint8_t>(mask[0]),
    //     static_cast<uint8_t>(mask[1]), static_cast<uint8_t>(mask[1]),
    //     static_cast<uint8_t>(mask[1]), static_cast<uint8_t>(mask[1]),
    //     static_cast<uint8_t>(mask[1]), static_cast<uint8_t>(mask[1]),
    //     static_cast<uint8_t>(mask[1]), static_cast<uint8_t>(mask[1])};

    // Alternative 2 (like above, too lazy to type here)

    // Alternative 3:
    uint8x16_t converted_mask = vreinterpretq_u8_u64(vreinterpretq_u64_u8(vec_mask) * 0x0101010101010101);

    // Finalize is the same for all alternatives.
    converted_mask *= ELEMENT_SIZE;
    converted_mask += uint8x16_t{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    return vqtbl1q_u8(vec, converted_mask);
  }
}
#endif

template <typename MaskT, typename VectorT>
MaskT gcc_comparison_to_bitmask(VectorT bytemask) {
  // Works properly for few large elements (https://godbolt.org/z/Eb4KGqvh8)
  // but gets worse with more small elements (https://godbolt.org/z/b1GhK7nz7)
  using ElementT = std::decay_t<decltype(bytemask[0])>;
  constexpr size_t NUM_ELEMENTS = sizeof(VectorT) / sizeof(ElementT);
  static_assert(sizeof(MaskT) * 8 >= NUM_ELEMENTS, "Number of elements greater than size of mask");

  const auto element_masks = simd::positional_single_bit_mask<VectorT>();
  const auto single_bits_set = bytemask & element_masks;

  const auto* element_ptr = reinterpret_cast<const ElementT*>(&single_bits_set);
  constexpr size_t ELEMENTS_COMBINED_PER_ITERATION = sizeof(ElementT) * 8;

  MaskT result = 0;
  for (size_t start_element = 0; start_element < NUM_ELEMENTS; start_element += ELEMENTS_COMBINED_PER_ITERATION) {
    const auto sub_bits_begin = element_ptr + start_element;
    const auto sub_bits_end = std::min(sub_bits_begin + ELEMENTS_COMBINED_PER_ITERATION, element_ptr + NUM_ELEMENTS);
    // NOLINTNEXTLINE(bugprone-fold-init-type): We might intentionally add up few big types into one small type here
    const MaskT sub_accumulation_result = std::accumulate(sub_bits_begin, sub_bits_end, MaskT{0});
    result |= sub_accumulation_result << start_element;
  }
  return result;
}

}  // namespace detail

/**
 * Shuffling vectors still seems to be a key pain point when using GCC vector intrinsics. There is different behavior in
 * GCC and Clang, and the generated ARM assembly is very bad.
 *
 * ARM:
 * ---
 *  Unfortunately, builtin shuffling generates very bad assembly, so we need to use a custom implementation here. We
 *  could shuffle with one instruction, but we get O(n) code for n vector elements.
 *  See: https://godbolt.org/z/38Mxvv91q.
 *  By default, we want to show all code without platform-specific variants. So we use the "bad" version here for ARM.
 *  If you need the fast option, explicitly use `detail::neon_shuffle_vector()` directly.
 *
 * GCC vs. Clang:
 * --------------
 *  Clang and GCC have slightly different calls for builtin shuffles, so we need to distinguish between theme here.
 *  GCC has __builtin_shuffle and __builtin_shufflevector (the latter only since GCC12), clang only has
 *  __builtin_shufflevector. However, __builtin_shufflevector is called differently on both. GCC requires two input
 *  vectors whereas clang requires only one. While it is possible to call it with two vectors in clang, this in turn
 *  requires the mask indexes to be "constant integers", i.e., compile-time values. The documentation on these functions
 *  is quite messy/non-existent. See: https://github.com/llvm/llvm-project/issues/59678
 */
// This specialized method gives a ~60% speedup compared to clang's builtin in the dict scan (on M1 MacBook Pro).
template <typename VectorT, typename MaskT>
inline VectorT shuffle_vector(VectorT vec, MaskT mask) {
#if GCC_COMPILER
  // The vector element size must be equal, so we convert the mask to VecT. This is a no-op if they are the same.
  // Note that this conversion can have a runtime cost, so consider using the correct type.
  // See: https://godbolt.org/z/3xdahP67o
  return __builtin_shuffle(vec, __builtin_convertvector(mask, VectorT));
#else
  return __builtin_shufflevector(vec, mask);
#endif
}

template <typename VectorT>
inline VectorT load(const void* ptr) {
  return *reinterpret_cast<const VectorT*>(ptr);
}

template <typename VectorT>
inline VectorT load_unaligned(const void* ptr) {
  using UnalignedVectorT __attribute__((aligned(1))) = VectorT;
  return *reinterpret_cast<const UnalignedVectorT*>(ptr);
}

template <typename VectorT>
inline void store(void* ptr, VectorT value) {
  *reinterpret_cast<VectorT*>(ptr) = value;
}

template <typename VectorT>
inline void store_unaligned(void* ptr, VectorT value) {
  using UnalignedVector __attribute__((aligned(1))) = VectorT;
  *reinterpret_cast<UnalignedVector*>(ptr) = value;
}

template <typename VectorT, size_t NUM_MASK_BITS = sizeof(VectorT) / sizeof(VectorT{}[0]),
          typename MaskT = typename UnsignedInt<NUM_MASK_BITS / 8>::T>
inline MaskT comparison_to_bitmask(VectorT vec) {
#if CLANG_COMPILER
  using MaskVecT __attribute__((ext_vector_type(NUM_MASK_BITS))) = bool;
  MaskVecT mask_vector = __builtin_convertvector(vec, MaskVecT);
  MaskT mask = reinterpret_cast<MaskT&>(mask_vector);
  if constexpr (NUM_MASK_BITS != 8 * sizeof(MaskT)) {
    mask &= (MaskT{1} << NUM_MASK_BITS) - MaskT{1};
  }
  return mask;
#else
  return detail::gcc_comparison_to_bitmask<MaskT>(vec);
#endif
}

}  // namespace simd
