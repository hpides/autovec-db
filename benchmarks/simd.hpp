#include "common.hpp"

#if defined(__aarch64__)
#include <arm_neon.h>
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

namespace detail {

#if defined(__aarch64__)

// All 64 Bit vectors have the same flow, but require the instruction without `q` in it.
template <typename VectorT>
concept CanShuffleVecNeon = sizeof(VectorT) == 16;

template <CanShuffleVecNeon VectorT>
inline VectorT neon_shuffle_vector(VectorT vec, VectorT mask) {
  constexpr uint8_t ELEMENT_SIZE = sizeof(decltype(vec[0]));

  if constexpr (ELEMENT_SIZE == 1) {
    // Case: uint8x16_t. This can be represented with one instruction.
    return vqtbl1q_u8(vec, mask);
  } else if constexpr (ELEMENT_SIZE == 2) {
    // Case: uint16x8_t
    // Takes every even-numbered element from the input vectors. As we pass the same vector twice, we pick byte
    // 0/2/4/... from `mask` twice, duplicate it in `converted_mask`.
    uint8x16_t converted_mask = vtrn1q_u8(mask, mask);
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
    uint8x16_t converted_mask = vreinterpretq_u8_u32(mask * 0x01010101);

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

    // Alternative 2 (like above, too lazy to type here):

    // Alternative 3:
    uint8x16_t converted_mask = vreinterpretq_u8_u64(mask * 0x0101010101010101);

    // Finalize is the same for all alternatives.
    converted_mask *= ELEMENT_SIZE;
    converted_mask += uint8x16_t{0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    return vqtbl1q_u8(vec, converted_mask);
  }
}
#endif

template <typename VectorT, typename MaskT>
inline VectorT builtin_shuffle_vector(VectorT vec, MaskT mask) {
#if GCC_COMPILER
  // The vector element size must be equal, so we convert the mask to VecT. This is a no-op if they are the same.
  // Note that this conversion can have a high runtime cost, so consider using the correct type.
  // See: https://godbolt.org/z/fdvGsWqPa
  return __builtin_shuffle(vec, __builtin_convertvector(mask, VectorT));
#else
  return __builtin_shufflevector(vec, mask);
#endif
}

// TODO: maybe this is more efficient with a magic multiply (https://zeux.io/2022/09/02/vpexpandb-neon-z3/)
template <typename MaskT, typename VectorT>
MaskT gcc_comparison_to_bitmask(VectorT matches) {
  using ElementT = decltype(VectorT{}[0]);
  constexpr size_t NUM_ELEMENTS = sizeof(VectorT) / sizeof(ElementT);

  static_assert(sizeof(MaskT) * 8 >= NUM_ELEMENTS, "Number of elements greater than size of mask");

  // TODO: I want to do this, but I'm not quite sure how to handle the constexpr lambda return value.

  //  constexpr auto get_and_mask = [] {
  //    if constexpr (NUM_ELEMENTS == 1) {
  //      return VectorT{1};
  //    } else if (NUM_ELEMENTS == 2) {
  //      return VectorT{1, 2};
  //    } else if (NUM_ELEMENTS == 4) {
  //      return VectorT{1, 2, 4, 8};
  //    } else if (NUM_ELEMENTS == 8) {
  //      using XX = GccVec<ElementT, sizeof(ElementT) * NUM_ELEMENTS>::T;
  //      return XX{1, 2, 4, 8, 16, 32, 64, 128};
  //    } else if (NUM_ELEMENTS == 16 && sizeof(ElementT) >= 2) {
  //      using XX = GccVec<ElementT, sizeof(ElementT) * NUM_ELEMENTS>::T;
  //      return XX{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
  //    } else {
  //      static_assert(always_false<VectorT>, "GCC is not your friend. Use clang :)");
  //    }
  //  };
  //
  //  constexpr VectorT and_mask = get_and_mask();
  //  auto single_bits = matches & and_mask;
  //
  //  alignas(alignof(VectorT)) std::array<ElementT, NUM_ELEMENTS> single_bit_array{};
  //  std::memcpy(&single_bit_array, &single_bits, sizeof(single_bit_array));
  //  return std::accumulate(single_bit_array.begin(), single_bit_array.end(), 0u);

  MaskT mask = 0;
  for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
    mask |= matches[i] ? static_cast<MaskT>(1) << i : 0;
  }
  return mask;
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
template <typename VectorT, typename MaskT>
inline VectorT shuffle_vector(VectorT vec, MaskT mask) {
#if defined(__aarch64__)
  if constexpr (detail::CanShuffleVecNeon<VectorT>) {
    // This specialized method gives a ~60% speedup compared to clang's builtin in the dict scan (on M1 MacBook Pro).
    return detail::neon_shuffle_vector(vec, mask);
  } else {
    return detail::builtin_shuffle_vector(vec, mask);
  }
#else
  return detail::builtin_shuffle_vector(vec, mask);
#endif
}

template <typename VectorT, typename ElementT = decltype(std::declval<VectorT>()[0])>
inline VectorT broadcast(ElementT value) {
  return value - VectorT{};
}

template <typename VectorT>
inline VectorT load(const void* ptr) {
  return *reinterpret_cast<const VectorT*>(ptr);
}

template <typename VectorT>
inline void store(void* ptr, VectorT value) {
  *reinterpret_cast<VectorT*>(ptr) = value;
}

template <typename VectorT>
inline void unaligned_store(void* ptr, VectorT value) {
  using UnalignedVector __attribute__((aligned(1))) = VectorT;
  *reinterpret_cast<UnalignedVector*>(ptr) = value;
}

template <typename VectorT, size_t NUM_MASK_BITS = sizeof(VectorT) / sizeof(VectorT{}[0]),
          typename MaskT = typename UnsignedInt<NUM_MASK_BITS / 8>::T>
inline MaskT comparison_to_bitmask(VectorT vec) {
#if CLANG_COMPILER
  using MaskVecT __attribute__((ext_vector_type(NUM_MASK_BITS))) = bool;
  MaskVecT mask = __builtin_convertvector(vec, MaskVecT);
  return reinterpret_cast<MaskT&>(mask);
#else
  return detail::gcc_comparison_to_bitmask<MaskT>(vec);
#endif
}

}  // namespace simd
