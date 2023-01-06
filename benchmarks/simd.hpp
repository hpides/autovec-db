#include "common.hpp"

namespace simd {

template <typename ElementT, size_t VECTOR_SIZE_IN_BYTES, size_t ALIGNMENT = VECTOR_SIZE_IN_BYTES>
struct GccVec {
  using T __attribute__((vector_size(VECTOR_SIZE_IN_BYTES), aligned(ALIGNMENT))) = ElementT;
  using UnalignedT __attribute__((vector_size(VECTOR_SIZE_IN_BYTES), aligned(1))) = ElementT;
};

#if CLANG_COMPILER
template <size_t NUM_BITS>
struct ClangBitmask {
  using T __attribute__((ext_vector_type(NUM_BITS))) = bool;
};
#endif

/**
 * Clang and GCC have slightly different calls for builtin shuffles, so we need to distinguish between theme here.
 * GCC has __builtin_shuffle and __builtin_shufflevector (the latter only since GCC12), clang only has
 * __builtin_shufflevector. However, __builtin_shufflevector is called differently on both. GCC requires two input
 * vectors whereas clang requires only one. While it is possible to call it with two vectors in clang, this in turn
 * requires the mask indexes to be "constant integers", i.e., compile-time values. The documentation on these functions
 * is quite messy/non-existent.
 * Clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-shufflevector
 *   --> This says that we always need two vectors and constant integer indexes.
 *   But if you look at the actual implementation in clang:
 *     https://github.com/llvm/llvm-project/blob/ef992b60798b6cd2c50b25351bfc392e319896b7/clang/lib/CodeGen/CGExprScalar.cpp#L1645-L1678
 *   you see that the second argument (vec2 in the documentation) can actually be a runtime mask.
 */
template <typename VectorT, typename MaskT>
inline VectorT shuffle_vector(VectorT vec, MaskT mask) {
#if GCC_COMPILER
  // The vector element size must be equal, so we convert the mask to VecT. This is a no-op if they are the same.
  // Note that this conversion can have a high runtime cost, so consider using the correct type.
  // See: https://godbolt.org/z/fdvGsWqPa
  return __builtin_shuffle(vec, __builtin_convertvector(mask, VectorT));
#else
  return __builtin_shufflevector(vec, mask);
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

}  // namespace simd
