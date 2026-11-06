#pragma once

/**
 * Shared block-partial accumulation for the CUDA/HIP reduce kernels.
 *
 * The four reduce-kernel headers (Dense, DenseDistributed, DenseDisg, Block) all fold
 * per-block partials into a global scalar under @ref carbon::reduce::ReduceFinalize::BlockAtomic.
 * Keeping the type dispatch here ensures every backend gets the same HIP-aware atomic path
 * and that an unsupported element type is a compile-time error rather than a silent no-op.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/detail/backend/cudalike/atomic.h>

#include <cstdint>
#include <type_traits>

namespace carbon {
namespace detail {

template <typename>
inline constexpr bool dependent_false_v = false;

template <typename T>
CARBON_HD inline auto
reduce_atomic_accumulate(T* dst,
                         T value) -> void {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> ||
                std::is_same_v<T, int32_t> || std::is_same_v<T, int64_t>) {
    (void)atomic_add(dst, value);
  } else {
    static_assert(dependent_false_v<T>,
                  "reduce_atomic_accumulate: unsupported element type");
  }
}

}  // namespace detail
}  // namespace carbon
