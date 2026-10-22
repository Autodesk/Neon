#pragma once

/**
 * CUDA disaggregated-dense grid kernel. Twin of @c detail/backend/cudalike/dense_kernel.h
 * but parameterised on @c dense_disg::Span / @c dense_disg::Idx so the per-cell pitch
 * math goes through the five-region disaggregated layout instead of the aggregated
 * SoA. Include only from @c .cu translation units (or any TU compiled with nvcc).
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense_disg/span.h>

#include <concepts>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon::detail::dense_disg {

/**
 * Callable type for @ref disg_grid_kernel : @c void fn(span, idx) where @c span is a
 * @c dense_disg::Span and @c idx is a @c dense_disg::Idx . Same shape as the aggregated
 * dense grid's @c dense_kernel_fn ; the difference is in the @c Span / @c Idx pair the
 * concept is instantiated against.
 */
template <typename F>
concept disg_kernel_fn = requires(F fn, Span span, Idx idx) {
  {fn(span, idx)}->std::same_as<void>;
};

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * Maps each CUDA thread to @c (ix, iy, iz) = @c blockIdx * blockDim + threadIdx and
 * invokes @p fn(span, idx) only for threads that fall inside @c span.iter_extent(view) .
 * Running @c Internal then @c Boundary (in either order) covers exactly the same set
 * of owned cells as a single @c Standard launch — the three iteration domains
 * partition the slab, same as the aggregated dense grid.
 */
template <typename F>
requires disg_kernel_fn<F> __global__ void
disg_grid_kernel(Span span,
                 ::carbon::DataView view,
                 F fn) {
  const int64_t ix = static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int64_t iy = static_cast<int64_t>(blockIdx.y * blockDim.y + threadIdx.y);
  const int64_t iz = static_cast<int64_t>(blockIdx.z * blockDim.z + threadIdx.z);
  const ValidatedIdx r = span.set_and_validate(view, ix, iy, iz);
  if (!r.valid) {
    return;
  }
  fn(span, r.idx);
}

#endif

}  // namespace carbon::detail::dense_disg
