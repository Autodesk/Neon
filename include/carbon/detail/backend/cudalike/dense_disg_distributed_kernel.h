#pragma once

/**
 * CUDA kernel for the @c DenseDisgDistributed grid. Twin of
 * @c detail/backend/cudalike/dense_disg_kernel.h parameterised on
 * @c dense_disg_distributed::Span / @c Idx . Include only from @c .cu translation units (or any
 * TU compiled with nvcc).
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense_disg_distributed/span.h>

#include <concepts>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon::detail::dense_disg_distributed {

/**
 * Callable type for @ref disg_grid_kernel : @c void fn(span, idx) where @c span is a
 * @c dense_disg_distributed::Span and @c idx is a @c dense_disg_distributed::Idx .
 */
template <typename F>
concept disg_kernel_fn = requires(F fn, Span span, Idx idx) {
  {fn(span, idx)}->std::same_as<void>;
};

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * Maps each CUDA thread to @c (ix, iy, iz) = @c blockIdx * blockDim + threadIdx and invokes
 * @p fn(span, idx) only for threads inside @c span.iter_extent(view) . @c Internal + @c Boundary
 * cover the same owned cells as one @c Standard launch.
 *
 * @tparam Kind  Stamped onto a @ref TypedSpan so @c Partition hot paths can
 *               @c if constexpr -specialize (pass the span into user lambdas as @c auto ).
 */
template <typename F, StorageKind Kind>
requires disg_kernel_fn<F> __global__ void
disg_grid_kernel(Span span,
                 ::carbon::DataView view,
                 F fn) {
  const int64_t ix = static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int64_t iy = static_cast<int64_t>(blockIdx.y * blockDim.y + threadIdx.y);
  const int64_t iz = static_cast<int64_t>(blockIdx.z * blockDim.z + threadIdx.z);
  // Copy out of parameter space: when fn's neighbour reads take the span by reference, nvcc 12.6
  // otherwise addresses the by-value parameter through the local window, which faults on R535.
  const TypedSpan<Kind> local{span};
  const ValidatedIdx r = local.set_and_validate(view, ix, iy, iz);
  if (!r.valid) {
    return;
  }
  fn(local, r.idx);
}

#endif

}  // namespace carbon::detail::dense_disg_distributed
