#pragma once

/**
 * CUDA dense grid kernel. Include only from @c .cu translation units (or any TU compiled with nvcc).
 * Requires C++20 and @c --extended-lambda on the CUDA compile line when using device lambdas.
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense/span.h>

#include <concepts>
#include <utility>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon::detail::dense {

/**
 * Callable type for @ref dense_grid_kernel: @c void fn(span, idx).
 *
 * Host-side checking requires the call to be valid on the host too, so prefer a functor with
 * @c CARBON_HOST_DEVICE @c void @c operator()(Span, @c Idx) @c const (or a non-capturing
 * @c CARBON_HOST_DEVICE lambda at namespace scope with @c --extended-lambda). A @c __device__-only
 * callable can still be used with the kernel but may not satisfy this concept during instantiation.
 */
template <typename F>
concept dense_kernel_fn = requires(F fn, Span span, Idx idx) {
  {fn(span, idx)}->std::same_as<void>;
};

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * Maps each CUDA thread to @c (ix, iy, iz) = @c blockIdx * blockDim + threadIdx and invokes
 * @p fn(span, idx) only for threads that fall inside @c span.iter_extent(view). The launch
 * grid is the iteration shape for @p view (Standard / Internal / Boundary; see
 * @c Span::iter_extent), so halo layers are never iterated.
 *
 * The @p idx handed to @p fn is buffer-relative: x and y are passed through verbatim, while
 * z has been shifted by @c Span::set_and_validate so that it points at the correct cell in
 * the buffer-local SoA layout for @p view. Use @c span.global_location(idx) if you need the
 * grid-global coordinates.
 *
 * Running Internal then Boundary (in either order) covers exactly the same set of owned
 * cells as a single Standard launch — the three iteration domains partition the slab.
 *
 * @note We deliberately do @b not annotate this kernel with @c __launch_bounds__ . The launch
 * block is warp-aligned and capped at 512 threads (@c dense_make_kernel_block), and a measured
 * A100 sweep of LBM D3Q19 (the register-heaviest dense consumer) showed a @c __launch_bounds__(512)
 * cap to be neutral-to-slightly-negative across N=128..384 — the pull-stencil kernels are HBM-
 * bandwidth bound (already near the roofline), so there is no occupancy headroom for the hint to
 * recover. Left off to avoid constraining ptxas for no gain; revisit only for a compute-bound
 * dense consumer.
 */
template <typename F>
requires dense_kernel_fn<F> __global__ void
dense_grid_kernel(Span span,
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

}  // namespace carbon::detail::dense
