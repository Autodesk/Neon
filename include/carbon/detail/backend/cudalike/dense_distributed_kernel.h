#pragma once

/**
 * CUDA dense grid kernel. Include only from @c .cu translation units (or any TU compiled with nvcc).
 * Requires C++20 and @c --extended-lambda on the CUDA compile line when using device lambdas.
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense_distributed/span.h>

#include <concepts>
#include <utility>

#if CARBON_CUDALIKE_DEVICE_TU
#include <carbon/detail/backend/cuda_like.h>
#endif

namespace carbon::detail::dense_distributed {

/**
 * Callable type for @ref dense_grid_kernel: @c void fn(span, idx).
 *
 * Host-side checking requires the call to be valid on the host too, so prefer a functor with
 * @c CARBON_HOST_DEVICE @c void @c operator()(Span, @c Idx) @c const (or a non-capturing
 * @c CARBON_HOST_DEVICE lambda at namespace scope with @c --extended-lambda). A @c __device__-only
 * callable can still be used with the kernel but may not satisfy this concept during instantiation.
 */
template <typename F, typename SpanT>
concept dense_kernel_fn_for = requires(F fn, SpanT span, Idx idx) {
  {fn(span, idx)}->std::same_as<void>;
};

template <typename F>
concept dense_kernel_fn = dense_kernel_fn_for<F, Span>;

template <typename F>
concept dense_distributed_kernel_fn =
    dense_kernel_fn_for<F, TypedSpan<StorageKind::Slab>> &&
    dense_kernel_fn_for<F, TypedSpan<StorageKind::Pencil>> &&
    dense_kernel_fn_for<F, TypedSpan<StorageKind::Brick>>;

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * Maps each CUDA thread to @c (ix, iy, iz) = @c blockIdx * blockDim + threadIdx and invokes
 * @p fn(span, idx) for threads inside the box @c span.window_extent(). The cell handed to @p fn is
 * @c window_origin() + (ix, iy, iz), i.e. buffer-relative, so halo layers are never iterated. Use
 * @c span.global_location(idx) if you need the grid-global coordinates.
 *
 * The kernel is @b not templated on @ref DataView. A view is realised on the host as one (Standard,
 * Internal) or several (Boundary) @ref ViewWindow boxes, and the launcher stamps each box into the
 * span with @c Span::with_window before every launch (see @c launch_dense_grid). Running Internal
 * then Boundary (in either order) covers exactly the owned cells of a single Standard launch.
 *
 * @par Why the window lives in the span, not in a separate kernel argument
 * Measured on sm_89 with the D3Q19-BGK fast op (nvcc 13.5): reading origin/extent from the span
 * copy gives 48 regs / ~944 SASS instructions; passing the same six numbers as an extra
 * struct/scalar argument gives 64 regs / ~1128 -- ptxas then no longer folds the origin into the
 * uniform-datapath base offset. Keep it this way unless you re-measure.
 *
 * @note We deliberately do @b not annotate this kernel with @c __launch_bounds__ . The launch
 * block is warp-aligned and capped at 512 threads (@c dense_make_kernel_block), and a measured
 * A100 sweep of LBM D3Q19 (the register-heaviest dense consumer) showed a @c __launch_bounds__(512)
 * cap to be neutral-to-slightly-negative across N=128..384 -- the pull-stencil kernels are HBM-
 * bandwidth bound (already near the roofline), so there is no occupancy headroom for the hint to
 * recover. Left off to avoid constraining ptxas for no gain; revisit only for a compute-bound
 * dense consumer.
 *
 * @tparam Kind Stamped onto a @ref TypedSpan so @c Partition::neighbour can
 *              @c if constexpr -specialise the slab path (pass the span into user lambdas as
 *              @c auto ).
 */
template <StorageKind Kind, typename F>
requires dense_kernel_fn_for<F, TypedSpan<Kind>> __global__ void
dense_grid_kernel(TypedSpan<Kind> span,
                  F fn) {
  const int64_t ix = static_cast<int64_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int64_t iy = static_cast<int64_t>(blockIdx.y * blockDim.y + threadIdx.y);
  const int64_t iz = static_cast<int64_t>(blockIdx.z * blockDim.z + threadIdx.z);
  const TypedSpan<Kind> local{span};
  const int64_3d e = local.window_extent();
  const int64_3d o = local.window_origin();
  if (ix >= e.x || iy >= e.y || iz >= e.z) {
    return;
  }
  fn(local, Idx(o.x + ix, o.y + iy, o.z + iz));
}

#endif

}  // namespace carbon::detail::dense_distributed
