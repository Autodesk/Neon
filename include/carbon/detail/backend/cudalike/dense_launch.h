#pragma once

/**
 * Single-GPU launch helper for @c carbon::detail::dense::dense_grid_kernel. Lives in
 * @c detail/ so that the public @c carbon/dense.h does not have to mention any backend
 * stream / error type. Today this is the CUDA implementation; the SYCL backend will
 * provide an equivalent free function with the same name and signature inside its own
 * detail subtree, selected at compile time by the future backend dispatcher.
 *
 * Internal API: only @c carbon::detail::DeviceKernelImpl reaches in here. User code should
 * compose a @c carbon::Kernel via @c Dense::make_kernel / @c Kernel::compute and call
 * @c Kernel::launch — never @c launch_dense_grid directly.
 *
 * @par Why a free function (rather than a @c Dense member)
 * The previous @c Dense::launch_dense_grid friend method tied the public @c Dense type to
 * @c dim3 and @c gpu_stream_t. With the kernel impls already gating their bodies on
 * @c CARBON_CUDALIKE_DEVICE_TU, the helper does not need member access to @c Dense's privates — it can
 * read @c grid.system() and @c grid.spans() through the public accessors.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/dense.h>
#include <carbon/detail/dense/kernel.h>
#include <carbon/detail/gpu.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>

#include <carbon/detail/backend/cuda_like.h>

#include <cstddef>
#include <stdexcept>

namespace carbon {

namespace detail::dense {

/** Convert a backend-neutral @ref LaunchShape to CUDA's @c dim3 at the launch site. */
[[nodiscard]] inline auto
to_dim3(LaunchShape s) noexcept -> dim3 {
  return dim3(s.x, s.y, s.z);
}

// Dependent false predicate so the non-CUDA branch's static_assert below only fires when
// launch_dense_grid is actually instantiated by user code, not on plain header inclusion.
template <typename>
inline constexpr bool dense_launch_requires_cudacc_ = false;

#if CARBON_CUDALIKE_DEVICE_TU

/**
 * @brief Launch @c dense_grid_kernel on @p relative_gpu_id with the given launch shape.
 *
 * @param grid              The @c Dense layout this launch is bound to.
 * @param grid_shape        Per-axis grid extent (in blocks); produced by
 *                          @ref compute_dense_grid in @c Dense's constructor.
 * @param block_shape       Per-axis block extent (in threads); the same value is used for
 *                          every device today (see @ref dense_make_kernel_block).
 * @param relative_gpu_id   Index into @c grid.system().gpu_group().devices() and @c grid.spans().
 * @param shared_mem        Dynamic shared memory in bytes per launch.
 * @param stream            CUDA stream to enqueue on; the caller has already picked the
 *                          per-GPU handle out of the public @c Stream group.
 * @param view              Iteration sub-domain (Standard / Internal / Boundary).
 * @param fn                Per-cell device callable; must satisfy @c dense_kernel_fn.
 *
 * @return @c gpu_error_t result of @c gpu_runtime_launch_kernel — non-success values are surfaced
 *         to the user through @c DeviceKernelImpl's error path.
 */
template <typename F>
requires dense_kernel_fn<F> [[nodiscard]] inline auto
launch_dense_grid(const ::carbon::Dense& grid,
                  LaunchShape grid_shape,
                  LaunchShape block_shape,
                  size_t relative_gpu_id,
                  size_t shared_mem,
                  gpu_stream_t stream,
                  ::carbon::DataView view,
                  F fn) -> gpu_error_t {
  if (relative_gpu_id >= grid.gpu_count()) {
    ::carbon::log::throw_with_log<std::out_of_range>(
        "launch_dense_grid: relative_gpu_id out of range (relative_gpu_id={}, gpu_count={})",
        relative_gpu_id, grid.gpu_count());
  }
  if (!grid_shape.is_nonzero() || !block_shape.is_nonzero()) {
    // Empty iteration — no work, so report success without touching the driver. This is
    // reachable two ways: a Boundary view on a slab whose z extent is fully owned, and a slab
    // that got no cells at all because the grid has fewer z planes than GPUs. Note that CUDA
    // does *not* forgive a zero grid dimension: cudaLaunchKernel rejects dim3(0, 0, 0) with
    // cudaErrorInvalidConfiguration, so without this guard those launches fail outright.
    return gpu_success;
  }
  const ::carbon::detail::Gpu& gpu = grid.system().gpu_group().devices()[relative_gpu_id];
  const Span span = grid.spans()[relative_gpu_id];
  return gpu.template typed_launch<&dense_grid_kernel<F>>(to_dim3(grid_shape), to_dim3(block_shape),
                                                          shared_mem, stream, span, view, fn);
}

#else

template <typename F>
requires dense_kernel_fn<F> [[nodiscard]] inline auto
launch_dense_grid(const ::carbon::Dense&,
                  LaunchShape,
                  LaunchShape,
                  size_t,
                  size_t,
                  gpu_stream_t,
                  ::carbon::DataView,
                  F) -> gpu_error_t {
  static_assert(
      dense_launch_requires_cudacc_<F>,
      "carbon::detail::dense::launch_dense_grid is only supported in translation "
      "units compiled with nvcc (CARBON_CUDALIKE_DEVICE_TU). Move the call site to a .cu file (or "
      "build the TU with nvcc) and include <carbon/kernel.h> from there. Only the "
      "*call site* needs to be CUDA — the rest of your project can stay on the host "
      "compiler.");
  return gpu_error_not_supported;
}

#endif

}  // namespace detail::dense

}  // namespace carbon
