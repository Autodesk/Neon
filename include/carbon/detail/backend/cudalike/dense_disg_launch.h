#pragma once

/**
 * Single-GPU launch helper for @c carbon::detail::dense_disg::disg_grid_kernel . Twin
 * of @c detail/backend/cudalike/dense_launch.h but for the disaggregated-dense grid.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/dense_disg.h>
#include <carbon/detail/backend/cudalike/dense_disg_kernel.h>
#include <carbon/detail/gpu.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>

#include <carbon/detail/backend/cuda_like.h>

#include <cstddef>
#include <stdexcept>

namespace carbon::detail::dense_disg {

[[nodiscard]] inline auto
to_dim3(LaunchShape s) noexcept -> dim3 {
  return dim3(s.x, s.y, s.z);
}

template <typename>
inline constexpr bool disg_launch_requires_cudacc_ = false;

#if CARBON_CUDALIKE_DEVICE_TU

template <typename F>
requires disg_kernel_fn<F> [[nodiscard]] inline auto
launch_disg_grid(const ::carbon::DenseDisg& grid,
                 LaunchShape grid_shape,
                 LaunchShape block_shape,
                 size_t relative_gpu_id,
                 size_t shared_mem,
                 gpu_stream_t stream,
                 ::carbon::DataView view,
                 F fn) -> gpu_error_t {
  if (relative_gpu_id >= grid.gpu_count()) {
    ::carbon::log::throw_with_log<std::out_of_range>(
        "launch_disg_grid: relative_gpu_id out of range (relative_gpu_id={}, gpu_count={})",
        relative_gpu_id, grid.gpu_count());
  }
  if (!grid_shape.is_nonzero() || !block_shape.is_nonzero()) {
    // Empty iteration — see the matching guard in dense_launch.h for why CUDA needs it.
    return gpu_success;
  }
  const ::carbon::detail::Gpu& gpu = grid.system().gpu_group().devices()[relative_gpu_id];
  const Span span = grid.spans()[relative_gpu_id];
  return gpu.template typed_launch<&disg_grid_kernel<F>>(to_dim3(grid_shape), to_dim3(block_shape),
                                                         shared_mem, stream, span, view, fn);
}

#else

template <typename F>
requires disg_kernel_fn<F> [[nodiscard]] inline auto
launch_disg_grid(const ::carbon::DenseDisg&,
                 LaunchShape,
                 LaunchShape,
                 size_t,
                 size_t,
                 gpu_stream_t,
                 ::carbon::DataView,
                 F) -> gpu_error_t {
  static_assert(
      disg_launch_requires_cudacc_<F>,
      "carbon::detail::dense_disg::launch_disg_grid is only supported in translation "
      "units compiled with nvcc (CARBON_CUDALIKE_DEVICE_TU). Move the call site to a .cu file (or "
      "build the TU with nvcc) and include <carbon/kernel.h> from there.");
  return gpu_error_not_supported;
}

#endif

}  // namespace carbon::detail::dense_disg
