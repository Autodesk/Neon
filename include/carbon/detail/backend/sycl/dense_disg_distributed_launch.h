#pragma once

/**
 * SYCL twin of @c detail/backend/cudalike/dense_disg_distributed_launch.h . Same nd_range shape
 * as @c sycl/dense_disg_launch.h , over @c dense_disg_distributed::Span / @c Idx .
 */

#include <carbon/data_view.h>
#include <carbon/dense_disg_distributed.h>
#include <carbon/detail/dense_disg_distributed/span.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace carbon::detail::dense_disg_distributed {

template <typename F>
concept disg_kernel_fn = requires(F fn, Span span, Idx idx) {
  {fn(span, idx)}->std::same_as<void>;
};

template <typename F>
requires disg_kernel_fn<F> inline auto
launch_disg_grid(const ::carbon::DenseDisgDistributed& grid,
                 LaunchShape grid_shape,
                 LaunchShape block_shape,
                 size_t relative_gpu_id,
                 size_t shared_mem,
                 sycl::queue& queue,
                 ::carbon::DataView view,
                 F fn) -> void {
  if (relative_gpu_id >= grid.gpu_count()) {
    ::carbon::log::throw_with_log<std::out_of_range>(
        "launch_disg_grid: relative_gpu_id out of range (relative_gpu_id={}, gpu_count={})",
        relative_gpu_id, grid.gpu_count());
  }
  if (shared_mem != 0U) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_disg_grid: dynamic shared memory ({} bytes) is not yet supported under "
        "the SYCL backend.",
        shared_mem);
  }
  if (!grid_shape.is_nonzero() || !block_shape.is_nonzero()) {
    return;
  }

  const Span span = grid.spans()[relative_gpu_id];

  const sycl::range<3> local{static_cast<size_t>(block_shape.z), static_cast<size_t>(block_shape.y),
                             static_cast<size_t>(block_shape.x)};
  const sycl::range<3> global{
      static_cast<size_t>(grid_shape.z) * static_cast<size_t>(block_shape.z),
      static_cast<size_t>(grid_shape.y) * static_cast<size_t>(block_shape.y),
      static_cast<size_t>(grid_shape.x) * static_cast<size_t>(block_shape.x)};

  try {
    queue.parallel_for(sycl::nd_range<3>{global, local}, [=](sycl::nd_item<3> it) {
      const int64_t ix = static_cast<int64_t>(it.get_global_id(2));
      const int64_t iy = static_cast<int64_t>(it.get_global_id(1));
      const int64_t iz = static_cast<int64_t>(it.get_global_id(0));
      const ValidatedIdx r = span.set_and_validate(view, ix, iy, iz);
      if (!r.valid) {
        return;
      }
      fn(span, r.idx);
    });
  } catch (const sycl::exception& ex) {
    ::carbon::log::throw_with_log<std::runtime_error>(
        "launch_disg_grid: SYCL parallel_for submission failed (relative_gpu_id={}): {}",
        relative_gpu_id, ex.what());
  }
}

}  // namespace carbon::detail::dense_disg_distributed
