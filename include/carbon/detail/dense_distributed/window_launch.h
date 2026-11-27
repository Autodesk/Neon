#pragma once

/**
 * Backend-neutral helpers for turning @ref ViewWindow boxes into launch shapes and stage-1
 * reduce unit counts. Keeping the arithmetic in one place ensures the host-side scratch sizing
 * and the actual per-window launches agree.
 */

#include <carbon/detail/dense_distributed/span.h>
#include <carbon/integer3d.h>
#include <carbon/launch_shape.h>

#include <cstdint>

namespace carbon::detail::dense_distributed {

/** Blocks needed to cover @p extent with @p block, per axis (0 on an empty axis). */
[[nodiscard]] inline auto
window_grid_shape(const int64_3d& extent, ::carbon::LaunchShape block) noexcept
    -> ::carbon::LaunchShape {
  auto ceil_div = [](int64_t n, std::uint32_t d) -> std::uint32_t {
    return n <= 0 ? 0U : static_cast<std::uint32_t>((n + static_cast<int64_t>(d) - 1) / d);
  };
  return ::carbon::LaunchShape{ceil_div(extent.x, block.x), ceil_div(extent.y, block.y),
                               ceil_div(extent.z, block.z)};
}

/** Stage-1 blocks for the flat 1-D reduce kernel over @p w. */
[[nodiscard]] inline auto
window_flat_units(const ViewWindow& w, int64_t block_threads) noexcept -> int64_t {
  const int64_t cells = w.volume();
  if (cells <= 0 || block_threads <= 0) {
    return 0;
  }
  return (cells + block_threads - 1) / block_threads;
}

/** Stage-1 blocks for the 2-D `(x, y*z)` reduce kernel over @p w. */
[[nodiscard]] inline auto
window_2d_units(const ViewWindow& w,
                int64_t block_dim_x,
                int64_t block_dim_y) noexcept -> int64_t {
  if (w.empty() || block_dim_x <= 0 || block_dim_y <= 0) {
    return 0;
  }
  const int64_t grid_x = (w.extent.x + block_dim_x - 1) / block_dim_x;
  const int64_t yz_extent = w.extent.y * w.extent.z;
  const int64_t grid_y = (yz_extent + block_dim_y - 1) / block_dim_y;
  return grid_x * grid_y;
}

}  // namespace carbon::detail::dense_distributed
