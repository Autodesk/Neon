#pragma once

#include <carbon/integer3d.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace carbon {

namespace detail {
class Gpu;
}

/**
 * @brief Derive a 3D thread block for @p target_threads total threads.
 *
 * The dense buffer is x-contiguous (unit stride on x, row stride on y, slab
 * stride on z — see @c Span::linear_offset), and @c dense_grid_kernel maps
 * @c threadIdx.x onto the x cell coordinate. So the *only* block shape that
 * coalesces global loads is one whose x extent is a whole number of warps:
 * a 32-lane warp must sweep 32 consecutive x cells to land on a single
 * 128-byte cache line. A near-cubic block (e.g. 4x4x32) instead scatters each
 * warp across ~32 rows/slabs, collapsing achieved HBM bandwidth to a fraction
 * of peak — the dominant bottleneck for the LBM-style pull-stencil kernels.
 *
 * We therefore give x a full warp (or a small multiple, capped at 64 to keep
 * some threads for y), spend the remainder on y so the y±1 neighbour rows stay
 * warm in L2, and keep z == 1 (the z±1 planes are served by L2 across slabs).
 * x stays a multiple of 32 because @p target_threads is warp-aligned first.
 */
[[nodiscard]] inline auto
dense_factor_block_threads(int target_threads) noexcept -> LaunchShape {
  if (target_threads <= 0) {
    return LaunchShape{1U, 1U, 1U};
  }
  int t = std::min(target_threads, 512);
  t = (t / 32) * 32;
  if (t < 32) {
    t = 32;
  }

  const std::uint32_t bx = std::min<std::uint32_t>(static_cast<std::uint32_t>(t), 64U);
  const std::uint32_t by = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(t) / bx);
  const std::uint32_t bz = 1U;
  return LaunchShape{bx, by, bz};
}

/**
 * @brief Per-device launch block for dense-family compute kernels.
 * Cached by @c gpu.device_id(); sized from @c max_threads_per_block and
 * @c max_threads_per_multiprocessor (warp-aligned, capped at 512 threads).
 */
[[nodiscard]] auto
dense_make_kernel_block(const detail::Gpu& gpu) noexcept -> LaunchShape;

namespace detail {

/**
 * @brief Validates @p block and writes @p grid so threads cover @p ex for @c dense_grid_kernel.
 * Uses ceiling division per axis; @p block is fixed input — only @p grid is derived from @p ex.
 * @throws std::runtime_error after @c CARBON_LOG_WARN if the block violates per-axis or
 *         @p max_threads_per_block limits.
 */
inline auto
compute_dense_grid(const int64_3d& ex,
                   LaunchShape block,
                   int max_threads_per_block,
                   LaunchShape& grid) -> void {
  constexpr std::uint32_t kMaxBlockDim = 1024U;
  const int64_t nx = ex.x;
  const int64_t ny = ex.y;
  const int64_t nz = ex.z;

  auto ceil_div = [](int64_t n, std::uint32_t d) -> std::uint32_t {
    if (n <= 0) {
      return 0U;
    }
    const int64_t dd = static_cast<int64_t>(d);
    return static_cast<std::uint32_t>((n + dd - 1) / dd);
  };

  if (nx <= 0 || ny <= 0 || nz <= 0) {
    grid = LaunchShape{0U, 0U, 0U};
    return;
  }

  const std::uint32_t bx = block.x;
  const std::uint32_t by = block.y;
  const std::uint32_t bz = block.z;
  if (bx == 0U || by == 0U || bz == 0U) {
    CARBON_LOG_WARN(
        "compute_dense_grid: block dimension is zero (block=({}, {}, {})); refusing launch shape",
        bx, by, bz);
    throw std::runtime_error("compute_dense_grid: block dimension is zero");
  }
  if (bx > kMaxBlockDim || by > kMaxBlockDim || bz > kMaxBlockDim) {
    CARBON_LOG_WARN(
        "compute_dense_grid: block dimension exceeds {} (block=({}, {}, {})); refusing launch "
        "shape",
        kMaxBlockDim, bx, by, bz);
    throw std::runtime_error("compute_dense_grid: block dimension exceeds per-axis limit");
  }
  const auto threads = static_cast<std::uint64_t>(bx) * static_cast<std::uint64_t>(by) *
                       static_cast<std::uint64_t>(bz);
  if (threads > static_cast<std::uint64_t>(max_threads_per_block)) {
    CARBON_LOG_WARN(
        "compute_dense_grid: block threads {} (block=({}, {}, {})) exceeds max_threads_per_block "
        "{}; "
        "refusing launch shape",
        threads, bx, by, bz, max_threads_per_block);
    throw std::runtime_error(
        "compute_dense_grid: block thread count exceeds device max_threads_per_block");
  }

  grid = LaunchShape{ceil_div(nx, bx), ceil_div(ny, by), ceil_div(nz, bz)};
}

}  // namespace detail

}  // namespace carbon
