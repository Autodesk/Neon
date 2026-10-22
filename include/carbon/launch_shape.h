#pragma once

/**
 * Backend-neutral 3D launch shape used by the public @c Dense API to expose per-view
 * grid / block extents without dragging @c <cuda_runtime.h> into installed headers.
 *
 * @par Why a dedicated type
 * Carbon's compute-kernel hierarchy is moving toward dual-backend support (CUDA + SYCL).
 * The previous @c std::vector<dim3> on @c Dense::gpu_grid / @c Dense::gpu_block forced
 * every consumer of @c carbon/dense.h to transitively pull in CUDA headers, which would
 * break a SYCL-only build. @c LaunchShape is a tiny POD that:
 *
 *  - matches @c dim3's per-axis layout (unsigned 32-bit @c x, @c y, @c z);
 *  - converts trivially to @c dim3 (CUDA backend) and to @c sycl::range<3> (SYCL backend)
 *    inside @c carbon::detail (where the backend headers live);
 *  - has no implicit numeric promotions: arithmetic on @c LaunchShape is intentional and
 *    explicit, so compute kernels keep using their existing per-axis variables.
 *
 * The struct is value-typed, trivially copyable, default-constructible to a zero shape,
 * and supports equality comparison. It deliberately has no operator overloads beyond
 * comparison; any per-axis math (ceil-div, clamp, ...) lives in the helpers that compute
 * @c LaunchShape values from iteration extents.
 */

#include <cstdint>

namespace carbon {

/**
 * @brief 3D launch extent in unsigned threads / blocks per axis. POD; trivially copyable.
 *
 * Used by @c Dense::gpu_grid / @c Dense::gpu_block to expose the per-view, per-GPU launch
 * shape carbon will use under the active compute backend. Construction is an aggregate
 * initialisation (@c LaunchShape{8U, 8U, 8U}) or via the explicit ctor (@c LaunchShape(8, 8,
 * 8)). The "logical extent in threads" reading matches @c dim3's documented layout, so
 * conversions are 1:1 for the CUDA backend.
 */
struct LaunchShape {
  std::uint32_t x{0};
  std::uint32_t y{0};
  std::uint32_t z{0};

  constexpr LaunchShape() noexcept = default;

  constexpr LaunchShape(std::uint32_t x_,
                        std::uint32_t y_,
                        std::uint32_t z_) noexcept
      : x(x_),
        y(y_),
        z(z_) {}

  /** Total threads / blocks across all three axes. */
  [[nodiscard]] constexpr auto
  volume() const noexcept -> std::uint64_t {
    return static_cast<std::uint64_t>(x) * static_cast<std::uint64_t>(y) *
           static_cast<std::uint64_t>(z);
  }

  /** True iff every axis is non-zero (i.e. the launch covers at least one thread). */
  [[nodiscard]] constexpr auto
  is_nonzero() const noexcept -> bool {
    return x != 0U && y != 0U && z != 0U;
  }

  friend constexpr auto
  operator==(const LaunchShape& a,
             const LaunchShape& b) noexcept -> bool {
    return a.x == b.x && a.y == b.y && a.z == b.z;
  }

  friend constexpr auto
  operator!=(const LaunchShape& a,
             const LaunchShape& b) noexcept -> bool {
    return !(a == b);
  }
};

}  // namespace carbon
