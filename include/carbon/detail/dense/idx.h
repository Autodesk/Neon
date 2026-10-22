#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/integer3d.h>

#include <stdint.h>

namespace carbon::detail::dense {

/**
 * Local logical indices for a CUDA thread (or cell): @c (ix, iy, iz) from @c threadIdx / @c
 * blockIdx mapping. Global coordinates (including slab z-origin) live on @ref Span because they
 * are shared by every field over the same grid / slab — see @c Span::global_location.
 *
 * @par Why is this a class wrapping @c int64_3d?
 *
 * For the dense grid this looks redundant: an @c int64_3d would already do. The wrapper exists so
 * that the kernel-facing @c Idx type can grow per grid family without changing the
 * @c Partition::read API. Sparse/block/multi-resolution grids (mirroring Neon's @c bGrid /
 * @c mGrid abstractions) need to carry extra fields next to @c (ix, iy, iz) — typically a block
 * id, a refinement level, an in-block bitmask offset, and similar — and they all need to flow
 * into device code through the same @c Idx slot. By making @c Idx an opaque class today (instead
 * of @c int64_3d), kernels written against the dense grid keep compiling unchanged when those
 * other grids land and start specialising the @c Idx type. Treat it as a reserved extension
 * point.
 */
class Idx {
  int64_3d cell_{};

public:
  CARBON_HD constexpr Idx() noexcept = default;

  /** @p ix, @p iy, @p iz are the computed local grid indices for this thread. */
  CARBON_HD constexpr Idx(int64_t ix,
                          int64_t iy,
                          int64_t iz) noexcept
      : cell_{ix,
              iy,
              iz} {}

  CARBON_HD constexpr explicit Idx(int64_3d cell) noexcept
      : cell_(cell) {}

  /** Local extent holding @c (ix, iy, iz). */
  [[nodiscard]] CARBON_HD constexpr auto
  cell() const noexcept -> int64_3d {
    return cell_;
  }
};

}  // namespace carbon::detail::dense
