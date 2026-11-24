#pragma once

/**
 * @file layout.h
 * @brief Pure host-side decomposition arithmetic for @ref DenseDisgDistributed.
 *
 * Three storage shapes, selected by the process-grid dims (not by @ref DecompositionMode , so a
 * @c Brick grid that factorizes to @c Px == 1 uses pencil or slab storage):
 *
 * - @b Slab ( @c Px == Py == 1 , @c y not periodic): the global @c z extent is split over the
 *   @c Pz coordinates with the same splitting rule as @ref DenseDisg and @ref DenseDistributed
 *   (the first @c n % P coordinates receive one extra layer). Storage is the five-region slab map.
 * - @b Pencil ( @c Px == 1 , @c y split or periodic): @c y and @c z are split independently with
 *   the same rule. Storage reserves one ghost row on each @c y side in addition to the @c z ghost
 *   planes and uses the nested pencil map (see @c dense_disg_distributed::Span ).
 * - @b Brick ( @c Px > 1 ): all three axes are split with the same rule. Storage reserves one
 *   ghost layer on every side of every axis and uses the nested brick map.
 *
 * The per-chunk storage always reserves one ghost plane on each @c z side regardless of the
 * exchanged halo thickness, which is what makes the disaggregated map well-formed for every owned
 * cell.
 *
 * Everything here is a plain function of (global extent, @ref ProcessGrid, boundary mode,
 * stencil) so unit tests can check the geometry without a GPU or MPI.
 */

#include <carbon/boundary_mode.h>
#include <carbon/boundary_mode_distributed.h>
#include <carbon/detail/dense_distributed/layout.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/process_grid.h>
#include <carbon/stencil.h>

#include <cstdint>
#include <stdexcept>

namespace carbon::detail::dense_disg_distributed {

/** Ghost planes reserved on each @c z side of every chunk buffer (pinned by the disg map). */
inline constexpr int64_t kGhostRadius = 1;

/**
 * Exchanged halo thickness for @p stencil : @c 0 for an empty / radius-0 stencil, @c 1
 * otherwise. Applies to every exchanged axis ( @c z always, @c y on pencil and brick layouts,
 * @c x on brick layouts).
 *
 * @throws std::invalid_argument when @c stencil.radius() > 1 (the disaggregated layout pins both
 *         ghost and boundary radius to 1).
 */
[[nodiscard]] inline auto
halo_z_from_stencil(const Stencil& stencil) -> int64_t {
  const int32_t radius = stencil.radius();
  if (radius != 0 && radius != 1) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDisgDistributed: stencil radius must be 0 or 1 (the disaggregated layout pins "
        "ghost / boundary radius to 1); got radius={}",
        radius);
  }
  return static_cast<int64_t>(radius);
}

/**
 * Validate @p mode and translate its @c z axis to the Z-only @ref BoundaryMode consumed by the
 * slab halo helpers. @c PeriodicY is accepted (it turns on the pencil storage, see
 * @ref ghost_y_for ).
 *
 * @throws std::invalid_argument when @p mode wraps @c x (no layout exchanges a periodic x halo).
 */
[[nodiscard]] inline auto
to_boundary_mode(BoundaryModeDistributed mode) -> BoundaryMode {
  if (is_periodic(mode, Axis::X)) {
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDisgDistributed: BoundaryModeDistributed::PeriodicX is not supported (no "
        "disaggregated layout exchanges a periodic x halo; got mode bits={})",
        static_cast<int>(mode));
  }
  return is_periodic(mode, Axis::Z) ? BoundaryMode::PeriodicZ : BoundaryMode::Open;
}

/** Ghost columns reserved on each @c x side: @c 1 when @p pg splits @c x (brick storage). */
[[nodiscard]] inline auto
ghost_x_for(const ProcessGrid& pg) noexcept -> int64_t {
  return pg.dims().x > 1 ? 1 : 0;
}

/**
 * Ghost rows reserved on each @c y side: @c 1 when @p pg splits @c x or @c y or @p mode wraps
 * @c y (pencil / brick storage), @c 0 otherwise (slab storage, bit-identical to @ref DenseDisg ).
 */
[[nodiscard]] inline auto
ghost_y_for(const ProcessGrid& pg,
            BoundaryModeDistributed mode) noexcept -> int64_t {
  return (ghost_x_for(pg) != 0 || pg.dims().y > 1 || is_periodic(mode, Axis::Y)) ? 1 : 0;
}

/**
 * Chunk owned by process coordinate @p coords of @p pg over @p global .
 *
 * @c owned follows @c dense_distributed::rank_chunk . @c haloed and @c offset include
 * @ref kGhostRadius planes on @c z , @ref ghost_y_for rows on @c y and @ref ghost_x_for columns
 * on @c x (the storage extent and buffer origin), while @c halo carries the exchanged thickness
 * @c {ghost_x * halo, ghost_y * halo, halo} . @c xy_pitch is @c haloed.x * haloed.y .
 */
[[nodiscard]] inline auto
disg_chunk(const int64_3d& global,
           const ProcessGrid& pg,
           int32_3d coords,
           int64_t halo,
           BoundaryModeDistributed mode) -> dense_distributed::RankChunk {
  (void)to_boundary_mode(mode);
  const int64_t ghost_x = ghost_x_for(pg);
  const int64_t ghost_y = ghost_y_for(pg, mode);
  dense_distributed::RankChunk chunk =
      dense_distributed::rank_chunk(global, pg, coords, int64_3d{ghost_x, ghost_y, kGhostRadius});
  chunk.halo = int64_3d{ghost_x * halo, ghost_y * halo, halo};
  return chunk;
}

/**
 * Slab-only spelling of @ref disg_chunk kept for the Phase-1 call sites.
 *
 * @throws std::invalid_argument unless @p pg is a slab ( @c Px == Py == 1 ).
 */
[[nodiscard]] inline auto
slab_chunk(const int64_3d& global,
           const ProcessGrid& pg,
           int32_3d coords,
           int64_t halo_z) -> dense_distributed::RankChunk {
  if (!pg.is_slab() || pg.mode() == DecompositionMode::Pencil ||
      pg.mode() == DecompositionMode::Brick) {
    const int32_3d dims = pg.dims();
    carbon::log::throw_with_log<std::invalid_argument>(
        "DenseDisgDistributed::slab_chunk: process grid is not a slab (mode={}, dims=({}, {}, {}))",
        static_cast<int>(pg.mode()), dims.x, dims.y, dims.z);
  }
  return disg_chunk(global, pg, coords, halo_z, BoundaryModeDistributed::Open);
}

}  // namespace carbon::detail::dense_disg_distributed
