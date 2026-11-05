#pragma once

#include <carbon/export.h>
#include <carbon/integer3d.h>

#include <cstdint>

namespace carbon {

/**
 * Axis index used by per-axis process-grid and boundary queries.
 */
enum class Axis : std::uint8_t {
  X = 0,
  Y = 1,
  Z = 2,
};

/**
 * How the global domain is split across ranks.
 *
 * @c Slab reproduces carbon's historical Z-only layout: @c (Px,Py,Pz)=(1,1,P).
 * @c Pencil and @c Brick expose Y and X as additional split axes.
 */
enum class DecompositionMode : std::uint8_t {
  Slab = 0,
  Pencil = 1,
  Brick = 2,
  Custom = 3,
};

/**
 * Rank <-> process-coordinate mapping for a 3D decomposition.
 *
 * Coordinates are stored as @c (px,py,pz) and ranks are assigned in row-major order with
 * Z fastest:
 * @code
 * rank = px * (Py * Pz) + py * Pz + pz
 * @endcode
 * Consequently @c Slab (Px=Py=1) has @c rank == pz, matching carbon's existing Z-only
 * partitioning.
 *
 * @par Auto-factorization
 * @ref make splits @p world_size across the axes that are free for the requested mode. Prime
 * factors are assigned largest-first to the currently smallest free axis, considering Z before Y
 * before X. This keeps the grid balanced while keeping Z as the fastest-varying axis. Call
 * @ref custom when an exact grid is required.
 *
 * @par Host-only
 * This type is a host-side layout description; it is not copied into device memory. It performs
 * bounds checks and throws rather than returning sentinels.
 */
class CARBON_API ProcessGrid {
public:
  /** Default: a one-rank slab, @c (1,1,1). */
  ProcessGrid() noexcept = default;

  /**
   * Auto-factorize @p world_size into a grid matching @p mode.
   *
   * @throws std::invalid_argument if @p world_size <= 0, or if @p mode is @c Custom (use
   *         @ref custom instead).
   */
  [[nodiscard]] static auto
  make(int32_t world_size,
       DecompositionMode mode) -> ProcessGrid;

  /**
   * Build an exact @c (Px,Py,Pz) grid.
   *
   * @throws std::invalid_argument if @p world_size <= 0, any dimension is <= 0, or
   *         @c px*py*pz != world_size.
   */
  [[nodiscard]] static auto
  custom(int32_t world_size,
         int32_t px,
         int32_t py,
         int32_t pz) -> ProcessGrid;

  /** Process-grid dimensions as @c (Px,Py,Pz). */
  [[nodiscard]] auto
  dims() const noexcept -> int32_3d {
    return dims_;
  }

  /** Decomposition mode supplied at construction (@c Custom for @ref custom). */
  [[nodiscard]] auto
  mode() const noexcept -> DecompositionMode {
    return mode_;
  }

  /** Total number of ranks, @c Px*Py*Pz. */
  [[nodiscard]] auto
  world_size() const noexcept -> int32_t {
    return static_cast<int32_t>(dims_.volume());
  }

  /** True when @c Px == Py == 1, i.e. the historical Z-slab layout. */
  [[nodiscard]] auto
  is_slab() const noexcept -> bool {
    return dims_.x == 1 && dims_.y == 1;
  }

  /**
   * Map a global rank to process coordinates.
   *
   * @throws std::out_of_range if @p rank is outside @c [0,world_size()).
   */
  [[nodiscard]] auto
  coords_of(int32_t rank) const -> int32_3d;

  /**
   * Map process coordinates back to the global rank.
   *
   * @throws std::out_of_range if any coordinate is outside its axis' grid extent.
   */
  [[nodiscard]] auto
  rank_of(int32_3d coords) const -> int32_t;

private:
  ProcessGrid(int32_3d dims,
              DecompositionMode mode) noexcept
      : dims_(dims),
        mode_(mode) {}

  int32_3d dims_{1, 1, 1};
  DecompositionMode mode_{DecompositionMode::Slab};
};

}  // namespace carbon