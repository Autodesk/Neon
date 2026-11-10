#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/process_grid.h>

#include <cstdint>

namespace carbon {

/**
 * Per-axis boundary semantics of a @ref DenseDistributed grid (propagated into every per-rank
 * @c Span).
 *
 * The enumerators are independent bits, so axes can mix open and periodic boundaries:
 * @c PeriodicX | @c PeriodicZ wraps X and Z while leaving Y open.
 *
 * Drives two pieces of behaviour:
 * - @c Partition::neighbour : an out-of-domain read on an open axis returns @c valid = false; on a
 *   periodic axis the read wraps to the opposite face.
 * - Halo exchange (@ref HaloDistributed): schedules the wrap pair on every periodic axis and omits
 *   it on an open axis.
 *
 * @note This mask only expresses periodicity. Per-face Dirichlet / Neumann conditions on open
 * faces are a separate, opt-in type: see @ref BoundaryConditions and
 * @c Partition::neighbour_bc . A condition on a periodic axis is rejected at construction.
 *
 * @note This is deliberately a different type from @ref BoundaryMode, which belongs to the
 * Z-slab-only @ref Dense / @ref DenseDisg families and keeps its original numeric values
 * (@c PeriodicZ == 1). Keeping the two apart means a per-axis mask can never be handed to a grid
 * that only understands a Z wrap.
 */
enum class BoundaryModeDistributed : std::uint8_t {
  /** Non-toroidal grid (default): every axis is open. */
  Open = 0,

  /** Wrap the X axis. */
  PeriodicX = 1U << 0,

  /** Wrap the Y axis. */
  PeriodicY = 1U << 1,

  /** Wrap the Z axis. */
  PeriodicZ = 1U << 2,
};

/** True when @p mode wraps @p axis. */
[[nodiscard]] constexpr CARBON_HD auto
is_periodic(BoundaryModeDistributed mode,
            Axis axis) noexcept -> bool {
  const auto bits = static_cast<std::uint8_t>(mode);
  switch (axis) {
    case Axis::X:
      return (bits & static_cast<std::uint8_t>(BoundaryModeDistributed::PeriodicX)) != 0;
    case Axis::Y:
      return (bits & static_cast<std::uint8_t>(BoundaryModeDistributed::PeriodicY)) != 0;
    case Axis::Z:
      return (bits & static_cast<std::uint8_t>(BoundaryModeDistributed::PeriodicZ)) != 0;
  }
  return false;
}

[[nodiscard]] constexpr CARBON_HD auto
operator|(BoundaryModeDistributed a,
          BoundaryModeDistributed b) noexcept -> BoundaryModeDistributed {
  return static_cast<BoundaryModeDistributed>(static_cast<std::uint8_t>(a) |
                                              static_cast<std::uint8_t>(b));
}

[[nodiscard]] constexpr CARBON_HD auto
operator&(BoundaryModeDistributed a,
          BoundaryModeDistributed b) noexcept -> BoundaryModeDistributed {
  return static_cast<BoundaryModeDistributed>(static_cast<std::uint8_t>(a) &
                                              static_cast<std::uint8_t>(b));
}

}  // namespace carbon
