#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/integer3d.h>

#include <stdint.h>

namespace carbon::detail::block {

/**
 * Sentinel value for "no such block" in the 27-way @c Block connectivity table and in
 * neighbour resolution. Mirrors @c bSpan::getInvalidBlockId in Neon (max @c uint32_t ).
 */
inline constexpr uint32_t kInvalidBlockIdx = static_cast<uint32_t>(-1);

/**
 * The @e user-block ("tray") view of a voxel for the multi-resolution (mGrid) addressing.
 * Mirrors Neon's @c MicroIndex from @c bIndex.h .
 *
 * When a memory brick (@c MemX × MemY × MemZ ) is sub-divided into user blocks
 * (@c UserX × UserY × UserZ , so @c BlockRatio = Mem / User user blocks per axis), the
 * same voxel has two equivalent addressings:
 *
 * - the @e data-block view carried by @ref Idx : @c (block_idx, intra) over the memory
 *   brick;
 * - the @e tray view carried here: @c (tray_block_idx, in_tray) , where each memory brick
 *   fans out into @c BlockRatioX·BlockRatioY·BlockRatioZ trays (one per user block) and
 *   @c in_tray is the offset within the user block.
 *
 * The conversion is pure index arithmetic and lives on @c Span (which knows @c SBlock 's
 * ratios) as @c Span::to_micro_index / @c Span::from_micro_index — keeping this struct a
 * thin data carrier, symmetric with @ref Idx . In the single-resolution default
 * (@c User == Mem , every @c BlockRatio == 1 ) the tray view collapses onto the data-block
 * view: @c tray_block_idx == block_idx and @c in_tray == intra .
 */
class MicroIndex {
  int32_t tray_block_idx_{};
  int8_3d in_tray_{};

public:
  CARBON_HD constexpr MicroIndex() noexcept = default;

  CARBON_HD constexpr MicroIndex(int32_t tray_block_idx,
                                 int8_t ix,
                                 int8_t iy,
                                 int8_t iz) noexcept
      : tray_block_idx_(tray_block_idx),
        in_tray_{ix,
                 iy,
                 iz} {}

  /** 1-D index of the containing user-block tray (memory brick fanned out by @c BlockRatio ). */
  [[nodiscard]] CARBON_HD constexpr auto
  tray_block_idx() const noexcept -> int32_t {
    return tray_block_idx_;
  }

  /** Offset @c (x, y, z) within the user block, in @c [0, UserX) × [0, UserY) × [0, UserZ) . */
  [[nodiscard]] CARBON_HD constexpr auto
  in_tray() const noexcept -> int8_3d {
    return in_tray_;
  }
};

/**
 * Local logical index for a thread / voxel on the @c carbon::Block (block-sparse) grid.
 * Mirrors Neon's @c bIndex from @c Neon-disg/libNeonDomain/include/Neon/domain/details/
 * bGrid/bIndex.h .
 *
 * Carries @b two coordinates:
 *
 * - @c block_idx : 1-D index into the per-GPU list of retained blocks (the blocks the
 *   construction-time predicate marked as containing at least one active voxel). All
 *   field reads and bitmask checks pivot off this id.
 * - @c intra : (x, y, z) inside the block, in @c [0, MemX) × [0, MemY) × [0, MemZ) .
 *   Stored as @c int8_3d : @c MemX/Y/Z fit in @c int8_t for any reasonable brick size
 *   (Neon caps the same way), and the smaller footprint matters because every CUDA
 *   thread has one of these in registers.
 *
 * The pitch resolver lives on @c Span::linear_offset (so the field layout stays
 * encapsulated there); @c Idx is just the data the kernel-facing API plumbs through.
 * The multi-resolution @e user-block (tray) view of the same voxel is @ref MicroIndex ,
 * obtained via @c Span::to_micro_index — @c Idx itself stays a pure data-block coordinate
 * so the single-resolution hot path is unchanged.
 */
class Idx {
  uint32_t block_idx_{kInvalidBlockIdx};
  int8_3d intra_{};

public:
  CARBON_HD constexpr Idx() noexcept = default;

  CARBON_HD constexpr Idx(uint32_t block_idx,
                          int8_t ix,
                          int8_t iy,
                          int8_t iz) noexcept
      : block_idx_(block_idx),
        intra_{ix,
               iy,
               iz} {}

  /** 1-D id of the containing block on this device's retained-block list. */
  [[nodiscard]] CARBON_HD constexpr auto
  block_idx() const noexcept -> uint32_t {
    return block_idx_;
  }

  /** Intra-block (x, y, z) in the brick's local frame. */
  [[nodiscard]] CARBON_HD constexpr auto
  intra() const noexcept -> int8_3d {
    return intra_;
  }

  /** True iff this @c Idx points at a real block (i.e. has not been sentinelled). */
  [[nodiscard]] CARBON_HD constexpr auto
  valid_block() const noexcept -> bool {
    return block_idx_ != kInvalidBlockIdx;
  }
};

}  // namespace carbon::detail::block
