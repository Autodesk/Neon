#pragma once

#include <carbon/cuda_annotation.h>

#include <stdint.h>

namespace carbon::detail::block {

/**
 * Compile-time block-size policy for the @c carbon::Block (block-sparse) grid family.
 *
 * Mirrors Neon's @c Neon::domain::details::bGrid::StaticBlock at
 * @c Neon/libNeonDomain/include/Neon/domain/details/bGrid/StaticBlock.h .
 *
 * - @c MemX / @c MemY / @c MemZ : memory block extent in voxels (the brick size). Each
 *   retained block stores a contiguous @c MemX · MemY · MemZ · Components element slab
 *   per cardinality, addressed via the standard x-fastest pitch
 *   @c x + MemX · y + MemX · MemY · z .
 *
 * - @c UserX / @c UserY / @c UserZ : the @e user block extent — the logical sub-division a
 *   memory brick is partitioned into. For the single-resolution grid these default to the
 *   memory extent (@c BlockRatio == 1), so the brick is one user block and every voxel is a
 *   user cell. For a multi-resolution (mGrid) port one user block becomes one parent cell
 *   whose @c BlockRatioX · BlockRatioY · BlockRatioZ children live on the next finer level
 *   (Neon's 2×2×2 octree uses @c StaticBlock&lt;8,8,8, 2,2,2&gt; per level). The ratio
 *   @c MemX / UserX is the per-axis branching factor.
 *
 * - @c IsMultiResMode : tag selecting the multi-resolution interpretation of the user-block
 *   sub-division. Default @c false keeps the single-resolution semantics bit-identical (the
 *   ratios are all 1 in that case anyway); a future mGrid instantiates with @c true so the
 *   per-level construction and @c Idx micro-index addressing key off it.
 *
 * - @c BitMask : packed activity mask, one bit per voxel (in the same x-fastest pitch),
 *   stored as a contiguous run of @c uint32_t words. Tracks which voxels in a retained
 *   block actually pass the user's "is active" predicate; @c Span::set_and_validate
 *   tests this bit before invoking the user kernel.
 *
 * The block sizes are non-type template parameters so the bitmask width, pitch math, and
 * thread-block dimensions all fold to compile-time constants in device code (matching
 * Neon's design — same trade: instantiation cost vs hot-path simplicity).
 *
 * @par Default brick (<tt>StaticBlock&lt;8, 8, 8&gt;</tt>)
 *
 * 8×8×8 = 512 voxels = 16 @c uint32_t bitmask words, @c UserX/Y/Z == @c MemX/Y/Z so all
 * @c BlockRatio == 1 and @c IsMultiResMode == false. This is what Neon ships as the only
 * explicit instantiation in @c bGrid.cpp and what carbon's @c Block uses by default.
 * Other sizes work but instantiate from headers (no @c extern @c template stub yet).
 */
template <int MemX_,
          int MemY_,
          int MemZ_,
          int UserX_ = MemX_,
          int UserY_ = MemY_,
          int UserZ_ = MemZ_,
          bool IsMultiResMode_ = false>
struct StaticBlock {
  static constexpr int MemX = MemX_;
  static constexpr int MemY = MemY_;
  static constexpr int MemZ = MemZ_;
  static constexpr int VoxelCount = MemX * MemY * MemZ;
  static constexpr int PitchX = 1;
  static constexpr int PitchY = MemX;
  static constexpr int PitchZ = MemX * MemY;

  static constexpr int UserX = UserX_;
  static constexpr int UserY = UserY_;
  static constexpr int UserZ = UserZ_;
  static constexpr int BlockRatioX = MemX / UserX;
  static constexpr int BlockRatioY = MemY / UserY;
  static constexpr int BlockRatioZ = MemZ / UserZ;
  static constexpr bool IsMultiResMode = IsMultiResMode_;

  static_assert(MemX >= UserX && MemY >= UserY && MemZ >= UserZ,
                "StaticBlock: user block extent must not exceed the memory brick extent");
  static_assert(MemX % UserX == 0 && MemY % UserY == 0 && MemZ % UserZ == 0,
                "StaticBlock: memory brick extent must be a whole multiple of the user block "
                "extent (the per-axis refinement ratio must be integral)");

  using BitMaskWord = uint32_t;
  static constexpr int BitsPerWord = 32;
  static constexpr int WordCount = (VoxelCount + BitsPerWord - 1) / BitsPerWord;

  /**
   * Voxel-membership bitmask for one block. Stores @c VoxelCount bits in @c WordCount
   * @c uint32_t words, indexed by the same x-fastest pitch the field uses. Designed to
   * sit on the device alongside the per-block field storage and be cheap to test from a
   * thread (one shift + one bitwise-and per @c set_and_validate ).
   *
   * Layout decision: bit @c (x + MemX · y + MemX · MemY · z) lives at word
   * @c (pitch / 32) , bit position @c (pitch % 32) . Same convention Neon uses; required
   * so a host-side fill via @c set_active produces a mask the device-side @c is_active
   * reads bit-for-bit.
   */
  struct BitMask {
    BitMaskWord words[WordCount];

    [[nodiscard]] CARBON_HD constexpr static auto
    word_and_bit(int x,
                 int y,
                 int z) noexcept -> uint32_t {
      const uint32_t pitch = static_cast<uint32_t>(x + MemX * y + MemX * MemY * z);
      return pitch;
    }

    /** Set the (x, y, z) voxel's bit to 1. Host-only because field init runs on the host. */
    auto
    set_active(int x,
               int y,
               int z) noexcept -> void {
      const uint32_t pitch = word_and_bit(x, y, z);
      words[pitch / BitsPerWord] |= (BitMaskWord{1} << (pitch % BitsPerWord));
    }

    /** Test the (x, y, z) voxel. Available on host @b and device. */
    [[nodiscard]] CARBON_HD constexpr auto
    is_active(int x,
              int y,
              int z) const noexcept -> bool {
      const uint32_t pitch = word_and_bit(x, y, z);
      return (words[pitch / BitsPerWord] & (BitMaskWord{1} << (pitch % BitsPerWord))) != 0U;
    }

    /** Zero the bitmask. */
    auto
    clear() noexcept -> void {
      for (int i = 0; i < WordCount; ++i) {
        words[i] = 0U;
      }
    }
  };
};

/** Default brick size: 8×8×8 = 512 voxels / 16 mask words. Matches Neon's default. */
using DefaultStaticBlock = StaticBlock<8, 8, 8>;

}  // namespace carbon::detail::block
