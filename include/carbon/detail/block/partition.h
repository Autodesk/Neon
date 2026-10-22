#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/block/static_block.h>
#include <carbon/integer3d.h>
#include <carbon/neighbour_value.h>

#include <stddef.h>
#include <stdint.h>

namespace carbon::detail::block {

/**
 * Deprecated alias for @ref ::carbon::NeighbourValue retained for source
 * compatibility with the original @c bPartition::neighbour return type. New
 * call sites — and every grid family — should use @c ::carbon::NeighbourValue
 * directly so generic code can share one neighbour-result type across Dense /
 * DenseDisg / Block without branching on grid. Same shape (@c value + @c valid )
 * and the canonical name now also exposes @c value_or(alternative) for
 * one-line boundary defaults.
 */
template <typename T>
using NghData = ::carbon::NeighbourValue<T>;

/**
 * Non-owning device-visible view of one cardinality plane of a @c Block::Field<T,
 * Components, SBlock> . Mirrors Neon's @c bPartition with one deliberate cleanup: the
 * per-block bitmask and the 27-way block-to-block connectivity table do @b not live on
 * @c Partition . They live on @c Span and are read through it on every neighbour walk.
 *
 * Why: bitmask + connectivity are properties of the @c (Block, GPU) topology, not of
 * the field's values, so they should have a single owner — @c Span — rather than being
 * duplicated into every per-component, per-Field, per-GPU @c Partition built over the
 * same @c Block . This also makes @c block::Partition symmetric with
 * @c carbon::detail::dense::Partition (both are thin "view of values" wrappers) and
 * collapses the lifetime contract to "the @c Block must outlive any @c Span that
 * references it" — @c Field carries no @c Block -borrowed pointers.
 *
 * Read-side API:
 *
 * - @c read(span, idx, c) : direct field read at a known-active voxel. Same contract as
 *   the @c carbon::detail::dense::Partition family — caller must already have validated
 *   @c idx via @c span.set_and_validate .
 * - @c write : the writable counterpart.
 * - @c neighbour(span, idx, offset, c) : returns
 *   @c ::carbon::NeighbourValue<T> for the voxel at @c idx.intra + offset ,
 *   walking across to the neighbour block via @c span.connectivity() when
 *   @c offset takes the target outside the source brick. Validity tracks both
 *   "neighbour block exists" and "the target voxel's bit is set in that
 *   block's mask".
 */
template <typename T, int Components, typename SBlock>
class Partition {
public:
  using SpanT = Span<SBlock>;
  using BitMask = typename SBlock::BitMask;
  static constexpr int MemX = SBlock::MemX;
  static constexpr int MemY = SBlock::MemY;
  static constexpr int MemZ = SBlock::MemZ;

  /**
   * Connectivity flat-offset for an integer @c (dx, dy, dz) in @c {-1, 0, 1}^3 .
   * Forwarder to @c Span::connectivity_slot — kept on @c Partition as well so existing
   * call sites that already type @c Partition<T, C, SBlock>::connectivity_slot keep
   * working without source churn.
   */
  [[nodiscard]] CARBON_HD constexpr static auto
  connectivity_slot(int dx,
                    int dy,
                    int dz) noexcept -> uint32_t {
    return SpanT::connectivity_slot(dx, dy, dz);
  }

  /** Slots per block (always 27; exposed so @c Block can size the connectivity buffer). */
  static constexpr uint32_t kSlotsPerBlock = SpanT::kSlotsPerBlock;

  CARBON_HD constexpr Partition() noexcept = default;

  CARBON_HD constexpr explicit Partition(T* ptr) noexcept
      : ptr_(ptr) {}

  [[nodiscard]] CARBON_HD constexpr auto
  data() const noexcept -> T* {
    return ptr_;
  }

  /**
   * Raw device pointer — alias for @ref data , kept for API parity with
   * @c dense::Partition::ptr() and @c dense_disg::Partition::ptr() so generic
   * code can use the same accessor across grid families.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  ptr() const noexcept -> T* {
    return ptr_;
  }

  /**
   * Read the @p component_id -th component of voxel @p idx . @p span must be the same
   * @c Span the @p idx was minted from (so the pitch math agrees on @c Components ).
   */
  [[nodiscard]] CARBON_HD constexpr auto
  read(const SpanT& span,
       const Idx& idx,
       int component_id) const noexcept -> const T& {
    return ptr_[span.linear_offset(idx, Components, component_id)];
  }

  /**
   * Write @p value into the @p component_id -th component of voxel @p idx . Same
   * shape as @c carbon::detail::dense::Partition::write : the partition is
   * conceptually unchanged by the call, only the underlying storage is mutated, so
   * the method is @c const and works with the @c [=] device-lambda captures the
   * @c KernelLoader hands out.
   */
  CARBON_HD constexpr auto
  write(const SpanT& span,
        const Idx& idx,
        int component_id,
        T value) const noexcept -> void {
    ptr_[span.linear_offset(idx, Components, component_id)] = value;
  }

  /**
   * Resolve the voxel at @c idx + @p offset and return its value + validity. Walks the
   * 27-way connectivity table (read from @p span ) when @p offset takes the target
   * outside the source brick; caps @p offset components to @c [-MemX, +MemX] etc. --
   * larger offsets than one brick away are not supported in this pass (matches Neon's
   * bGrid: stencil radius <= brick size).
   *
   * @note Carbon's @c carbon::Stencil already constrains stencil offsets to fit
   *       @c int8_t (radius <= 127), and the brick sizes used in practice
   *       (8 / 16) make the "walk one block" rule the dominant case; multi-brick
   *       jumps remain a follow-up tied to wide-radius stencils.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  neighbour(const SpanT& span,
            const Idx& idx,
            int8_3d offset,
            int component_id) const noexcept -> ::carbon::NeighbourValue<T> {
    ::carbon::NeighbourValue<T> out{};
    if (!idx.valid_block()) {
      return out;
    }

    const auto intra = idx.intra();
    int tx = static_cast<int>(intra.x) + static_cast<int>(offset.x);
    int ty = static_cast<int>(intra.y) + static_cast<int>(offset.y);
    int tz = static_cast<int>(intra.z) + static_cast<int>(offset.z);

    int dx = 0;
    int dy = 0;
    int dz = 0;
    if (tx < 0) {
      dx = -1;
      tx += MemX;
    } else if (tx >= MemX) {
      dx = +1;
      tx -= MemX;
    }
    if (ty < 0) {
      dy = -1;
      ty += MemY;
    } else if (ty >= MemY) {
      dy = +1;
      ty -= MemY;
    }
    if (tz < 0) {
      dz = -1;
      tz += MemZ;
    } else if (tz >= MemZ) {
      dz = +1;
      tz -= MemZ;
    }

    // Multi-brick jumps (e.g. tx still out of [0, MemX)) are not supported here.
    if (tx < 0 || tx >= MemX || ty < 0 || ty >= MemY || tz < 0 || tz >= MemZ) {
      return out;
    }

    uint32_t target_block = idx.block_idx();
    if (dx != 0 || dy != 0 || dz != 0) {
      const uint32_t* connectivity = span.connectivity();
      if (connectivity == nullptr) {
        return out;
      }
      const uint32_t slot = SpanT::connectivity_slot(dx, dy, dz);
      target_block = connectivity[idx.block_idx() * SpanT::kSlotsPerBlock + slot];
      if (target_block == kInvalidBlockIdx) {
        return out;
      }
    }

    const BitMask* bitmask = span.bitmask();
    if (bitmask == nullptr || !bitmask[target_block].is_active(tx, ty, tz)) {
      return out;
    }

    const Idx target{target_block, static_cast<int8_t>(tx), static_cast<int8_t>(ty),
                     static_cast<int8_t>(tz)};
    out.value = ptr_[span.linear_offset(target, Components, component_id)];
    out.valid = true;
    return out;
  }

private:
  T* ptr_{nullptr};
};

}  // namespace carbon::detail::block
