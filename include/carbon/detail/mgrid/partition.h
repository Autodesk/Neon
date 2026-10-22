#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/block/static_block.h>
#include <carbon/detail/grid_concepts.h>
#include <carbon/integer3d.h>
#include <carbon/neighbour_value.h>

#include <stdint.h>

namespace carbon::detail::mgrid {

/**
 * Device-visible cross-level view of one cardinality plane of a @c MultiResField at a
 * fixed level @c L of the multi-resolution (mGrid) tower. Mirrors Neon's @c mPartition
 * (@c Neon/libNeonDomain/include/Neon/domain/details/mGrid/mPartition*.h ) and satisfies
 * @ref ::carbon::detail::MultiResPartitionConcept .
 *
 * The own-level surface (@ref read / @ref write ) is identical to @c block::Partition : a
 * thin @c {T*} over the level's per-GPU SoA storage, with the bitmask / connectivity /
 * pitch math living on the @c block::Span the caller threads in. What this type adds is the
 * @b cross-level surface: from a voxel at level @c L it resolves the @b parent voxel at
 * level @c L+1 and the @b child voxels at level @c L-1 , reading the adjacent levels'
 * storage directly. Cross-level access is transfer-free because mGrid keeps a parent block
 * and its children co-resident on the same GPU (single-GPU in this pass; the co-resident
 * partitioning rule generalises that to multi-GPU — see @c docs/design_mgrid.md §4).
 *
 * @par Backing data (set by @c MultiResField::make_partition )
 *
 * - own level: @c mem_ (this level's storage) + @c ref_factor_ (octree branching @c R );
 * - parent (level @c L+1 , present iff @c L < depth-1 ): @c parent_mem_ + @c parent_span_ +
 *   @c parent_block_id_ (one parent @b block id per child block on this level);
 * - child (level @c L-1 , present iff @c L >= 1 ): @c child_mem_ + @c child_span_ +
 *   @c child_block_id_ (one child block id per @b voxel of each block on this level).
 *
 * A null parent / child pointer set (e.g. the coarsest level has no parent, the finest no
 * child) makes @ref has_parent / @ref has_children return @c false , so a kernel that walks
 * the tower edges composes correctly without special-casing.
 *
 * @par Cross-level arithmetic (level-local voxel coordinates; reconciled to Neon)
 *
 * Adjacent levels differ by exactly @c R per axis. For a voxel @c idx at level @c L :
 * - @c get_parent : @c parent.block = parent_block_id_[idx.block] ;
 *   @c parent.intra = (global_voxel(idx) / R) mod Mem per axis.
 * - @c get_child(offset) , @c offset in @c [0,R)^3 :
 *   @c child.block = child_block_id_[idx.block · VoxelCount + voxel_pitch(idx.intra)] ;
 *   @c child.intra = (R · idx.intra + offset) mod Mem per axis.
 *
 * See @c docs/design_mgrid.md §2.3 for the derivation and the Neon line references.
 */
template <typename T, int Components, typename SBlock>
class Partition {
public:
  using SpanT = block::Span<SBlock>;
  using Idx = block::Idx;
  using BitMask = typename SBlock::BitMask;
  static constexpr int MemX = SBlock::MemX;
  static constexpr int MemY = SBlock::MemY;
  static constexpr int MemZ = SBlock::MemZ;
  static constexpr int VoxelCount = SBlock::VoxelCount;

  CARBON_HD constexpr Partition() noexcept = default;

  CARBON_HD constexpr Partition(T* mem,
                                int ref_factor,
                                T* parent_mem,
                                SpanT parent_span,
                                const uint32_t* parent_block_id,
                                T* child_mem,
                                SpanT child_span,
                                const uint32_t* child_block_id) noexcept
      : mem_(mem),
        parent_mem_(parent_mem),
        child_mem_(child_mem),
        parent_block_id_(parent_block_id),
        child_block_id_(child_block_id),
        parent_span_(parent_span),
        child_span_(child_span),
        ref_factor_(ref_factor) {}

  /** Raw device pointer to this level's storage. */
  [[nodiscard]] CARBON_HD constexpr auto
  ptr() const noexcept -> T* {
    return mem_;
  }

  /**
   * Read the @p component_id -th component of voxel @p idx at this level. @p span must be
   * the level's own @c block::Span (so the pitch math agrees on @c Components ).
   */
  [[nodiscard]] CARBON_HD constexpr auto
  read(const SpanT& span,
       const Idx& idx,
       int component_id) const noexcept -> const T& {
    return mem_[span.linear_offset(idx, Components, component_id)];
  }

  /** Write @p value into voxel @p idx 's @p component_id -th component at this level. */
  CARBON_HD constexpr auto
  write(const SpanT& span,
        const Idx& idx,
        int component_id,
        T value) const noexcept -> void {
    mem_[span.linear_offset(idx, Components, component_id)] = value;
  }

  /** Does voxel @p idx have a parent voxel at the next-coarser level? */
  [[nodiscard]] CARBON_HD constexpr auto
  has_parent([[maybe_unused]] const SpanT& span,
             const Idx& idx) const noexcept -> bool {
    if (parent_mem_ == nullptr || parent_block_id_ == nullptr || !idx.valid_block()) {
      return false;
    }
    return parent_block_id_[idx.block_idx()] != block::kInvalidBlockIdx;
  }

  /** Does voxel @p idx have at least one child block at the next-finer level? */
  [[nodiscard]] CARBON_HD constexpr auto
  has_children([[maybe_unused]] const SpanT& span,
               const Idx& idx) const noexcept -> bool {
    if (child_mem_ == nullptr || child_block_id_ == nullptr || !idx.valid_block()) {
      return false;
    }
    return child_block_id_[child_pitch_(idx)] != block::kInvalidBlockIdx;
  }

  /**
   * @c Idx of voxel @p idx 's parent at level @c L+1 . Returns a sentinel (invalid) @c Idx
   * when @ref has_parent is @c false . @p span is this level's own span (used to recover
   * the global voxel coordinates).
   */
  [[nodiscard]] CARBON_HD constexpr auto
  get_parent(const SpanT& span,
             const Idx& idx) const noexcept -> Idx {
    if (!has_parent(span, idx)) {
      return Idx{};
    }
    const uint32_t parent_block = parent_block_id_[idx.block_idx()];
    const int64_3d g = span.global_location(idx);
    const int px = static_cast<int>((g.x / ref_factor_) % MemX);
    const int py = static_cast<int>((g.y / ref_factor_) % MemY);
    const int pz = static_cast<int>((g.z / ref_factor_) % MemZ);
    return Idx{parent_block, static_cast<int8_t>(px), static_cast<int8_t>(py),
               static_cast<int8_t>(pz)};
  }

  /**
   * @c Idx of the child voxel of @p idx at local octant @p child_offset (each component in
   * @c [0, ref_factor) ) at level @c L-1 . Returns a sentinel (invalid) @c Idx when
   * @ref has_children is @c false .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  get_child([[maybe_unused]] const SpanT& span,
            const Idx& idx,
            int8_3d child_offset) const noexcept -> Idx {
    if (!has_children(span, idx)) {
      return Idx{};
    }
    const uint32_t child_block = child_block_id_[child_pitch_(idx)];
    const auto intra = idx.intra();
    const int cx =
        (ref_factor_ * static_cast<int>(intra.x) + static_cast<int>(child_offset.x)) % MemX;
    const int cy =
        (ref_factor_ * static_cast<int>(intra.y) + static_cast<int>(child_offset.y)) % MemY;
    const int cz =
        (ref_factor_ * static_cast<int>(intra.z) + static_cast<int>(child_offset.z)) % MemZ;
    return Idx{child_block, static_cast<int8_t>(cx), static_cast<int8_t>(cy),
               static_cast<int8_t>(cz)};
  }

  /**
   * Value of voxel @p idx 's parent component @p component_id . @c valid is @c false when
   * @p idx has no parent or the parent voxel is inactive in the parent level's bitmask.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  parent(const SpanT& span,
         const Idx& idx,
         int component_id) const noexcept -> ::carbon::NeighbourValue<T> {
    ::carbon::NeighbourValue<T> out{};
    const Idx p = get_parent(span, idx);
    if (!p.valid_block()) {
      return out;
    }
    const BitMask* mask = parent_span_.bitmask();
    const auto intra = p.intra();
    if (mask == nullptr || !mask[p.block_idx()].is_active(intra.x, intra.y, intra.z)) {
      return out;
    }
    out.value = parent_mem_[parent_span_.linear_offset(p, Components, component_id)];
    out.valid = true;
    return out;
  }

  /**
   * Value of voxel @p idx 's child at octant @p child_offset , component @p component_id .
   * @c valid is @c false when there is no child block or the child voxel is inactive in
   * the child level's bitmask.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  child(const SpanT& span,
        const Idx& idx,
        int8_3d child_offset,
        int component_id) const noexcept -> ::carbon::NeighbourValue<T> {
    ::carbon::NeighbourValue<T> out{};
    const Idx c = get_child(span, idx, child_offset);
    if (!c.valid_block()) {
      return out;
    }
    const BitMask* mask = child_span_.bitmask();
    const auto intra = c.intra();
    if (mask == nullptr || !mask[c.block_idx()].is_active(intra.x, intra.y, intra.z)) {
      return out;
    }
    out.value = child_mem_[child_span_.linear_offset(c, Components, component_id)];
    out.valid = true;
    return out;
  }

private:
  [[nodiscard]] CARBON_HD constexpr auto
  child_pitch_(const Idx& idx) const noexcept -> int64_t {
    const auto intra = idx.intra();
    const int64_t voxel =
        static_cast<int64_t>(intra.x) + static_cast<int64_t>(MemX) * static_cast<int64_t>(intra.y) +
        static_cast<int64_t>(MemX) * static_cast<int64_t>(MemY) * static_cast<int64_t>(intra.z);
    return static_cast<int64_t>(idx.block_idx()) * static_cast<int64_t>(VoxelCount) + voxel;
  }

  T* mem_{nullptr};
  T* parent_mem_{nullptr};
  T* child_mem_{nullptr};
  const uint32_t* parent_block_id_{nullptr};
  const uint32_t* child_block_id_{nullptr};
  SpanT parent_span_{};
  SpanT child_span_{};
  int ref_factor_{1};
};

namespace concept_anchor {

// Compile-time anchor: the octree mGrid Partition must satisfy the cross-level concept.
// Failing this points at the first get_parent / get_child / parent / child member that
// drifted from MultiResPartitionConcept, rather than a deep instantiation stack at a use
// site. Mirrors the dense_like_span_partition_contract anchors used by the kernel impls.
using AnchorSBlock = block::StaticBlock<8, 8, 8, 2, 2, 2, true>;
static_assert(::carbon::detail::MultiResPartitionConcept<Partition<float,
                                                                   1,
                                                                   AnchorSBlock>,
                                                         block::Span<AnchorSBlock>,
                                                         block::Idx,
                                                         float>,
              "mgrid::Partition must satisfy detail::MultiResPartitionConcept");

}  // namespace concept_anchor

}  // namespace carbon::detail::mgrid
