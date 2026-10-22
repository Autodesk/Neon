#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/static_block.h>
#include <carbon/detail/validated_idx.h>
#include <carbon/integer3d.h>

#include <stddef.h>
#include <stdint.h>

namespace carbon::detail::block {

/**
 * Cross-family alias for the @c set_and_validate return carrier. Same shape
 * as @c dense::ValidatedIdx / @c dense_disg::ValidatedIdx — see
 * @c detail/validated_idx.h for the rationale.
 */
using ValidatedIdx = ::carbon::detail::Validated<Idx>;

/**
 * Per-GPU iteration / pitch / topology descriptor for the block-sparse grid.
 *
 * Mirrors Neon's @c bSpan from @c Neon-disg/libNeonDomain/include/Neon/domain/details/
 * bGrid/bSpan.h , with a deliberate cleanup: in carbon both the per-block bitmask @b and
 * the 27-way block-to-block connectivity table live on @c Span , not on @c Partition .
 * That keeps @c Partition a thin "view of values" (one @c T* ) — symmetric with
 * @c carbon::detail::dense::Partition — and means the @c Block -borrowed device pointers
 * have a single owner on the kernel side.
 *
 * Owns the following:
 *
 * - @c bitmask : pointer to the array of per-block bitmasks ( @c block_count entries of
 *   @c SBlock::BitMask ). Set once at construction by @c Block , read on every voxel
 *   visit by @c set_and_validate and on every neighbour walk by
 *   @c Partition::neighbour .
 * - @c connectivity : pointer to the 27-way per-block connectivity table
 *   ( @c block_count · 27 entries of @c uint32_t ). Slot @c (dz+1)·9 + (dy+1)·3 +
 *   (dx+1) holds the @c block_idx of the neighbour at integer offset @c (dx, dy, dz)
 *   in @c {-1, 0, 1}^3 , or @c kInvalidBlockIdx when the neighbour does not exist on
 *   this device. The (0, 0, 0) slot stores the block's own id (handy for uniform
 *   indexing). One table per @c (Block, GPU) — type-independent, so all
 *   @c Field<T, C> built over the same @c Block share it through their per-GPU
 *   @c Span .
 * - @c block_count : how many blocks this device's slab carries. Used by kernel launch
 *   shaping ( @c grid_x = block_count for @c DataView::Standard ) and by host-side
 *   iteration helpers.
 * - @c n_internal : how many of those blocks are @b internal (i.e. no face touches a
 *   slab edge that has a remote neighbour, so a stencil read of radius 1 stays
 *   entirely within this slab's owned bricks). Owned bricks are arranged in
 *   @c [0, n_internal) for internal and @c [n_internal, block_count) for boundary
 *   in storage; @c Block::ctor stable-partitions them at construction time. With
 *   @c block_count == n_internal (single global slab, or stencil radius zero) the
 *   Boundary view is empty and @c "Internal == Standard" .
 * - @c linear_offset : pure pitch math @c (block_idx · C + c) · MemX · MemY · MemZ +
 *   @c (x + MemX · y + MemX · MemY · z) , exposed so @c Partition stays a thin wrapper.
 */
template <typename SBlock>
class Span {
public:
  using BitMask = typename SBlock::BitMask;
  static constexpr int MemX = SBlock::MemX;
  static constexpr int MemY = SBlock::MemY;
  static constexpr int MemZ = SBlock::MemZ;
  static constexpr int VoxelCount = SBlock::VoxelCount;
  /** Slots per block in the connectivity table (always 27 for a 3D 1-radius stencil). */
  static constexpr uint32_t kSlotsPerBlock = 27U;

  CARBON_HD constexpr Span() noexcept = default;

  CARBON_HD constexpr Span(const BitMask* bitmask,
                           uint32_t block_count,
                           const uint32_t* connectivity = nullptr,
                           const int32_3d* origins = nullptr,
                           int level = 0,
                           int32_3d spacing = int32_3d{1,
                                                       1,
                                                       1}) noexcept
      : bitmask_(bitmask),
        connectivity_(connectivity),
        origins_(origins),
        block_count_(block_count),
        n_internal_(block_count),
        level_(level),
        spacing_(spacing) {}

  CARBON_HD constexpr Span(const BitMask* bitmask,
                           uint32_t block_count,
                           uint32_t n_internal,
                           const uint32_t* connectivity = nullptr,
                           const int32_3d* origins = nullptr,
                           int level = 0,
                           int32_3d spacing = int32_3d{1,
                                                       1,
                                                       1}) noexcept
      : bitmask_(bitmask),
        connectivity_(connectivity),
        origins_(origins),
        block_count_(block_count),
        n_internal_(n_internal),
        level_(level),
        spacing_(spacing) {}

  /** Pointer to the array of per-block bitmasks (length = @ref block_count ). */
  [[nodiscard]] CARBON_HD constexpr auto
  bitmask() const noexcept -> const BitMask* {
    return bitmask_;
  }

  /**
   * Pointer to the 27-way per-block connectivity table (length =
   * @ref block_count · @ref kSlotsPerBlock ). May be @c nullptr on a single-block
   * configuration that doesn't need cross-block neighbour walks; @c Partition::neighbour
   * treats a null connectivity as "no cross-block neighbour exists" so kernels still
   * compose correctly.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  connectivity() const noexcept -> const uint32_t* {
    return connectivity_;
  }

  /**
   * Pointer to the per-block voxel-origin table (length = @ref block_count ): the
   * global @c (x, y, z) of voxel @c (0, 0, 0) within each retained brick. Set at
   * @c Block construction time alongside @ref bitmask and @ref connectivity ;
   * used by @ref global_location to recover a kernel-side @c Idx 's global voxel
   * coordinates. May be @c nullptr on configurations that don't need the lookup;
   * @ref global_location then returns @c {0,0,0} .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  origins() const noexcept -> const int32_3d* {
    return origins_;
  }

  /**
   * Refinement level this span belongs to in a multi-resolution (mGrid) tower —
   * @c 0 is the finest level, larger is coarser. Defaulted to @c 0 so the
   * single-level @c Block builds spans at the finest level with no behaviour change.
   * mGrid stacks one @c Block (and therefore one @c Span per GPU) per level and tags
   * each with its level so cross-level kernels can resolve parent/child geometry.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  level() const noexcept -> int {
    return level_;
  }

  /**
   * Voxel spacing of this level expressed in finest-level units (how many level-0
   * voxels one voxel here spans per axis). @c {1, 1, 1} at the finest level — the
   * default, so the single-level @c Block is unaffected — and grows by the per-level
   * refinement factor as levels coarsen (e.g. @c {2, 2, 2} , @c {4, 4, 4} for a 2×
   * octree). Carried as topology metadata for mGrid; @ref global_location keeps its
   * single-level semantics (origin + intra) in this pass — the cross-level voxel↔world
   * mapping that consumes @c spacing lands with the mGrid port.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  spacing() const noexcept -> int32_3d {
    return spacing_;
  }

  /**
   * Connectivity flat-offset for an integer @c (dx, dy, dz) in @c {-1, 0, 1}^3 .
   * Pure function of the offset — independent of @c SBlock parameters — but kept on
   * @c Span so callers don't have to instantiate @c Partition just to compute a slot
   * index when building the connectivity table on the host side.
   */
  [[nodiscard]] CARBON_HD constexpr static auto
  connectivity_slot(int dx,
                    int dy,
                    int dz) noexcept -> uint32_t {
    return static_cast<uint32_t>((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1));
  }

  /** Number of retained blocks on this device. */
  [[nodiscard]] CARBON_HD constexpr auto
  block_count() const noexcept -> uint32_t {
    return block_count_;
  }

  /**
   * Number of internal blocks on this device — blocks at @c [0, n_internal) that
   * have no slab-edge face facing a remote neighbour (so a stencil of radius 1
   * read stays in this slab's owned bricks). Boundary blocks are at
   * @c [n_internal, block_count) .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  n_internal() const noexcept -> uint32_t {
    return n_internal_;
  }

  /**
   * Number of blocks the kernel iterates over for @p view :
   *   - Standard  → @c block_count .
   *   - Internal  → @c n_internal .
   *   - Boundary  → @c block_count - n_internal .
   * The launch helper picks @c grid_x = iter_block_count(view) ; the kernel
   * translates the launch-grid x index back to a storage @c block_idx via
   * @ref view_block_idx .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  iter_block_count(::carbon::DataView view) const noexcept -> uint32_t {
    switch (view) {
      case ::carbon::DataView::Standard:
        return block_count_;
      case ::carbon::DataView::Internal:
        return n_internal_;
      case ::carbon::DataView::Boundary:
        return (block_count_ > n_internal_) ? (block_count_ - n_internal_) : 0U;
    }
    return block_count_;
  }

  /**
   * Translate a launch-grid @p iter_idx (in @c [0, iter_block_count(view)) ) into a
   * storage block index (in @c [0, block_count) ). Standard / Internal pass through;
   * Boundary shifts by @c n_internal_ .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  view_block_idx(::carbon::DataView view,
                 uint32_t iter_idx) const noexcept -> uint32_t {
    switch (view) {
      case ::carbon::DataView::Standard:
      case ::carbon::DataView::Internal:
        return iter_idx;
      case ::carbon::DataView::Boundary:
        return n_internal_ + iter_idx;
    }
    return iter_idx;
  }

  /**
   * Build an @ref Idx for the thread at block @p block_idx , intra-block @p ix , @p iy ,
   * @p iz @b and check the per-block bitmask. Out-of-block (negative or @c >= MemX/Y/Z )
   * coordinates short-circuit to @c valid=false without touching the bitmask. A bit-clear
   * voxel inside the brick produces an @c Idx with the right coordinates but
   * @c valid=false so the kernel skips it.
   *
   * Mirrors @c bSpan::setAndValidateGPUDevice .
   *
   * Out-param + @c bool return: the historical Block-side shape. The
   * value-returning overload below ( @c -> ValidatedIdx ) is the canonical
   * cross-family form (matches Dense / DenseDisg's return shape); both stay
   * for back-compat per P2.α's "additive only" contract.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  set_and_validate(uint32_t block_idx,
                   int ix,
                   int iy,
                   int iz,
                   Idx& out_idx) const noexcept -> bool {
    if (ix < 0 || ix >= MemX || iy < 0 || iy >= MemY || iz < 0 || iz >= MemZ) {
      return false;
    }
    if (block_idx >= block_count_ || block_idx == kInvalidBlockIdx) {
      return false;
    }
    out_idx =
        Idx{block_idx, static_cast<int8_t>(ix), static_cast<int8_t>(iy), static_cast<int8_t>(iz)};
    if (bitmask_ == nullptr) {
      return false;
    }
    return bitmask_[block_idx].is_active(ix, iy, iz);
  }

  /**
   * Canonical (cross-family) value-returning form of @ref set_and_validate .
   * Same predicate logic as the out-param overload above; the returned
   * @ref ValidatedIdx ( @c carbon::detail::Validated<Idx> ) carries the
   * resulting @c Idx and the @c valid flag together so callers do not have
   * to thread an out-param @c Idx through their iteration body. Source-compat
   * alias for the dense / disg @c set_and_validate(view, ix, iy, iz) shape;
   * use this overload when writing new cross-grid code.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  set_and_validate(uint32_t block_idx,
                   int ix,
                   int iy,
                   int iz) const noexcept -> ValidatedIdx {
    ValidatedIdx out{};
    out.valid = set_and_validate(block_idx, ix, iy, iz, out.idx);
    return out;
  }

  /**
   * Convert a kernel-side @c Idx to its element offset inside the per-GPU SoA buffer:
   *
   * @code
   * (block_idx · C + c) · MemX · MemY · MemZ + (x + MemX · y + MemX · MemY · z)
   * @endcode
   *
   * Same formula Neon uses in @c bPartition::helpGetValidIdxPitchExplicit . @c C is
   * @b the @c Field 's component count, passed in by the caller because @c Span itself
   * is component-agnostic (one storage span backs every cardinality).
   */
  [[nodiscard]] CARBON_HD constexpr auto
  linear_offset(const Idx& idx,
                int components,
                int component_id) const noexcept -> int64_t {
    const auto intra = idx.intra();
    const int64_t in_block =
        static_cast<int64_t>(intra.x) + static_cast<int64_t>(MemX) * static_cast<int64_t>(intra.y) +
        static_cast<int64_t>(MemX) * static_cast<int64_t>(MemY) * static_cast<int64_t>(intra.z);
    const int64_t block_card_pitch =
        static_cast<int64_t>(idx.block_idx()) * static_cast<int64_t>(components) +
        static_cast<int64_t>(component_id);
    return block_card_pitch * static_cast<int64_t>(VoxelCount) + in_block;
  }

  /** Total field elements per cardinality plane (same value as @c block_count · VoxelCount ). */
  [[nodiscard]] CARBON_HD constexpr auto
  per_card_elements() const noexcept -> int64_t {
    return static_cast<int64_t>(block_count_) * static_cast<int64_t>(VoxelCount);
  }

  /** Total field elements for @p components cardinality planes. */
  [[nodiscard]] CARBON_HD constexpr auto
  linear_storage_elements(int components) const noexcept -> int64_t {
    return per_card_elements() * static_cast<int64_t>(components);
  }

  /**
   * Convert the data-block coordinate @p idx into its @e tray (user-block) view —
   * the @ref MicroIndex carbon's multi-resolution (mGrid) construction and cross-level
   * addressing key off. Mirrors Neon's @c bIndex::getMicroIndex .
   *
   * Each memory brick fans out into @c BlockRatioX·BlockRatioY·BlockRatioZ user-block
   * trays; the returned tray index is @c block_idx scaled by that fan-out plus the local
   * tray offset, and @c in_tray is the voxel's offset inside its user block. In the
   * single-resolution default (@c User == Mem , all ratios 1) this is the identity:
   * @c tray_block_idx == block_idx and @c in_tray == intra .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  to_micro_index(const Idx& idx) const noexcept -> MicroIndex {
    const auto intra = idx.intra();
    const int32_t block_fanout = static_cast<int32_t>(SBlock::BlockRatioX) *
                                 static_cast<int32_t>(SBlock::BlockRatioY) *
                                 static_cast<int32_t>(SBlock::BlockRatioZ);
    const int32_t ex_block_offset = static_cast<int32_t>(idx.block_idx()) * block_fanout;
    const int32_t tray_x = static_cast<int32_t>(intra.x) / static_cast<int32_t>(SBlock::UserX);
    const int32_t tray_y = static_cast<int32_t>(intra.y) / static_cast<int32_t>(SBlock::UserY);
    const int32_t tray_z = static_cast<int32_t>(intra.z) / static_cast<int32_t>(SBlock::UserZ);
    const int32_t ex_tray_offset = tray_x + tray_y * static_cast<int32_t>(SBlock::BlockRatioX) +
                                   tray_z * static_cast<int32_t>(SBlock::BlockRatioX) *
                                       static_cast<int32_t>(SBlock::BlockRatioY);
    return MicroIndex{
        ex_block_offset + ex_tray_offset, static_cast<int8_t>(intra.x % SBlock::UserX),
        static_cast<int8_t>(intra.y % SBlock::UserY), static_cast<int8_t>(intra.z % SBlock::UserZ)};
  }

  /**
   * Inverse of @ref to_micro_index : rebuild the data-block coordinate @ref Idx from its
   * @ref MicroIndex tray view. Mirrors Neon's @c bIndex::init . Round-trips
   * @c from_micro_index(to_micro_index(idx)) == idx for any in-range @p idx ; in the
   * single-resolution default it is the identity.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  from_micro_index(const MicroIndex& micro) const noexcept -> Idx {
    const int32_t ratio_x = static_cast<int32_t>(SBlock::BlockRatioX);
    const int32_t ratio_y = static_cast<int32_t>(SBlock::BlockRatioY);
    const int32_t block_fanout = ratio_x * ratio_y * static_cast<int32_t>(SBlock::BlockRatioZ);
    const int32_t tray = micro.tray_block_idx();
    const auto in_tray = micro.in_tray();
    const uint32_t block_idx = static_cast<uint32_t>(tray / block_fanout);
    int32_t remainder = tray % block_fanout;
    const int32_t rem_z = remainder / (ratio_x * ratio_y);
    const int32_t iz =
        static_cast<int32_t>(in_tray.z) + rem_z * static_cast<int32_t>(SBlock::UserZ);
    remainder = remainder % (ratio_x * ratio_y);
    const int32_t rem_y = remainder / ratio_x;
    const int32_t iy =
        static_cast<int32_t>(in_tray.y) + rem_y * static_cast<int32_t>(SBlock::UserY);
    const int32_t rem_x = remainder % ratio_x;
    const int32_t ix =
        static_cast<int32_t>(in_tray.x) + rem_x * static_cast<int32_t>(SBlock::UserX);
    return Idx{block_idx, static_cast<int8_t>(ix), static_cast<int8_t>(iy),
               static_cast<int8_t>(iz)};
  }

  /**
   * Recover the global voxel coordinates of voxel @p idx . Returns
   * @c origins()[idx.block_idx()] + idx.intra() , cast to @c int64_3d for
   * symmetry with @c carbon::detail::dense::Span::global_location . The result
   * is meaningful only when @p idx points at a real block on this span (i.e.
   * @c idx.valid_block() and @c idx.block_idx() < block_count() ); for an
   * invalid @c Idx or a span built without origins this method returns
   * @c {0, 0, 0} .
   *
   * Component-agnostic — origin is a topology property of the @c (Block, GPU)
   * pair, shared across every @c Field built over the same @c Block .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  global_location(const Idx& idx) const noexcept -> int64_3d {
    if (origins_ == nullptr || !idx.valid_block() || idx.block_idx() >= block_count_) {
      return {0, 0, 0};
    }
    const int32_3d origin = origins_[idx.block_idx()];
    const int8_3d intra = idx.intra();
    return {static_cast<int64_t>(origin.x) + static_cast<int64_t>(intra.x),
            static_cast<int64_t>(origin.y) + static_cast<int64_t>(intra.y),
            static_cast<int64_t>(origin.z) + static_cast<int64_t>(intra.z)};
  }

private:
  const BitMask* bitmask_{nullptr};
  const uint32_t* connectivity_{nullptr};
  const int32_3d* origins_{nullptr};
  uint32_t block_count_{0};
  uint32_t n_internal_{0};
  int level_{0};
  int32_3d spacing_{1, 1, 1};
};

}  // namespace carbon::detail::block
