#pragma once

// Public API for the carbon::MultiResBlock (multi-resolution / mGrid) grid family.
//
// Port of Neon's mGrid: a tower of carbon::Block (block-sparse) levels — level 0 finest,
// increasing levels coarser — with cross-level parent/child addressing exposed through a
// MultiResPartition that satisfies detail::MultiResPartitionConcept. Multi-GPU + cross-rank
// (MPI/NCCL) co-resident partitioning has landed: the coarsest level is z-brick partitioned
// and every finer level inherits the same world-space cut planes, so a parent brick and all
// its descendants are co-resident on one GPU/rank and cross-level access is Transfer-free by
// construction (docs/design_mgrid.md §4). KernelLoader/Skeleton integration, per-level
// halo, and bind_cross_level for cross-level Skeleton dependencies are all shipped — see
// docs/CHANGELOG.md (2026-06 entries).
//
// Assumes a uniform octree refinement factor (R == SBlock::UserX), reusing one
// multi-resolution StaticBlock for every level.

#include <carbon/block.h>
#include <carbon/detail/block/static_block.h>
#include <carbon/detail/mgrid/descriptor.h>
#include <carbon/detail/mgrid/field.h>
#include <carbon/detail/mgrid/locator.h>
#include <carbon/detail/mgrid/partition.h>
#include <carbon/export.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/system.h>

#include <stdint.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon {

class Kernel;  // forward (defined in <carbon/kernel.h>); MultiResBlock::make_kernel returns one
               // and is defined in the kernel.h tail include <carbon/detail/kernel_mgrid.inl>.

/**
 * Default per-level brick policy for @c MultiResBlock : an 8×8×8 memory brick subdivided
 * into 2×2×2 user blocks (octree branching factor 2), tagged multi-resolution. Matches
 * Neon's @c kStaticBlock for mGrid (@c StaticBlock<8,8,8> memory brick + @c userBlockSize 2 ).
 */
using DefaultMultiResBlock = detail::block::StaticBlock<8, 8, 8, 2, 2, 2, true>;

/**
 * Multi-resolution block-sparse grid (port of Neon's @c mGrid). Holds a tower of
 * @c Block<SBlock> levels over a shared Cartesian domain — level 0 is the finest and spans
 * @p finest_extent voxels; each coarser level @c l has voxel extent
 * @c ceil_div(finest_extent, R^l) where @c R is the (uniform) refinement factor. A
 * @ref MultiResField allocates one @c Block::Field per level and a kernel reads across levels
 * through the @ref detail::mgrid::Partition this grid wires up.
 *
 * @par Partitioning (multi-GPU + cross-rank, co-resident)
 *
 * The coarsest level is partitioned across the @c System 's GPUs (single-process) or ranks
 * (one-rank-per-GPU under @c CARBON_HAVE_MPI_NCCL ) by even z-brick rows; every finer level
 * inherits the @b same world-space cut planes (the coarsest cut projected down by the
 * refinement factor — see @ref Block::ZBrickPartition and @c docs/design_mgrid.md §4). This
 * keeps a coarse cell and all of its finer descendants on one GPU/rank, so the cross-level
 * @c get_parent / @c get_child / @c parent / @c child path is a pure local dereference — no
 * peer copy, no @c Transfer , no NCCL. The cross-level block-id tables are addressed per GPU
 * accordingly. A single-GPU, single-rank @c System uses the plain even split (no projection)
 * and is byte-for-byte the original single-GPU layout. Uniform octree refinement
 * (@c R == @c SBlock::UserX ) is enforced.
 *
 * @par Multi-GPU z-alignment precondition
 *
 * When the partition spans more than one slab (multi-GPU or multi-rank), @p finest_extent .z
 * must be a multiple of @c MemZ · R^(depth-1) (the coarsest brick's z-thickness in level-0
 * voxels) so the coarsest cut planes project to exact finer-level brick-row boundaries with no
 * remainder. A single-slab @c System has no such constraint (relaxing it for the multi-slab
 * case via descendant-weighted cuts is a documented follow-up).
 *
 * @tparam SBlock @c StaticBlock brick policy shared by every level. Defaults to
 *                @ref DefaultMultiResBlock (the 8³ / 2³ octree brick).
 */
template <typename SBlock = DefaultMultiResBlock>
class MultiResBlock {
public:
  using BlockGrid = ::carbon::Block<SBlock>;
  using Descriptor = detail::mgrid::MultiResDescriptor;
  using Idx = detail::block::Idx;
  using Span = detail::block::Span<SBlock>;
  // Brick iteration, same as Block: this is what grid_has_dense_iteration_v probes for, and
  // without it the launch traits would route the tower down the z-slab dense path.
  using BlockPolicy = SBlock;

  template <typename T, int Components>
  using Partition = detail::mgrid::Partition<T, Components, SBlock>;

  template <typename T, int Components>
  using Field = detail::mgrid::Field<T, Components, SBlock>;

  static constexpr int MemX = SBlock::MemX;
  static constexpr int MemY = SBlock::MemY;
  static constexpr int MemZ = SBlock::MemZ;
  static constexpr int VoxelCount = SBlock::VoxelCount;

  /**
   * Per-level activity predicate. Called once per voxel of every level as
   * @c is_active(level, voxel) , where @c voxel is in @b level- @c level voxel coordinates
   * (i.e. @c [0, level_extent(level)) per axis). A brick on a level is retained iff at least
   * one of its voxels passes the predicate at that level.
   */
  using ActiveCellPredicate = std::function<bool(int, const int64_3d&)>;

  MultiResBlock() = default;

  /**
   * Build the tower over @p system 's GPU(s) / rank(s).
   *
   * @param system        Backing System. Any GPU count single-process; one GPU per rank under
   *                      @c CARBON_HAVE_MPI_NCCL . The coarsest level is partitioned across
   *                      slabs and the cut is projected to every finer level (co-resident).
   * @param finest_extent Level-0 Cartesian voxel extent (all components positive). When the
   *                      partition spans more than one slab, @c finest_extent.z must be a
   *                      multiple of @c MemZ · R^(depth-1) (see the class-level note).
   * @param depth         Number of levels (>= 1).
   * @param is_active     Per-level activity predicate (see @ref ActiveCellPredicate ).
   * @param ref_factor    Uniform octree branching factor. Defaults to @c SBlock::UserX and
   *                      must equal it (Neon's @c refFactor == userBlockSize coupling).
   * @param name          Optional debug label.
   *
   * @throws std::invalid_argument on @c depth < 1 , a @c ref_factor that does not match the
   *         brick's user-block size, a non-positive extent, a multi-slab partition whose
   *         coarsest brick z-extent is smaller than the slab count, or a multi-slab
   *         @c finest_extent.z that violates the z-alignment precondition.
   */
  MultiResBlock(System system,
                int64_3d finest_extent,
                int depth,
                const ActiveCellPredicate& is_active,
                int ref_factor = SBlock::UserX,
                std::string name = {});

  [[nodiscard]] auto
  depth() const noexcept -> int {
    return static_cast<int>(levels_.size());
  }

  [[nodiscard]] auto
  descriptor() const noexcept -> const Descriptor& {
    return descriptor_;
  }

  [[nodiscard]] auto
  ref_factor() const noexcept -> int {
    return descriptor_.ref_factor();
  }

  [[nodiscard]] auto
  finest_extent() const noexcept -> const int64_3d& {
    return finest_extent_;
  }

  /** Local GPU count. Every level is partitioned over the same devices. */
  [[nodiscard]] auto
  gpu_count() const noexcept -> size_t {
    return levels_.empty() ? 0U : levels_.front().gpu_count();
  }

  [[nodiscard]] auto
  system() noexcept -> System& {
    return system_;
  }

  [[nodiscard]] auto
  system() const noexcept -> const System& {
    return system_;
  }

  /** Per-level @c Block grid (0 = finest). Throws @c std::out_of_range on a bad level. */
  [[nodiscard]] auto
  grid(int level) -> BlockGrid& {
    return levels_.at(static_cast<size_t>(level));
  }

  [[nodiscard]] auto
  grid(int level) const -> const BlockGrid& {
    return levels_.at(static_cast<size_t>(level));
  }

  /**
   * Brick-index tower for @p gpu_id : every level's @ref detail::block::Locator plus its
   * @c Span , so a device function can resolve an arbitrary world point to the finest active
   * cell covering it (see @ref detail::mgrid::MultiResLocator ). Each level's table is built
   * lazily on first use and cached on that level's @c Block .
   *
   * The world frame is shared by the whole tower: level @c l 's voxel is
   * @c base_spacing · R^l wide, matching @c Descriptor::voxel_spacing . Because the coarsest
   * z-cut is projected to every level, all levels report the same owned world-z band.
   *
   * @param gpu_id       Local GPU index (relative id inside this rank's @c System ).
   * @param world_origin World position of the level-0 domain corner @c (0,0,0) .
   * @param base_spacing World size of one @b finest-level voxel. Must be positive.
   *
   * @throws std::out_of_range when @c depth() exceeds @c MultiResLocator::k_max_levels .
   */
  [[nodiscard]] auto
  locator(size_t gpu_id = 0,
          float32_3d world_origin = {},
          float base_spacing = 1.0F) const -> detail::mgrid::MultiResLocator<SBlock> {
    using LocatorT = detail::mgrid::MultiResLocator<SBlock>;
    if (depth() > LocatorT::k_max_levels) {
      log::throw_with_log<std::out_of_range>(
          "carbon::MultiResBlock::locator: depth {} exceeds MultiResLocator::k_max_levels ({}); "
          "raise the MaxLevels template argument to capture this tower in a kernel.",
          depth(), LocatorT::k_max_levels);
    }
    LocatorT tower{};
    tower.depth = depth();
    for (int l = 0; l < depth(); ++l) {
      const float spacing = base_spacing * static_cast<float>(descriptor_.voxel_spacing(l));
      tower.level_locator[l] = grid(l).locator(gpu_id, world_origin, spacing);
      tower.level_span[l] = grid(l).span(gpu_id);
    }
    return tower;
  }

  /**
   * Device pointer to the parent-block-id table for @p level on @p gpu_id : one entry per
   * block on @p level holding the @c block_idx of its parent brick at @c level+1 (or
   * @c detail::block::kInvalidBlockIdx ). @c nullptr for the coarsest level (no parent) or
   * an empty level.
   */
  [[nodiscard]] auto
  parent_block_id_dev(int level,
                      size_t gpu_id = 0) const noexcept -> const uint32_t* {
    if (level < 0 || level >= depth() - 1 ||
        gpu_id >= parent_block_id_dev_[static_cast<size_t>(level)].size()) {
      return nullptr;
    }
    const auto& alloc = parent_block_id_dev_[static_cast<size_t>(level)][gpu_id];
    return alloc.size() == 0U ? nullptr : alloc.data();
  }

  /**
   * Device pointer to the child-block-id table for @p level on @p gpu_id : one entry per
   * @b voxel of each block on @p level (@c block_count(level) · @c VoxelCount entries) holding
   * the @c block_idx of the child brick at @c level-1 containing that voxel's children (or
   * @c detail::block::kInvalidBlockIdx ). @c nullptr for the finest level (no children) or an
   * empty level.
   */
  [[nodiscard]] auto
  child_block_id_dev(int level,
                     size_t gpu_id = 0) const noexcept -> const uint32_t* {
    if (level < 1 || level >= depth() ||
        gpu_id >= child_block_id_dev_[static_cast<size_t>(level)].size()) {
      return nullptr;
    }
    const auto& alloc = child_block_id_dev_[static_cast<size_t>(level)][gpu_id];
    return alloc.size() == 0U ? nullptr : alloc.data();
  }

  /** Host mirror of @ref parent_block_id_dev for @p level on @p gpu_id (used by host tests). */
  [[nodiscard]] auto
  parent_block_id_host(int level,
                       size_t gpu_id = 0) const -> const std::vector<uint32_t>& {
    return parent_block_id_host_.at(static_cast<size_t>(level)).at(gpu_id);
  }

  /** Host mirror of @ref child_block_id_dev for @p level on @p gpu_id (used by host tests). */
  [[nodiscard]] auto
  child_block_id_host(int level,
                      size_t gpu_id = 0) const -> const std::vector<uint32_t>& {
    return child_block_id_host_.at(static_cast<size_t>(level)).at(gpu_id);
  }

  /**
   * Allocate a @ref Field over the whole tower: one @c Block::Field per level, sized to that
   * level's retained bricks. @p side selects which memory sides are reserved (forwarded to
   * each level's @c Block::newField ).
   */
  template <typename T,
            int Components>
  [[nodiscard]] auto
  newField(MemorySide side = MemorySide::Device,
           std::string name = {}) const -> Field<T,
                                                 Components>;

  /**
   * @brief Build a device compute @c Kernel that runs over a single @p level of the tower.
   *
   * The launch geometry is @c level 's @c Block (one CUDA block per retained brick, one
   * thread per voxel); inside the loading lambda, bind a @ref Field with
   * @c loader.read(field.level_view(level)) / @c write / @c read_write to get the cross-level
   * @ref detail::mgrid::Partition for @c (level, gpu) , so the per-cell body
   * @c (Span, Idx) -> void can read/write its own level @b and walk to the parent ( @c L+1 )
   * and child ( @c L-1 ) levels transfer-free (co-residency, @c docs/design_mgrid.md §4).
   *
   * Forwards to @c grid(level).make_kernel — multi-GPU and (one-rank-per-GPU) cross-rank are
   * inherited from the Block launch path. Defined in the @c <carbon/kernel.h> tail include
   * @c <carbon/detail/kernel_mgrid.inl> ; call from a @c .cu TU (CUDA) like any device kernel.
   *
   * @param level       Level to iterate (0 = finest). @c [0, depth()) .
   * @param loading_fn  Loading callable @c (KernelLoader&) -> F , @c F a @c (Span, Idx)
   *                    device callable.
   * @param shared_mem  Dynamic shared memory bytes per launch (default 0).
   *
   * @throws std::out_of_range when @p level is outside @c [0, depth()) .
   */
  template <typename L>
  [[nodiscard]] auto
  make_kernel(int level,
              L&& loading_fn,
              size_t shared_mem = 0U) const -> ::carbon::Kernel;

  /**
   * Host (OpenMP) twin of @ref make_kernel — forwards to @c grid(level).make_host_kernel .
   */
  template <typename L>
  [[nodiscard]] auto
  make_host_kernel(int level,
                   L&& loading_fn) const -> ::carbon::Kernel;

  /**
   * @brief Build a grid-wide reduction @c Kernel over a single @p level of the tower, writing
   *        into @p dst .
   *
   * Same shape as @ref make_kernel but the per-cell callable returns the @c T contribution
   * @c (Span, Idx) -> T (cross-level reads through @c loader.read(field.level_view(level)) are
   * fair game in the contribution — e.g. reduce a residual built from a voxel and its parent).
   * Forwards to @c grid(level).make_reduce_kernel ; the per-brick stage-1 reduce, the
   * intra-process combine (stage 2) and the cross-rank @c ncclAllReduce (stage 3) are all
   * inherited from the Block reducer, so multi-GPU + one-rank-per-GPU work unchanged.
   *
   * @tparam T   Reduction value type returned by the per-cell callable.
   * @param level       Level to reduce over (0 = finest). @c [0, depth()) .
   * @param loading_fn  Loading callable @c (KernelLoader&) -> F , @c F a @c (Span, Idx) -> T .
   * @param dst         Destination held by reference for the kernel's lifetime; must have been
   *                    built on this @c System ( @c dst.gpu_count() == @c gpu_count() ).
   * @param op          Reduction operator (default @c reduce::Sum<T> ).
   * @param shared_mem  Dynamic shared memory bytes per launch (default 0).
   *
   * @throws std::out_of_range when @p level is outside @c [0, depth()) .
   */
  template <typename T,
            typename L,
            typename Op = ::carbon::reduce::Sum<T>>
  [[nodiscard]] auto
  make_reduce_kernel(int level,
                     L&& loading_fn,
                     ::carbon::ReductionResult<T>& dst,
                     Op op = Op{},
                     size_t shared_mem = 0U) const -> ::carbon::Kernel;

  /**
   * Host (OpenMP) twin of @ref make_reduce_kernel — forwards to
   * @c grid(level).make_host_reduce_kernel .
   */
  template <typename T,
            typename L,
            typename Op = ::carbon::reduce::Sum<T>>
  [[nodiscard]] auto
  make_host_reduce_kernel(int level,
                          L&& loading_fn,
                          ::carbon::ReductionResult<T>& dst,
                          Op op = Op{}) const -> ::carbon::Kernel;

private:
  System system_{};
  int64_3d finest_extent_{};
  Descriptor descriptor_{};
  std::vector<BlockGrid> levels_{};
  // Device tables indexed [level]; each Buffer holds one allocation per GPU (so dev_[l][g] is
  // GPU g's table). Host mirrors indexed [level][gpu]. Parent meaningful for level in
  // [0, depth-1), child for [1, depth).
  std::vector<Buffer<uint32_t>> parent_block_id_dev_{};
  std::vector<Buffer<uint32_t>> child_block_id_dev_{};
  std::vector<std::vector<std::vector<uint32_t>>> parent_block_id_host_{};
  std::vector<std::vector<std::vector<uint32_t>>> child_block_id_host_{};
  std::string name_{};
};

}  // namespace carbon

#include <carbon/mgrid.inl>
