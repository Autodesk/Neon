#pragma once

// Public API for the carbon::Block (block-sparse) grid family.
//
// Multi-GPU + multi-rank: ships the full block-discovery / bitmask / 27-way
// connectivity machinery, multi-GPU partitioning along the Z block axis, the
// Internal/Boundary DataView split, ghost-brick halo updates via HaloBlock
// (intra-process via Transfer<T> and cross-rank via DistTransfer<T>), and the
// polymorphic Kernel handle (compute / halo / reduce / universal). Mirrors the
// cadence carbon::Dense and carbon::DenseDisg used (data plane first, then halo,
// then kernels) and reaches the same feature-completeness today.
//
// See the bGrid analysis report bundled into the 2026-05 CHANGELOG entry for
// the design brief and the Neon parallels (StaticBlock / bSpan / bPartition /
// bField).

#include <carbon/detail/block/field.h>
#include <carbon/detail/block/idx.h>
#include <carbon/detail/block/locator.h>
#include <carbon/detail/block/partition.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/block/static_block.h>
#include <carbon/export.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/system.h>

#include <stdint.h>

#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon {

/**
 * Block-sparse multi-GPU grid (port of Neon's @c bGrid). Both build-out phases
 * have landed: multi-GPU brick partitioning (phase A) and ghost bricks +
 * @c HaloBlock + the Internal/Boundary DataView split (phase B).
 *
 * Models a bounded Cartesian voxel domain of size @c domain_extent , chunked into
 * bricks of size @c SBlock::MemX × MemY × MemZ . Only bricks whose intersection with
 * the user predicate is non-empty are retained; per-voxel activity inside a retained
 * brick is tracked by an @c SBlock::BitMask sitting next to the field storage.
 * Inactive voxels are still allocated (the brick is dense in memory) but skipped at
 * iteration time via @ref Span::set_and_validate so they never call into the user
 * kernel.
 *
 * @par Multi-GPU partitioning
 *
 * The brick @b Z axis (i.e. the candidate-brick lattice's z-extent, expressed in
 * brick rows of thickness @c MemZ ) is split evenly across @c system.gpu_count()
 * devices. Each GPU owns a contiguous range of brick rows; bricks whose @c bz falls
 * in that range are retained on that GPU only. Brick storage is per-GPU SoA: each
 * device's @c device_buffer contains @c block_count_g · VoxelCount · Components
 * elements. The 27-way connectivity table proper resolves neighbours @b on the same
 * GPU; cross-slab neighbours resolve through the ghost-brick slots reserved at each
 * slab's lower/upper z-edge (see @ref ghost_lower_count / @ref ghost_upper_count )
 * once a @c HaloBlock<T, Components, SBlock> update has copied the adjacent slab's
 * boundary bricks into them — connectivity entries on boundary owned bricks point
 * into the ghost id range so @c Partition::neighbour resolves across slabs.
 *
 * @par What this implements
 *
 * - @ref Idx , @ref Span , @ref Partition , @ref Field accessible through the standard
 *   @c carbon::Block::Idx / @c Block::Span / @c Block::Partition<T, C> /
 *   @c Block::Field<T, C> aliases (matching the @c carbon::Dense surface).
 * - Construction takes a @c System (any GPU count), the global @c domain_extent in
 *   voxels, and an @c is_active host predicate. The ctor walks the brick lattice on
 *   the host, partitions retained bricks by their @c bz onto the per-GPU slabs,
 *   builds the per-block bitmask and 27-way intra-GPU connectivity, allocates the
 *   matching device buffers per GPU, and uploads.
 * - @ref newField allocates per-GPU SoA device buffers and binds a @ref Field whose
 *   per-GPU @ref Partition reads through @c (block_idx · C + c) · VoxelCount + voxel
 *   pitch, and whose @c neighbour walks the device-side bitmask + connectivity table.
 * - @ref make_kernel / @c Kernel::compute loops over GPUs and launches the
 *   block-sparse compute kernel on each device's stream, mirroring
 *   @c Dense::make_kernel 's per-GPU loop.
 * - Ghost bricks + @c HaloBlock<T, Components, SBlock> . Each GPU reserves
 *   "remote-mirror" slots at the upper / lower z-edges (@ref ghost_lower_count /
 *   @ref ghost_upper_count ); connectivity entries on boundary owned bricks point
 *   into the ghost id range so @c Partition::neighbour resolves across slabs after a
 *   halo update. @ref Field::make_halo_kernel builds the @c HaloBlock -backed
 *   @c Kernel (intra-process @c Transfer<T> + cross-rank @c DistTransfer<T> ).
 * - DataView Internal / Boundary split. Owned bricks are classified into internal
 *   (@c [0, n_internal_blocks(gpu_id)) ) and boundary subsets (same shape Neon's
 *   @c Partitioner1D produces), so @c DataView::Internal + @c DataView::Boundary
 *   recombine cell-for-cell to @c DataView::Standard and an OCC-overlapped
 *   halo / Internal / Boundary pipeline computes the Standard result.
 *
 * @par Out of scope (explicit follow-ups)
 *
 * - @c Block native binary I/O ( @c carbon::io::write_native / read_native ): the
 *   brick-list layout needs its own on-disk format (the Cartesian SoA path only
 *   covers @c Dense / @c DenseDisg today).
 *
 * @tparam SBlock @c StaticBlock policy controlling brick size and bitmask layout.
 *                Defaults to @c DefaultStaticBlock = @c StaticBlock<8, 8, 8> (matches
 *                Neon's only explicitly-instantiated brick size).
 */
template <typename SBlock = detail::block::DefaultStaticBlock>
class Block {
public:
  using BlockPolicy = SBlock;
  using BitMask = typename SBlock::BitMask;
  using Idx = detail::block::Idx;
  using Span = detail::block::Span<SBlock>;

  template <typename T, int Components>
  using Partition = detail::block::Partition<T, Components, SBlock>;

  template <typename T, int Components>
  using Field = detail::block::Field<T, Components, SBlock>;

  /** Predicate type accepted by the constructor. */
  using ActiveCellPredicate = std::function<bool(const int64_3d&)>;

  /**
   * @brief Explicit global brick-z partition: per global slab @c [begin, end) brick rows.
   *
   * When handed to the advanced @ref Block ctor overload, this overrides the default even
   * z-split. @c begin and @c end must each hold exactly @c global_slab_count_ entries
   * (@c gpu_count() in single-process mode, @c world_size under @c CARBON_HAVE_MPI_NCCL ),
   * be contiguous (@c begin[0] == 0 , @c begin[s] == end[s-1] ), have @c begin[s] < end[s]
   * (every slab owns at least one brick row), and cover the candidate z-extent
   * (@c end.back() == candidate_block_extent().z ). An empty partition (default-constructed)
   * means "use the even split" so the ordinary ctor delegates here with no behavioural change.
   *
   * @c MultiResBlock uses this to project the coarsest level's z-cut down to every finer
   * level so a parent brick and all its descendants stay co-resident on one GPU/rank — the
   * load-bearing rule that keeps cross-level access Transfer-free (see
   * @c docs/design_mgrid.md §4).
   */
  struct ZBrickPartition {
    std::vector<int64_t> begin{};
    std::vector<int64_t> end{};
  };

  static constexpr int MemX = SBlock::MemX;
  static constexpr int MemY = SBlock::MemY;
  static constexpr int MemZ = SBlock::MemZ;
  static constexpr int VoxelCount = SBlock::VoxelCount;

  /**
   * Build a multi-GPU block-sparse grid over @p system 's GPUs. The brick @b Z axis
   * is split evenly: GPU @c g owns brick rows
   * @c [g · candidate_z / N + min(g, candidate_z % N) ,
   * @c (g+1) · candidate_z / N + min(g+1, candidate_z % N))
   * (same shape as Neon's @c Partitioner1D for the @c d1b3 family).
   *
   * @param system        Backing System (any GPU count).
   * @param domain_extent Cartesian extent in voxels. Components must all be positive.
   *                      The brick lattice is @c ceil_div(domain_extent, brick) .
   * @param is_active     Host predicate called once per voxel @c (gx, gy, gz) in
   *                      @c [0, domain_extent.x) × ... A brick is retained iff at
   *                      least one of its voxels passes the predicate.
   * @param name          Optional debug label propagated to logging.
   *
   * @throws std::invalid_argument when @p system holds zero GPUs, when any
   *         @p domain_extent component is non-positive, or when the candidate brick
   *         z-extent is smaller than @c gpu_count (each GPU must own at least one
   *         brick row, so very small domains under multi-GPU need a coarser
   *         brick-z extent or a single-GPU @c System ).
   */
  Block(System system,
        int64_3d domain_extent,
        const ActiveCellPredicate& is_active,
        std::string name = {});

  /**
   * Advanced ctor: build with an explicit global brick-z partition @p z_partition instead of
   * the default even z-split. All other parameters behave exactly as the primary ctor. An
   * empty @p z_partition is equivalent to the primary ctor (even split). See @ref ZBrickPartition
   * for the validity contract; @c MultiResBlock uses this to project a shared cut across levels.
   *
   * @throws std::invalid_argument on the same conditions as the primary ctor, plus when
   *         @p z_partition is non-empty and violates the @ref ZBrickPartition contract.
   */
  Block(System system,
        int64_3d domain_extent,
        const ActiveCellPredicate& is_active,
        ZBrickPartition z_partition,
        std::string name = {});

  Block() = default;

  [[nodiscard]] auto
  domain_extent() const noexcept -> const int64_3d& {
    return domain_extent_;
  }

  /**
   * Cartesian voxel-domain extent — alias for @ref domain_extent for cross-grid API
   * parity with @c Dense::global_extent / @c DenseDisg::global_extent . The returned
   * triple is the voxel-aligned bounding box used by the predicate, before brick
   * retention; on @c Block this is the only "global extent" notion (the brick lattice
   * is @ref candidate_block_extent ).
   */
  [[nodiscard]] auto
  global_extent() const noexcept -> const int64_3d& {
    return domain_extent_;
  }

  /**
   * Number of bricks in each axis of the @b candidate brick lattice (i.e. before
   * predicate filtering). Always @c ceil_div(domain_extent, brick_size) .
   */
  [[nodiscard]] auto
  candidate_block_extent() const noexcept -> const int64_3d& {
    return candidate_extent_;
  }

  /** Total retained blocks across every GPU. */
  [[nodiscard]] auto
  block_count() const noexcept -> uint32_t {
    return total_block_count_;
  }

  /**
   * Number of retained blocks owned by @p gpu_id . Defaults to GPU 0 so
   * @c block.block_count() in single-GPU code stays meaningful.
   */
  [[nodiscard]] auto
  block_count(size_t gpu_id) const noexcept -> uint32_t {
    if (gpu_id >= host_origins_per_gpu_.size()) {
      return 0U;
    }
    return static_cast<uint32_t>(host_origins_per_gpu_[gpu_id].size());
  }

  /** Number of GPUs this @c Block is partitioned across. */
  [[nodiscard]] auto
  gpu_count() const noexcept -> size_t {
    return host_origins_per_gpu_.size();
  }

  /** Total active voxels across all retained blocks (every GPU). */
  [[nodiscard]] auto
  active_voxel_count() const noexcept -> int64_t {
    return active_voxel_count_;
  }

  [[nodiscard]] auto
  system() noexcept -> System& {
    return system_;
  }

  [[nodiscard]] auto
  system() const noexcept -> const System& {
    return system_;
  }

  /** Per-GPU spans (size = @ref gpu_count ). */
  [[nodiscard]] auto
  spans() const noexcept -> const std::vector<Span>& {
    return spans_;
  }

  /**
   * Per-GPU @b host spans: one @c Span per local GPU whose @c bitmask() and
   * @c connectivity() pointers point at the @b host-side mirror tables (
   * @c host_bitmasks(g) and @c host_connectivity(g) data pointers, both
   * already populated at @c Block construction). Same @c block_count and
   * @c n_internal as the device spans -- only the topology pointers differ.
   *
   * Used by @c Block::make_host_kernel / @c make_host_reduce_kernel : an
   * OpenMP loop dereferencing the device @c bitmask() / @c connectivity()
   * pointers from a host TU would segfault, so host launches construct a
   * host span and feed it to the user's per-cell callable instead.
   *
   * Vector is memoised on first call. Returns an empty vector when the grid
   * has no GPUs.
   */
  [[nodiscard]] auto
  host_spans() const -> const std::vector<Span>& {
    if (!host_spans_.empty() || host_bitmasks_per_gpu_.empty()) {
      return host_spans_;
    }
    host_spans_.reserve(host_bitmasks_per_gpu_.size());
    for (size_t g = 0; g < host_bitmasks_per_gpu_.size(); ++g) {
      const auto& bm = host_bitmasks_per_gpu_[g];
      const auto& conn = host_connectivity_per_gpu_[g];
      const uint32_t n_blocks = static_cast<uint32_t>(bm.size());
      const uint32_t n_internal =
          (g < n_internal_per_gpu_.size()) ? n_internal_per_gpu_[g] : n_blocks;
      const int32_3d* origins =
          host_origins_per_gpu_[g].empty() ? nullptr : host_origins_per_gpu_[g].data();
      host_spans_.emplace_back(bm.data(), n_blocks, n_internal, conn.data(), origins);
    }
    return host_spans_;
  }

  /**
   * Span for @p gpu_id (defaults to GPU 0 for single-GPU shorthand). Carries the
   * pointer math @ref Field 's @c device_partition() needs to walk per-voxel; safe
   * to copy into a device kernel by value.
   *
   * Throws @c std::out_of_range when @p gpu_id is past @c gpu_count() . The previous
   * silent-empty-sentinel behaviour was a footgun-class shape: a stale loop bound
   * surfaced as an empty span downstream rather than a clear precondition failure.
   */
  [[nodiscard]] auto
  span(size_t gpu_id = 0) const -> const Span& {
    if (gpu_id >= spans_.size()) {
      log::throw_with_log<std::out_of_range>(
          "carbon::Block::span: gpu_id out of range (gpu_id={}, gpu_count={}).", gpu_id,
          spans_.size());
    }
    return spans_[gpu_id];
  }

  /**
   * Host-side per-block voxel origins on @p gpu_id (size = @c block_count(gpu_id) ).
   *
   * Throws @c std::out_of_range when @p gpu_id is past @c gpu_count() .
   */
  [[nodiscard]] auto
  host_origins(size_t gpu_id = 0) const -> const std::vector<int32_3d>& {
    if (gpu_id >= host_origins_per_gpu_.size()) {
      log::throw_with_log<std::out_of_range>(
          "carbon::Block::host_origins: gpu_id out of range (gpu_id={}, gpu_count={}).", gpu_id,
          host_origins_per_gpu_.size());
    }
    return host_origins_per_gpu_[gpu_id];
  }

  /**
   * Device-side per-block voxel origins for @p gpu_id . One @c int32_3d per retained
   * brick on that GPU, in the same order as @c host_origins(gpu_id) . User kernels
   * capture this pointer to recover global voxel coordinates from a kernel-side
   * @c Idx :
   * @code
   * const int32_3d origin = origins[idx.block_idx()];
   * const int32_t gx = origin.x + idx.intra().x;
   * @endcode
   * Returns @c nullptr when @c block_count(gpu_id) is zero.
   */
  [[nodiscard]] auto
  device_origins_ptr(size_t gpu_id = 0) const noexcept -> const int32_3d* {
    if (gpu_id >= device_origins_.size() || device_origins_[gpu_id].size() == 0U) {
      return nullptr;
    }
    return device_origins_[gpu_id].data();
  }

  /**
   * Device-visible brick-index volume for @p gpu_id — the inverse of
   * @c Span::global_location : it answers "which cell contains this point", which the
   * flat @c origins list alone cannot do in better than @c O(block_count) .
   * See @c detail/block/locator.h for the memory cost and the ownership contract.
   *
   * Built on first call and cached for the lifetime of the grid (and shared by every
   * copy of this @c Block ), so a run that never renders never pays for it. The table
   * is pure topology; @p world_origin and @p world_spacing only stamp the returned view,
   * so different world frames over the same grid reuse one table.
   *
   * @param gpu_id        Relative GPU id, as everywhere else on @c Block .
   * @param world_origin  World position of the level's voxel @c (0, 0, 0) corner.
   * @param world_spacing World size of one voxel at this level ( @c R^level under mGrid ).
   *
   * A GPU whose slab retained no brick yields a view with @c valid() == false rather than
   * an error, so per-GPU render loops need no special case.
   */
  [[nodiscard]] auto
  locator(size_t gpu_id = 0,
          float32_3d world_origin = {},
          float world_spacing = 1.0F) const -> detail::block::Locator<SBlock>;

  /**
   * Host-side per-block bitmask copy on @p gpu_id (size = @c block_count(gpu_id) ).
   *
   * Throws @c std::out_of_range when @p gpu_id is past @c gpu_count() .
   */
  [[nodiscard]] auto
  host_bitmasks(size_t gpu_id = 0) const -> const std::vector<BitMask>& {
    if (gpu_id >= host_bitmasks_per_gpu_.size()) {
      log::throw_with_log<std::out_of_range>(
          "carbon::Block::host_bitmasks: gpu_id out of range (gpu_id={}, gpu_count={}).", gpu_id,
          host_bitmasks_per_gpu_.size());
    }
    return host_bitmasks_per_gpu_[gpu_id];
  }

  /**
   * Host-side flattened 27-way block connectivity for @p gpu_id (size =
   * @c block_count(gpu_id) · 27). Slot @c (dz+1)·9 + (dy+1)·3 + (dx+1) of block
   * @c b stores the local @c block_idx of the brick at
   * @c host_origins(gpu_id)[b] + (dx, dy, dz) · brick_size on @b the same GPU, or
 * @c kInvalidBlockIdx when that neighbour brick was not retained or lives on a
 * different GPU. (Cross-slab neighbours resolve at the @b device partition level
 * through the ghost-brick slots after a @c HaloBlock update; this host-side table
 * is the raw same-GPU connectivity captured at construction.)
   *
   * Throws @c std::out_of_range when @p gpu_id is past @c gpu_count() .
   */
  [[nodiscard]] auto
  host_connectivity(size_t gpu_id = 0) const -> const std::vector<uint32_t>& {
    if (gpu_id >= host_connectivity_per_gpu_.size()) {
      log::throw_with_log<std::out_of_range>(
          "carbon::Block::host_connectivity: gpu_id out of range (gpu_id={}, gpu_count={}).",
          gpu_id, host_connectivity_per_gpu_.size());
    }
    return host_connectivity_per_gpu_[gpu_id];
  }

  /**
   * Number of @b internal blocks on @p gpu_id 's slab — owned bricks that do not
   * touch a slab face with a remote neighbour, so a stencil read of radius 1
   * stays entirely within this slab's owned bricks. Internal bricks live in
   * @c [0, n_internal_blocks(gpu_id)) of every per-slab list (origins,
   * bitmasks, partition view); boundary bricks live in
   * @c [n_internal_blocks(gpu_id), block_count(gpu_id)) . With a single global
   * slab the partition collapses to "all internal".
   *
   * The classification is a brick-level analogue of @c Dense::Span 's
   * halo_z-deep Internal / Boundary bands, sized to match the existing one-
   * brick-row-deep ghost reservation. Iterating the @c DataView::Internal
   * subset of a kernel skips the boundary bricks; iterating
   * @c DataView::Boundary picks them up. Together they cover the same set of
   * cells as @c DataView::Standard , so a halo-exchange + Internal-launch +
   * Boundary-launch pipeline (the @c OverlappedStencil shape on @c Dense )
   * computes the same result as a Standard-only launch of the same kernel.
   */
  [[nodiscard]] auto
  n_internal_blocks(size_t gpu_id) const noexcept -> uint32_t {
    if (gpu_id >= n_internal_per_gpu_.size()) {
      return 0U;
    }
    return n_internal_per_gpu_[gpu_id];
  }

  /**
   * Number of @b boundary blocks on @p gpu_id 's slab — owned bricks at
   * @c [n_internal_blocks(gpu_id), block_count(gpu_id)) . Always zero in
   * single-global-slab layouts (no remote neighbour to halo against).
   */
  [[nodiscard]] auto
  n_boundary_blocks(size_t gpu_id) const noexcept -> uint32_t {
    const uint32_t total = block_count(gpu_id);
    const uint32_t internal = n_internal_blocks(gpu_id);
    return (total > internal) ? (total - internal) : 0U;
  }

  /** First brick row owned by @p gpu_id in the candidate brick lattice. */
  [[nodiscard]] auto
  bz_begin(size_t gpu_id) const noexcept -> int64_t {
    if (gpu_id >= bz_begin_per_gpu_.size()) {
      return 0;
    }
    return bz_begin_per_gpu_[gpu_id];
  }

  /** One past the last brick row owned by @p gpu_id in the candidate brick lattice. */
  [[nodiscard]] auto
  bz_end(size_t gpu_id) const noexcept -> int64_t {
    if (gpu_id >= bz_end_per_gpu_.size()) {
      return 0;
    }
    return bz_end_per_gpu_[gpu_id];
  }

  /**
   * Number of @b global slabs in the brick z-axis partition. In single-process
   * mode this equals @c gpu_count() ; in distributed mode (one-rank-per-GPU under
   * @c CARBON_HAVE_MPI_NCCL ) this equals @c world_size and each rank's local
   * @c gpu_count() is 1.
   */
  [[nodiscard]] auto
  global_slab_count() const noexcept -> size_t {
    return global_slab_count_;
  }

  /**
   * Index of this rank's first local slab inside the global brick z-axis
   * partition. Always 0 in single-process mode; equals @c world_rank in the
   * distributed one-rank-per-GPU case.
   */
  [[nodiscard]] auto
  local_slab_offset() const noexcept -> size_t {
    return local_slab_offset_;
  }

  /**
   * Number of @b ghost slots reserved at the lower (-z) edge of @p gpu_id 's slab.
   * Lower ghost slots mirror retained bricks at @c bz_begin(gpu_id) - 1 owned by
   * @c gpu_id - 1 ; cross-slab connectivity from owned bricks at @c bz_begin(gpu_id)
   * is rewritten to point into this range so @c Partition::neighbour resolves after
   * a @c HaloBlock update.
   */
  [[nodiscard]] auto
  ghost_lower_count(size_t gpu_id) const noexcept -> uint32_t {
    if (gpu_id >= ghost_lower_count_per_gpu_.size()) {
      return 0U;
    }
    return ghost_lower_count_per_gpu_[gpu_id];
  }

  /**
   * Number of @b ghost slots reserved at the upper (+z) edge of @p gpu_id 's slab.
   * Upper ghost slots mirror retained bricks at @c bz_end(gpu_id) owned by
   * @c gpu_id + 1 .
   */
  [[nodiscard]] auto
  ghost_upper_count(size_t gpu_id) const noexcept -> uint32_t {
    if (gpu_id >= ghost_upper_count_per_gpu_.size()) {
      return 0U;
    }
    return ghost_upper_count_per_gpu_[gpu_id];
  }

  /**
   * Total slot count on @p gpu_id : owned bricks plus lower- and upper-ghost slots.
   * @c Field 's per-GPU device buffer is sized for this many bricks; kernels still
   * only iterate over @c block_count(gpu_id) owned slots (ghosts are storage-only,
   * read through @c neighbour ).
   */
  [[nodiscard]] auto
  storage_block_count(size_t gpu_id) const noexcept -> uint32_t {
    return block_count(gpu_id) + ghost_lower_count(gpu_id) + ghost_upper_count(gpu_id);
  }

  /**
   * @brief One halo-update transfer descriptor.
   *
   * Copies a single brick worth of field data (i.e.
   * @c VoxelCount · Components · sizeof(T) bytes) from a source owned brick on
   * one GPU to a destination ghost slot on the adjacent GPU. @c HaloBlock walks
   * the per-pair upward / downward vectors and instantiates a @c Transfer<T>
   * per entry per component-fused brick.
   *
   * Used for the @b intra-process side of the halo plan (single-rank
   * multi-GPU). The cross-rank side under @c CARBON_HAVE_MPI_NCCL is captured
   * by the @c halo_dist_* accessors instead, since a cross-rank entry only
   * carries one local id (this rank's owned brick for Send, this rank's ghost
   * slot for Recv) and the peer rank — the matching id on the remote rank is
   * pinned by positional NCCL pairing.
   */
  struct HaloPair {
    uint32_t src_local_id;  ///< Owned brick local id on the source GPU.
    uint32_t dst_local_id;  ///< Ghost slot local id on the destination GPU.
  };

  /**
   * @brief One cross-rank halo entry from this rank's perspective.
   *
   * Distinct from @c HaloPair because cross-rank entries are one-sided: each
   * rank only sees its own local id and the peer rank that posts the
   * corresponding @c Recv (for @c Send) or @c Send (for @c Recv ). NCCL
   * pairs entries positionally (same FIFO order on both sides), so the
   * remote local id is implicit.
   */
  struct HaloDistEntry {
    uint32_t local_id;  ///< Owned brick (Send) or ghost slot (Recv) on this rank.
    int peer_rank;      ///< Peer rank for the @c ncclSend / @c ncclRecv .
  };

  /**
   * Halo pairs that send data @b upward across adjacency @p adj_idx (i.e. from
   * GPU @c adj_idx to GPU @c adj_idx + 1 ). @p adj_idx must be in
   * @c [0, gpu_count() - 1) ; throws @c std::out_of_range otherwise. Each entry
   * encodes the source brick on @c adj_idx (owned at @c bz_end(adj_idx) - 1 ) and
   * its matching @b lower-ghost slot on @c adj_idx + 1 .
   */
  [[nodiscard]] auto
  halo_pairs_upward(size_t adj_idx) const -> const std::vector<HaloPair>& {
    if (adj_idx >= halo_pairs_upward_.size()) {
      log::throw_with_log<std::out_of_range>(
          "carbon::Block::halo_pairs_upward: adj_idx out of range (adj_idx={}, "
          "adjacency_count={}).",
          adj_idx, halo_pairs_upward_.size());
    }
    return halo_pairs_upward_[adj_idx];
  }

  /**
   * Halo pairs that send data @b downward across adjacency @p adj_idx (i.e. from
   * GPU @c adj_idx + 1 to GPU @c adj_idx ). Each entry encodes the source brick
   * on @c adj_idx + 1 (owned at @c bz_begin(adj_idx + 1) ) and its matching
   * @b upper-ghost slot on @c adj_idx .
   *
   * Throws @c std::out_of_range when @p adj_idx is past the adjacency count
   * (@c gpu_count() - 1 ).
   */
  [[nodiscard]] auto
  halo_pairs_downward(size_t adj_idx) const -> const std::vector<HaloPair>& {
    if (adj_idx >= halo_pairs_downward_.size()) {
      log::throw_with_log<std::out_of_range>(
          "carbon::Block::halo_pairs_downward: adj_idx out of range (adj_idx={}, "
          "adjacency_count={}).",
          adj_idx, halo_pairs_downward_.size());
    }
    return halo_pairs_downward_[adj_idx];
  }

  /**
   * Cross-rank @c Send entries that ship this rank's owned upper-edge bricks
   * @b upward to the peer rank. Empty in single-process mode and on the
   * world-rank-N-1 side (no upward neighbour). Iteration order matches the
   * matching @c halo_dist_recv_upward call on the receiving rank, so NCCL
   * pairs Sends and Recvs positionally.
   */
  [[nodiscard]] auto
  halo_dist_send_upward() const noexcept -> const std::vector<HaloDistEntry>& {
    return halo_dist_send_upward_;
  }

  /**
   * Cross-rank @c Recv entries that fill this rank's lower-ghost slots from
   * data sent @b upward by the peer rank below us. Empty on rank 0.
   */
  [[nodiscard]] auto
  halo_dist_recv_upward() const noexcept -> const std::vector<HaloDistEntry>& {
    return halo_dist_recv_upward_;
  }

  /**
   * Cross-rank @c Send entries that ship this rank's owned lower-edge bricks
   * @b downward to the peer rank below us. Empty on rank 0 and in
   * single-process mode.
   */
  [[nodiscard]] auto
  halo_dist_send_downward() const noexcept -> const std::vector<HaloDistEntry>& {
    return halo_dist_send_downward_;
  }

  /**
   * Cross-rank @c Recv entries that fill this rank's upper-ghost slots from
   * data sent @b downward by the peer rank above us. Empty on the
   * world-rank-N-1 side.
   */
  [[nodiscard]] auto
  halo_dist_recv_downward() const noexcept -> const std::vector<HaloDistEntry>& {
    return halo_dist_recv_downward_;
  }

  /**
   * Allocate a @ref Field over this @c Block . Sizing per GPU is
   * @c block_count(g) · Components · VoxelCount elements on each device.
   *
   * @p side selects which sides of the field are reserved. @c MemorySide::Device
   * allocates only the device buffer. @c MemorySide::Host and
   * @c MemorySide::HostDevice eagerly allocate a per-GPU host mirror sized
   * identically to the device buffer, populating @ref Field::cpu_buffers and
   * @ref Field::host_partitions ; the device buffer is allocated in both
   * @c HostDevice and @c Host modes so a device-side launch over a
   * @c MemorySide::Host field is still valid (Carbon mirrors @c Dense::Field 's
   * symmetric ownership here).
   */
  template <typename T,
            int Components>
  [[nodiscard]] auto
  newField(MemorySide side = MemorySide::Device,
           std::string name = {}) const -> Field<T,
                                                 Components>;

  /**
   * Build a device compute @c Kernel from a Neon-style loading lambda. Mirrors
   * @c carbon::Dense::make_kernel ; see @c Kernel::compute for the launch shape
   * (one CUDA block per retained brick, brick-sized thread block, inactive voxels
   * skipped via the bitmask). @p shared_mem is forwarded to every per-GPU launch
   * (CUDA only; SYCL must use @c shared_mem == 0 in this pass).
   *
   * @note Must be called from a @c .cu translation unit (or any TU compiled with
   *       nvcc) under the CUDA backend; defined out-of-line in @c kernel.h .
   */
  template <typename L>
  [[nodiscard]] auto
  make_kernel(L&& loading_fn,
              size_t shared_mem = 0U) const -> ::carbon::Kernel;

  /**
   * Build a grid-wide reduction @c Kernel over this Block grid. Mirrors
   * @c Dense::make_reduce_kernel — same Neon-style loading lambda shape
   * (returns a per-voxel device callable invoked as
   * @c (Span, Idx) -> T ), same @c ReductionResult<T> destination, same
   * @ref DataView dispatch (Standard / Internal / Boundary; the Internal +
   * Boundary pair recombines to the Standard set of bricks).
   *
   * Per-brick stage 1 uses one CUDA block per retained brick with one
   * thread per voxel and a single @c cub::BlockReduce::Reduce; inactive
   * voxels (bitmask bit clear) contribute the reduction operator's
   * identity. Stages 2 (intra-process combine on the root GPU) and 3
   * (cross-rank @c ncclAllReduce ) are shared with the dense reducer.
   *
   * @note Must be called from a @c .cu translation unit (or any TU
   *       compiled with nvcc) under the CUDA backend, or from a regular
   *       @c .cpp TU under the SYCL backend; the body is defined
   *       out-of-line in @c kernel.h . The SYCL block reducer ships
   *       through @c detail/backend/sycl/block_reduce_kernel.h and uses
   *       @c sycl::reduction over a per-brick @c parallel_for ; stages
   *       2 and 3 are shared with the dense reducer regardless of
   *       backend.
   */
  template <typename T,
            typename L,
            typename Op = ::carbon::reduce::Sum<T>>
  [[nodiscard]] auto
  make_reduce_kernel(L&& loading_fn,
                     ::carbon::ReductionResult<T>& dst,
                     Op op = Op{},
                     size_t shared_mem = 0U) const -> ::carbon::Kernel;

  /**
   * Build a host (CPU / OpenMP) compute @c Kernel over this @c Block grid.
   * Mirrors @c Dense::make_host_kernel : the loading lambda is the same Neon-
   * style closure fed to @c make_kernel — the only difference is that
   * @c loader(field) routes to the per-GPU @b host mirror partitions and the
   * launch is a per-slab nested OpenMP loop (one block at a time, brick-shaped
   * collapse(3) on @c (iz, iy, ix) ) instead of a per-block CUDA / SYCL
   * launch.
   *
   * Every field touched by the lambda must have been built with
   * @c MemorySide::Host or @c MemorySide::HostDevice ; @c MemorySide::Device
   * fields trip the host loader's @c std::invalid_argument check the first
   * time @c loader(field) is called.
   *
   * Plain @c .cpp TUs work — the host kernel does not depend on @c __CUDACC__ .
   */
  template <typename L>
  [[nodiscard]] auto
  make_host_kernel(L&& loading_fn) const -> ::carbon::Kernel;

  /**
   * Build a host (CPU / OpenMP) grid-wide reduction @c Kernel over this @c Block
   * grid. Mirrors @c Dense::make_host_reduce_kernel : thread-local partials
   * combined sequentially through @p op , then folded across slabs and (when
   * @c System carries a @c Distributed handle) across ranks via
   * @c MPI_Allreduce . Result lands directly in @c dst.host_data(view) — no
   * device scalar touched.
   *
   * Plain @c .cpp TUs work; no CUDA / SYCL toolchain required.
   */
  template <typename T,
            typename L,
            typename Op = ::carbon::reduce::Sum<T>>
  [[nodiscard]] auto
  make_host_reduce_kernel(L&& loading_fn,
                          ::carbon::ReductionResult<T>& dst,
                          Op op = Op{}) const -> ::carbon::Kernel;

  /**
   * Build a dual-target (universal) compute @c Kernel over this @c Block grid.
   * Mirrors @c Dense::make_universal_kernel : one factory call instantiates
   * both the device launch path (CUDA / SYCL, one block per retained brick) and
   * the host launch path (per-slab OpenMP loop over brick voxels) from the same
   * loading lambda. The polymorphic @c Kernel::launch defaults to the device
   * path so a Universal Block kernel stored in @c std::vector<Kernel>  behaves
   * the same as a @c make_kernel handle when iterated through
   * @c launch(stream) ; users opt into the host path by calling
   * @c Kernel::launch_on_host(stream) on the same handle.
   *
   * Every field touched on the host launch must have been built with
   * @c MemorySide::Host or @c MemorySide::HostDevice (same contract as
   * @ref make_host_kernel ).
   *
   * @note Must be called from a @c .cu translation unit under the CUDA backend
   *       (the device launch body is gated on @c __CUDACC__ ). Plain @c .cpp
   *       TUs work under the SYCL backend.
   */
  template <typename L>
  [[nodiscard]] auto
  make_universal_kernel(L&& loading_fn,
                        size_t shared_mem = 0U) const -> ::carbon::Kernel;

  /**
   * Build a dual-target (universal) grid-wide reduction @c Kernel over this
   * @c Block grid. Mirrors @c Dense::make_universal_reduce_kernel : the device
   * path goes through the same fused per-brick map + reduce + stage 2 / 3
   * combine pipeline as @ref make_reduce_kernel , and the host path goes
   * through the same OpenMP thread-local partials + sequential combine
   * pipeline as @ref make_host_reduce_kernel . Default @c launch runs the
   * device path; @c Kernel::launch_on_host runs the host path. The destination
   * @c ReductionResult is bound by reference at factory time and must outlive
   * the returned @c Kernel handle.
   *
   * @note Must be called from a @c .cu translation unit under the CUDA backend
   *       (the device launch body is gated on @c __CUDACC__ ). Plain @c .cpp
   *       TUs work under the SYCL backend.
   */
  template <typename T,
            typename L,
            typename Op = ::carbon::reduce::Sum<T>>
  [[nodiscard]] auto
  make_universal_reduce_kernel(L&& loading_fn,
                               ::carbon::ReductionResult<T>& dst,
                               Op op = Op{},
                               size_t shared_mem = 0U) const -> ::carbon::Kernel;

  /**
   * Asynchronously copy each local GPU's @b owned-brick range from @p field
   * into newly allocated host buffers, returning one host @c Allocation per
   * local GPU (vector size == @c gpu_count() ).
   *
   * Each per-GPU result has @c block_count(g) · @c VoxelCount · @c Components
   * elements, indexed in the same SoA-per-block layout the device side uses
   * (block-major, component-major within a block, then z-y-x intra-brick).
   * Ghost slots are intentionally @b not downloaded — host-side processing
   * usually wants the owned, locally meaningful subset (RMS, I/O staging,
   * centerline samples, etc.).
   *
   * Copies are queued on @p stream and the function does @b not call
   * @c stream.wait() — the caller is responsible for the sync step before
   * touching the returned host data. This matches the convention used by the
   * @c Transfer / @c System::async_copy primitives.
   */
  template <typename T,
            int Components>
  [[nodiscard]] auto
  download_to_host(const Field<T,
                               Components>& field,
                   const ::carbon::Stream& stream) const -> std::vector<detail::Allocation<T>>;

private:
  /**
   * Inter-phase scratch state used to break the @c Block ctor into a sequence
   * of private member helpers (see @ref initialise_partition_ ...
   * @ref upload_topology_and_build_spans_ ). Carries exactly the locals that
   * used to be captured by the ctor's nested lambdas; lives only for the
   * duration of construction and is then discarded.
   *
   * @c bz_*_global hold the global brick z-row partition (full @c global_slab_count_
   * entries) so cross-rank ownership lookups in @ref reserve_ghost_slots_ stay
   * scoped to the ctor — only the local slice gets persisted in
   * @c bz_*_per_gpu_ . @c candidate_to_local maps every candidate brick to
   * @c (gpu_id, local_id) on this rank (or @c kInvalidBlockIdx when not retained
   * here); used by the connectivity build + the intra/dist halo plans to look
   * up neighbour bricks. @c ghost_*_peer / @c ghost_*_remote_owned /
   * @c ghost_*_bitmasks are parallel per-GPU vectors describing each reserved
   * ghost slot's source (peer rank or local-slab id, source brick local id on
   * the source slab, source bitmask). @c candidate_to_ghost_local maps a
   * candidate brick to the local ghost-slot id reserved for it on a given GPU.
   */
  struct BuildContext {
    struct CandidateMap {
      uint32_t gpu_id{detail::block::kInvalidBlockIdx};
      uint32_t local_id{detail::block::kInvalidBlockIdx};
    };
    size_t n_gpus{0};
    int64_t total_candidates{0};
    // When non-null, initialise_partition_ uses this explicit global brick-z cut instead of
    // the default even split. Points at the ctor's by-value ZBrickPartition argument (alive for
    // the whole construction) and is discarded at ctor exit. nullptr => even split.
    const ZBrickPartition* explicit_z_partition{nullptr};
    std::vector<int64_t> bz_begin_global{};
    std::vector<int64_t> bz_end_global{};
    std::vector<CandidateMap> candidate_to_local{};
    std::vector<std::vector<int>> ghost_lower_peer{};
    std::vector<std::vector<int>> ghost_upper_peer{};
    std::vector<std::vector<uint32_t>> ghost_lower_remote_owned{};
    std::vector<std::vector<uint32_t>> ghost_upper_remote_owned{};
    std::vector<std::vector<BitMask>> ghost_lower_bitmasks{};
    std::vector<std::vector<BitMask>> ghost_upper_bitmasks{};
    std::vector<std::vector<uint32_t>> candidate_to_ghost_local{};
  };

  /**
   * Validate the @c System / @p domain_extent invariants, compute
   * @c candidate_extent_ , resolve @c global_slab_count_ / @c local_slab_offset_
   * (honouring the @c CARBON_HAVE_MPI_NCCL one-rank-per-GPU contract), partition
   * the candidate brick z-axis into per-rank ranges, and reserve the per-GPU
   * host-side storage vectors. Populates @p ctx 's @c n_gpus ,
   * @c total_candidates , @c bz_begin_global , @c bz_end_global , and the
   * empty-but-reserved @c candidate_to_local table.
   */
  auto
  initialise_partition_(const int64_3d& domain_extent,
                        BuildContext& ctx) -> void;

  /**
   * Walk the candidate brick lattice over the z-row range owned by this rank.
   * For each retained brick, push its origin + bitmask into
   * @c host_origins_per_gpu_[g] / @c host_bitmasks_per_gpu_[g] , record the
   * (gpu_id, local_id) in @p ctx.candidate_to_local , and accumulate
   * @c active_voxel_count_ . Sets @c total_block_count_ at the end.
   */
  auto
  sweep_active_bricks_(const int64_3d& domain_extent,
                       const ActiveCellPredicate& is_active,
                       BuildContext& ctx) -> void;

  /**
   * Stable-partition each slab's owned bricks into
   * @c [internal, boundary) — boundary iff the brick's bz row touches a slab
   * face with a remote neighbour. Permutes @c host_origins_per_gpu_[i] /
   * @c host_bitmasks_per_gpu_[i] in place and remaps the @p ctx.candidate_to_local
   * entries on slab @c i so the new local ids are in effect for the
   * connectivity + halo passes that follow.
   */
  auto
  split_internal_boundary_(BuildContext& ctx) -> void;

  /**
   * Reserve ghost slots at the lower / upper z-edge of each local slab,
   * walking the candidate (bx, by) plane in row-major order so the per-slab
   * ghost-id assignment is deterministic across ranks (NCCL pairs Sends and
   * Recvs positionally). Populates @c ghost_lower_count_per_gpu_ /
   * @c ghost_upper_count_per_gpu_ on @c this and @p ctx.ghost_*_peer ,
   * @p ctx.ghost_*_remote_owned , @p ctx.ghost_*_bitmasks , and
   * @p ctx.candidate_to_ghost_local .
   */
  auto
  reserve_ghost_slots_(const int64_3d& domain_extent,
                       const ActiveCellPredicate& is_active,
                       BuildContext& ctx) -> void;

  /**
   * Build per-GPU 27-way connectivity over owned bricks. Cross-GPU neighbours
   * route to ghost slots reserved by @ref reserve_ghost_slots_ ; otherwise
   * stay invalid. Populates @c host_connectivity_per_gpu_ .
   */
  auto
  build_connectivity_(const BuildContext& ctx) -> void;

  /**
   * Compose the intra-process halo plan: one upward + one downward
   * @c HaloPair vector per adjacency. Walks @p ctx.ghost_*_remote_owned /
   * @p ctx.ghost_*_peer and emits a @c HaloPair entry for every ghost slot
   * whose source is on another @b local slab. Populates @c halo_pairs_upward_
   * and @c halo_pairs_downward_ .
   */
  auto
  build_intra_halo_plan_(const BuildContext& ctx) -> void;

  /**
   * Compose the cross-rank halo plan. Send entries are emitted by walking
   * owned bricks at this rank's local-slab edges that face a remote
   * neighbour; Recv entries are emitted by walking ghost slots with remote
   * sources (@c peer != -1 ). NCCL pairing across ranks is positional, so
   * both sides walk the same predicate over the same row in row-major order.
   * Populates the four @c halo_dist_send_/recv_upward_/downward_ vectors.
   */
  auto
  build_dist_halo_plan_(const BuildContext& ctx) -> void;

  /**
   * Allocate per-GPU device buffers (bitmask / connectivity / origins),
   * upload the host mirrors asynchronously, and build the per-GPU @c Span
   * vector. Short-circuits to all-empty placeholder spans when
   * @c total_block_count_ is zero so @c spans()[g] is always indexable.
   */
  auto
  upload_topology_and_build_spans_(const BuildContext& ctx) -> void;

  /**
   * Lazily-built brick-index tables behind @ref locator , held through a shared_ptr so
   * every copy of a @c Block shares one build (a @c Block is copied by value into kernel
   * impls, and @c BufferGroup is itself shallow-copy). @c std::once_flag because the
   * per-GPU loading lambdas that ask for a locator run under an @c omp parallel for .
   */
  struct LocatorStorage {
    std::once_flag once{};
    Buffer<uint32_t> table{};
    std::vector<detail::block::LocatorGeometry> geometry{};
  };

  /** Fill @c locator_storage_ ; called once through @c LocatorStorage::once . */
  auto
  build_locator_() const -> void;

  System system_{};
  int64_3d domain_extent_{};
  int64_3d candidate_extent_{};
  int64_t active_voxel_count_{0};
  uint32_t total_block_count_{0};

  std::vector<std::vector<int32_3d>> host_origins_per_gpu_{};
  std::vector<std::vector<BitMask>> host_bitmasks_per_gpu_{};
  std::vector<std::vector<uint32_t>> host_connectivity_per_gpu_{};

  Buffer<BitMask> device_bitmask_{};
  Buffer<uint32_t> device_connectivity_{};
  Buffer<int32_3d> device_origins_{};

  std::vector<Span> spans_{};
  // Memoised per-GPU @c Span values whose @c bitmask() / @c connectivity() pointers
  // point at the host mirror tables. Built on demand by @c host_spans() so the OpenMP
  // launch path can dereference topology metadata from a host TU.
  mutable std::vector<Span> host_spans_{};
  std::vector<int64_t> bz_begin_per_gpu_{};
  std::vector<int64_t> bz_end_per_gpu_{};
  std::vector<uint32_t> n_internal_per_gpu_{};
  std::vector<uint32_t> ghost_lower_count_per_gpu_{};
  std::vector<uint32_t> ghost_upper_count_per_gpu_{};
  std::vector<std::vector<HaloPair>> halo_pairs_upward_{};
  std::vector<std::vector<HaloPair>> halo_pairs_downward_{};
  std::vector<HaloDistEntry> halo_dist_send_upward_{};
  std::vector<HaloDistEntry> halo_dist_recv_upward_{};
  std::vector<HaloDistEntry> halo_dist_send_downward_{};
  std::vector<HaloDistEntry> halo_dist_recv_downward_{};
  size_t global_slab_count_{0};
  size_t local_slab_offset_{0};
  std::string name_{};
  std::shared_ptr<LocatorStorage> locator_storage_{std::make_shared<LocatorStorage>()};
};

}  // namespace carbon

namespace carbon {
class Kernel;  // forward; full definition pulled in by user-facing TUs via <carbon/kernel.h>
}

#include <carbon/block.inl>
