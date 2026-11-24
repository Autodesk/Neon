#pragma once

#include <carbon/boundary_mode_distributed.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense_disg_distributed/idx.h>
#include <carbon/detail/dense_disg_distributed/partition.h>
#include <carbon/detail/dense_disg_distributed/span.h>
#include <carbon/export.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>
#include <carbon/launch_shape.h>
#include <carbon/memory_side.h>
#include <carbon/process_grid.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace carbon {

class Kernel;

template <typename T>
class ReductionResult;

namespace reduce {
template <typename T>
struct Sum;
}  // namespace reduce

namespace detail::dense_disg_distributed {
template <typename T, int N>
class Field;
}  // namespace detail::dense_disg_distributed

/**
 * @b Disaggregated dense grid partitioned by a @ref ProcessGrid . Combines the construction and
 * accessor surface of @ref DenseDistributed (Slab / Pencil / Brick selection, per-axis
 * @ref BoundaryModeDistributed , @c Span::offset() / @c Span::halo() ) with the five-region
 * storage of @ref DenseDisg (lower ghost / down boundary / internal / up boundary / upper ghost,
 * card-major SoA inside each region). See @c carbon::detail::dense_disg_distributed::Span for
 * the storage map and pitch formula.
 *
 * @par Supported layouts
 *
 * - @ref DecompositionMode::Slab : each GPU/rank owns a z-slab with the full global @c x and
 *   @c y extent, split exactly like @ref DenseDisg (earlier slabs receive one extra layer when
 *   the global z is not divisible).
 * - @ref DecompositionMode::Pencil (and @c Custom grids with @c Px == 1 ): each GPU/rank owns a
 *   full-x, local-y, local-z chunk with a one-row y halo and a one-plane z halo, stored in the
 *   nested pencil map (see @c detail::dense_disg_distributed::Span ). A pencil grid whose world
 *   size leaves @c Py == 1 degenerates to slab storage unless @c y is periodic.
 * - @ref DecompositionMode::Brick (and @c Custom grids with @c Px > 1 ): each GPU/rank owns a
 *   local-x, local-y, local-z chunk with a one-cell halo on every axis, stored in the nested brick
 *   map (x faces y-fastest, internal box on a 128-byte-aligned pitch). The storage shape follows
 *   the factorized dims, so a @c Brick world that factorizes to @c Px == 1 (e.g. 4 ranks ->
 *   @c (1, 2, 2) ) uses pencil storage.
 *
 * @par Stencil radius and boundaries
 *
 * Ghost and boundary radius are pinned to 1. A stencil of radius 1 gives a one-cell halo on every
 * exchanged axis; radius 0 gives no halo (the ghosts are still reserved but never exchanged).
 * Any other radius is rejected. @ref BoundaryModeDistributed::PeriodicX is rejected (no x halo);
 * @c Open , @c PeriodicY and @c PeriodicZ are honoured end-to-end.
 *
 * @par Relationship to DenseDisg
 *
 * On slab layouts, for the same @c System , global extent and a radius-1 stencil, every per-GPU
 * @c Span matches @c DenseDisg 's (owned / haloed extent, z offset, storage size, pitch), so a
 * field filled through @c Field::fill_owned_host is bit-identical across the two families.
 *
 * @par Public surface
 *
 * - Per-slab geometry ( @ref spans ) and per- @c DataView launch shapes ( @ref gpu_grid /
 *   @ref gpu_block ), populated at construction.
 * - @ref newField allocates @c spans()[g].linear_storage_elements(C) elements per chunk
 *   ( @c (Lz + 2) · C · xy_pitch on slabs).
 * - The six kernel factories mirror @ref DenseDisg one-for-one.
 * - Halo exchange goes through @ref HaloDisgDistributed : contiguous face / edge (and, on brick,
 *   corner) messages ( @c Transfer intra-process, one @c DistTransfer NCCL group cross-rank),
 *   lattice run fusion under @c StencilSemantic::Lattice / @c LatticeDistributed .
 */
class CARBON_API DenseDisgDistributed {
public:
  /**
   * Marker consumed by @c detail::DeviceLaunchTraits / @c detail::ReduceLaunchTraits to route
   * kernel launches through the @c dense_disg_distributed launch helpers (same mechanism as
   * @c DenseDisg::DisgPolicy ).
   */
  using DenseDisgDistributedPolicy = void;

  using Span = detail::dense_disg_distributed::Span;
  using Idx = detail::dense_disg_distributed::Idx;

  template <typename T, int Components>
  using Partition = detail::dense_disg_distributed::Partition<T, Components>;

  template <typename T, int Components>
  using Field = detail::dense_disg_distributed::Field<T, Components>;

  /**
   * @brief Build the Slab layout over @p system for @p global_extent .
   *
   * @param system         Backing System (devices, streams, events). Stored by value.
   * @param global_extent  Full Cartesian @c (x, y, z) extent shared by all slabs.
   * @param stencil        Neighbour accesses over this grid; its radius (0 or 1) sets the z halo.
   * @param boundary_mode  Per-axis boundary semantics; @c PeriodicY switches the slab to pencil
   *                       storage, @c PeriodicX is rejected.
   * @param origin         Physical-space origin at global integer cell @c (0,0,0) .
   * @param spacing        Isotropic cell spacing; must be finite and strictly positive.
   * @param block_override When set, used as the launch block for every local GPU instead of
   *                       @c dense_make_kernel_block() .
   *
   * @throws std::invalid_argument on a stencil radius other than 0 / 1, a periodic x axis, a
   *         negative extent, or a chunk thinner than two owned z (or, on pencil storage, y)
   *         layers.
   */
  DenseDisgDistributed(System system,
                       int64_3d global_extent,
                       Stencil stencil = Stencil{},
                       BoundaryModeDistributed boundary_mode = BoundaryModeDistributed::Open,
                       float32_3d origin = {},
                       float spacing = 1.0F,
                       std::optional<LaunchShape> block_override = std::nullopt);

  /** @brief Build a layout by auto-factorizing the world size for @p decomposition . */
  DenseDisgDistributed(System system,
                       int64_3d global_extent,
                       DecompositionMode decomposition,
                       Stencil stencil = Stencil{},
                       BoundaryModeDistributed boundary_mode = BoundaryModeDistributed::Open,
                       float32_3d origin = {},
                       float spacing = 1.0F,
                       std::optional<LaunchShape> block_override = std::nullopt);

  /**
   * @brief Build a layout from an explicit @p process_grid .
   *
   * The grid's @c world_size must match the rank count (distributed) or the local GPU count
   * (single-process).
   *
   * @throws std::invalid_argument when a brick chunk ( @c dims.x > 1 ) owns fewer than two x
   *         layers.
   */
  DenseDisgDistributed(System system,
                       int64_3d global_extent,
                       ProcessGrid process_grid,
                       Stencil stencil = Stencil{},
                       BoundaryModeDistributed boundary_mode = BoundaryModeDistributed::Open,
                       float32_3d origin = {},
                       float spacing = 1.0F,
                       std::optional<LaunchShape> block_override = std::nullopt);

  /** Default-construct an empty grid (no devices, no slabs). Placeholder for member fields. */
  DenseDisgDistributed() = default;

  [[nodiscard]] auto
  process_grid() const noexcept -> const ProcessGrid&;

  [[nodiscard]] auto
  global_extent() const noexcept -> const int64_3d&;

  [[nodiscard]] auto
  stencil() const noexcept -> const Stencil&;

  /** Exchanged z halo thickness (0 or 1), derived from @ref stencil . */
  [[nodiscard]] auto
  halo_z() const noexcept -> int64_t;

  /** Exchanged y halo thickness: @ref halo_z on pencil / brick storage, 0 on slab storage. */
  [[nodiscard]] auto
  halo_y() const noexcept -> int64_t;

  /** Exchanged x halo thickness: @ref halo_z on brick storage, 0 otherwise. */
  [[nodiscard]] auto
  halo_x() const noexcept -> int64_t;

  [[nodiscard]] auto
  boundary_mode() const noexcept -> BoundaryModeDistributed;

  [[nodiscard]] auto
  origin() const noexcept -> float32_3d;

  [[nodiscard]] auto
  spacing() const noexcept -> float;

  [[nodiscard]] auto
  system() noexcept -> System&;

  [[nodiscard]] auto
  system() const noexcept -> const System&;

  /**
   * Number of @b local GPUs participating in this layout (same as
   * @c system().gpu_group().size() ). Always 1 in distributed mode.
   */
  [[nodiscard]] auto
  gpu_count() const noexcept -> size_t;

  /** Number of z layers in the process grid ( @c dims.z ). */
  [[nodiscard]] auto
  global_slab_count() const noexcept -> size_t;

  /** Process-grid @c z coordinate of local span @c 0 . */
  [[nodiscard]] auto
  local_slab_offset() const noexcept -> size_t;

  /**
   * Process-grid rank of local span @c 0 : the world rank in distributed mode, @c 0 otherwise.
   * Local span @c i is rank @c local_rank_offset() + i .
   */
  [[nodiscard]] auto
  local_rank_offset() const noexcept -> int32_t;

  /** Owned z extent on GPU @p gpu . Throws if @p gpu >= gpu_count() . */
  [[nodiscard]] auto
  local_z(size_t gpu) const -> int64_t;

  /**
   * Global z index of the first @b owned layer on GPU @p gpu
   * ( @c spans()[gpu].z_offset() + Span::ghost_radius ). Throws if @p gpu >= gpu_count() .
   */
  [[nodiscard]] auto
  z_offset_global(size_t gpu) const -> int64_t;

  /** Owned extent on GPU @p gpu . Throws if @p gpu >= gpu_count() . */
  [[nodiscard]] auto
  local_extent(size_t gpu) const -> int64_3d;

  /** Owned cell count on GPU @p gpu . Throws if @p gpu >= gpu_count() . */
  [[nodiscard]] auto
  linear_elements(size_t gpu) const -> size_t;

  /** One owned cell count per GPU. */
  [[nodiscard]] auto
  linear_counts_per_gpu() const -> std::vector<size_t>;

  [[nodiscard]] auto
  spans() const noexcept -> const std::vector<Span>& {
    return spans_;
  }

  /** Per-GPU launch grid under @p view . */
  [[nodiscard]] auto
  gpu_grid(DataView view = DataView::Standard) const noexcept -> const std::vector<LaunchShape>& {
    return gpu_grid_per_view_[static_cast<size_t>(view)];
  }

  /** Per-GPU launch block (same shape repeated for each device today). */
  [[nodiscard]] auto
  gpu_block(DataView view = DataView::Standard) const noexcept -> const std::vector<LaunchShape>& {
    return gpu_block_per_view_[static_cast<size_t>(view)];
  }

  /**
   * Allocate per-slab field storage in the disaggregated layout. @p side has the same meaning
   * as for @c DenseDisg::newField . Buffer sizing per slab is
   * @c spans()[g].linear_storage_elements(Components) = @c (Lz + 2) · Components · xy_pitch .
   */
  template <typename T,
            int Components>
  [[nodiscard]] auto
  newField(MemorySide side = MemorySide::Device,
           std::string name = {}) const -> Field<T,
                                                 Components>;

  /**
   * @brief Build a device-only compute @c Kernel . The @c DenseDisgDistributedPolicy marker
   *        routes the per-cell launch through @c dense_disg_distributed::launch_disg_grid . Must
   *        be called from a @c .cu translation unit (or any TU compiled with nvcc) under CUDA.
   */
  template <typename L>
  [[nodiscard]] auto
  make_kernel(L&& loading_fn,
              size_t shared_mem = 0U) const -> Kernel;

  /** @brief Build a dual-target (device + host) compute @c Kernel . */
  template <typename L>
  [[nodiscard]] auto
  make_universal_kernel(L&& loading_fn,
                        size_t shared_mem = 0U) const -> Kernel;

  /** @brief Build a host (OpenMP) compute @c Kernel over @c host_partitions() . */
  template <typename L>
  [[nodiscard]] auto
  make_host_kernel(L&& loading_fn) const -> Kernel;

  /** @brief Build a device-only reduction @c Kernel . */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_reduce_kernel(L&& loading_fn,
                     ReductionResult<T>& dst,
                     Op op = Op{},
                     size_t shared_mem = 0U) const -> Kernel;

  /** @brief Build a dual-target reduction @c Kernel . */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_universal_reduce_kernel(L&& loading_fn,
                               ReductionResult<T>& dst,
                               Op op = Op{},
                               size_t shared_mem = 0U) const -> Kernel;

  /** @brief Build a host (OpenMP) reduction @c Kernel . */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_host_reduce_kernel(L&& loading_fn,
                          ReductionResult<T>& dst,
                          Op op = Op{}) const -> Kernel;

private:
  int64_3d global_{};
  System system_;
  Stencil stencil_{};
  int64_t halo_z_{};
  int64_t halo_y_{};
  int64_t halo_x_{};
  BoundaryModeDistributed boundary_mode_{BoundaryModeDistributed::Open};
  float32_3d origin_{};
  float spacing_{1.0F};
  size_t global_slab_count_{};
  size_t local_slab_offset_{};
  int32_t local_rank_offset_{};
  std::vector<Span> spans_{};
  std::array<std::vector<LaunchShape>, data_view_count> gpu_grid_per_view_{};
  std::array<std::vector<LaunchShape>, data_view_count> gpu_block_per_view_{};
  ProcessGrid process_grid_{};

  void
  init_common_(System system,
               int64_3d global_extent,
               Stencil stencil,
               BoundaryModeDistributed boundary_mode,
               float32_3d origin,
               float spacing,
               ProcessGrid process_grid,
               std::optional<LaunchShape> block_override);
};

}  // namespace carbon

#include <carbon/detail/dense_disg_distributed/dense_disg_distributed.inl>
