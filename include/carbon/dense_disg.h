#pragma once

#include <carbon/boundary_mode.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense_disg/idx.h>
#include <carbon/detail/dense_disg/partition.h>
#include <carbon/detail/dense_disg/span.h>
#include <carbon/export.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>
#include <carbon/launch_shape.h>
#include <carbon/memory_side.h>
#include <carbon/system.h>

#include <array>
#include <cstddef>
#include <cstdint>
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

namespace detail::dense_disg {
template <typename T, int N>
class Field;
}  // namespace detail::dense_disg

/**
 * Public face of the @b disaggregated dense grid (port of Neon's
 * @c Neon::domain::details::disaggregated::dGrid ). Sibling of @c carbon::Dense — same
 * z-slab partitioning model, same one-rank-per-GPU distributed semantics, but each
 * per-slab buffer is laid out across five physically disjoint z-regions
 * (lower ghost / down boundary / internal / up boundary / upper ghost) with a card-major
 * SoA inside the internal region. See @c carbon::detail::dense_disg::Span class docs for
 * the storage map and pitch formula.
 *
 * @par Stencil radius
 *
 * The disaggregated layout pins both ghost and boundary radius to 1 ( the five-region
 * decomposition does not generalise to thicker boundaries without changing the region
 * count, see @c Span::ghost_radius / @c Span::boundary_radius ). A stencil with radius
 * @c != 1 is rejected at construction. Stencils with radius @c 0 are allowed and result
 * in a degenerate single-region buffer with no halo (single-slab kernels work; halo
 * exchanges are no-ops).
 *
 * @par Public surface
 *
 * - Per-slab geometry ( @ref spans ) and per- @c DataView launch shapes
 *   ( @ref gpu_grid / @ref gpu_block ), populated at construction.
 * - @ref newField allocates per-slab @c Buffer storage sized from
 *   @c span.linear_storage_elements(Components) — i.e. the five-region total
 *   @c (Lz + 2) · C · xy_pitch — and binds matching @c Partition views.
 * - The six kernel factories ( @ref make_kernel , @ref make_reduce_kernel ,
 *   @ref make_host_kernel , @ref make_host_reduce_kernel ,
 *   @ref make_universal_kernel , @ref make_universal_reduce_kernel ) mirror
 *   @c Dense one-for-one and produce @c carbon::Kernel handles that the
 *   skeleton / overlap layer consumes uniformly across grid families.
 * - The multi-GPU + cross-rank halo round-trip is wired through
 *   @ref HaloDisg (intra-process @c Transfer + cross-rank
 *   @c DistTransfer under @c CARBON_HAVE_MPI_NCCL ).
 */
class CARBON_API DenseDisg {
public:
  /**
   * Marker typedef used by @c Kernel::compute / @c Kernel::reduce / @c Kernel::host_compute
   * (and friends) to route the per-cell launch through the disg backend helpers
   * (@c launch_disg_grid , @c disg_grid_reduce_kernel , ...) instead of the aggregated
   * dense ones. The dispatch is a single @c if constexpr ( requires { typename
   * Grid::DisgPolicy; } ) check at the @c Kernel factory level — same pattern @c Block
   * uses to identify itself as a block-sparse grid family.
   */
  using DisgPolicy = void;

  using Span = detail::dense_disg::Span;
  using Idx = detail::dense_disg::Idx;

  template <typename T, int Components>
  using Partition = detail::dense_disg::Partition<T, Components>;

  template <typename T, int Components>
  using Field = detail::dense_disg::Field<T, Components>;

  /**
   * Build a multi-GPU z-slab layout over @p system for the given @p global_extent .
   *
   * @param system        Backing System (devices, streams, events). Stored by value
   *                      (shallow over the underlying GpuGroup).
   * @param global_extent Full Cartesian @c (x, y, z) extent shared by all slabs.
   * @param halo_z        Halo thickness on each z side. Must be 0 (no halo) or 1
   *                      (the only stencil radius the disaggregated layout supports).
   *                      Defaults to 1.
   * @param boundary_mode Boundary semantics propagated into every per-GPU @c Span and
   *                      consulted by future halo exchange. Defaults to
   *                      @c BoundaryMode::Open .
   * @param origin        Physical-space origin at global integer cell @c (0,0,0).
   * @param spacing       Isotropic cell spacing; must be finite and strictly positive.
   *
   * @throws std::invalid_argument when @p halo_z is not 0 or 1, when @p global_extent
   *         has a negative component, or when the global z is too small to split across
   *         the visible / configured GPU count.
   */
  DenseDisg(System system,
            int64_3d global_extent,
            int64_t halo_z = 1,
            BoundaryMode boundary_mode = BoundaryMode::Open,
            float32_3d origin = {},
            float spacing = 1.0F);

  /** Default-construct an empty grid (no devices, no slabs). Placeholder for member fields. */
  DenseDisg() = default;

  [[nodiscard]] auto
  global_extent() const noexcept -> const int64_3d&;

  [[nodiscard]] auto
  halo_z() const noexcept -> int64_t;

  [[nodiscard]] auto
  boundary_mode() const noexcept -> BoundaryMode;

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
   * @c system().gpu_group().size() ). Always 1 in distributed mode (carbon's
   * one-rank-per-GPU model).
   */
  [[nodiscard]] auto
  gpu_count() const noexcept -> size_t;

  [[nodiscard]] auto
  global_slab_count() const noexcept -> size_t;

  [[nodiscard]] auto
  local_slab_offset() const noexcept -> size_t;

  [[nodiscard]] auto
  local_z(size_t gpu) const -> int64_t;

  [[nodiscard]] auto
  z_offset_global(size_t gpu) const -> int64_t;

  [[nodiscard]] auto
  spans() const noexcept -> const std::vector<Span>& {
    return spans_;
  }

  /** Per-GPU launch grid for the disg-flavour kernel under @p view . */
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
   * Allocate per-slab field storage in the disaggregated layout.
   *
   * @p side selects which sides of the field are reserved (matching @c Dense::newField ):
   * @c Device populates only @c gpu_buffers ; @c Host populates only @c cpu_buffers ;
   * @c HostDevice populates both, with the host mirror allocated lazily on first access.
   *
   * Buffer sizing per slab is @c spans()[g].linear_storage_elements(Components) =
   * @c (Lz + 2) · Components · xy_pitch , i.e. the disaggregated total.
   */
  template <typename T,
            int Components>
  [[nodiscard]] auto
  newField(MemorySide side = MemorySide::Device,
           std::string name = {}) const -> Field<T,
                                                 Components>;

  /**
   * @brief Build a device-only compute @c Kernel over this disg grid. Twin of
   *        @c Dense::make_kernel ; the @c Grid::DisgPolicy marker routes the
   *        per-cell launch through @c launch_disg_grid . Must be called from a
   *        @c .cu translation unit (or any TU compiled with nvcc) under CUDA.
   */
  template <typename L>
  [[nodiscard]] auto
  make_kernel(L&& loading_fn,
              size_t shared_mem = 0U) const -> Kernel;

  /**
   * @brief Build a dual-target (device + host) compute @c Kernel over this disg
   *        grid. Twin of @c Dense::make_universal_kernel .
   */
  template <typename L>
  [[nodiscard]] auto
  make_universal_kernel(L&& loading_fn,
                        size_t shared_mem = 0U) const -> Kernel;

  /**
   * @brief Build a host (OpenMP) compute @c Kernel over this disg grid. Twin of
   *        @c Dense::make_host_kernel ; the per-cell lambda routes through
   *        @c host_partitions() and uses the same nested OMP loop as the dense
   *        host impl.
   */
  template <typename L>
  [[nodiscard]] auto
  make_host_kernel(L&& loading_fn) const -> Kernel;

  /**
   * @brief Build a device-only reduction @c Kernel over this disg grid. Twin of
   *        @c Dense::make_reduce_kernel .
   */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_reduce_kernel(L&& loading_fn,
                     ReductionResult<T>& dst,
                     Op op = Op{},
                     size_t shared_mem = 0U) const -> Kernel;

  /**
   * @brief Build a dual-target reduction @c Kernel over this disg grid. Twin of
   *        @c Dense::make_universal_reduce_kernel .
   */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_universal_reduce_kernel(L&& loading_fn,
                               ReductionResult<T>& dst,
                               Op op = Op{},
                               size_t shared_mem = 0U) const -> Kernel;

  /**
   * @brief Build a host (OpenMP) reduction @c Kernel over this disg grid. Twin
   *        of @c Dense::make_host_reduce_kernel .
   */
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
  int64_t halo_z_{};
  BoundaryMode boundary_mode_{BoundaryMode::Open};
  float32_3d origin_{};
  float spacing_{1.0F};
  size_t global_slab_count_{};
  size_t local_slab_offset_{};
  std::vector<Span> spans_{};
  std::array<std::vector<LaunchShape>, data_view_count> gpu_grid_per_view_{};
  std::array<std::vector<LaunchShape>, data_view_count> gpu_block_per_view_{};
};

}  // namespace carbon

#include <carbon/detail/dense_disg/dense_disg.inl>
