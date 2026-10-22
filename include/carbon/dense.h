#pragma once

#include <carbon/boundary_mode.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense/kernel.h>
#include <carbon/detail/dense/partition.h>
#include <carbon/detail/dense/span.h>
#include <carbon/export.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>
#include <carbon/launch_shape.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

namespace carbon {

class Kernel;

template <typename T>
class ReductionResult;

namespace reduce {
template <typename T>
struct Sum;
}  // namespace reduce

namespace detail {
template <typename Grid, typename LoadingFn>
class DeviceKernelImpl;
}  // namespace detail

namespace detail::dense {
template <typename T, int N>
class Field;
}  // namespace detail::dense

/**
 * Logical Cartesian grid over a (possibly distributed) multi-GPU System: fixed global
 * @c (x, y, z), with the z extent split across @b global slabs. The local @c Dense holds spans
 * for the slabs that live on @b this rank's local GPUs only.
 *
 * Partitioning model
 * ------------------
 * - @b Single-process (no @c Distributed attached): the global slab count equals
 *   @c system.gpu_group().size(), each local GPU owns exactly one global slab, and
 *   @c local_slab_offset() is @c 0.
 * - @b Distributed (one rank per GPU, the model carbon assumes for MPI builds): the global slab
 *   count equals @c distributed().world_size(), this rank's @c gpu_group() must hold exactly
 *   one device, and @c local_slab_offset() equals @c distributed().world_rank(). Multiple GPUs
 *   per rank are rejected at construction (carbon's distributed model is one rank per GPU).
 *
 * Z slabs are as even as possible across the @b global slab count: the first
 * @c (z % global_slab_count()) global slabs receive one extra layer when @c z is not divisible
 * by the global slab count. Local slab @c i corresponds to global slab
 * @c local_slab_offset() + i; @c spans()[i].z_offset() is the start of that global slab's lower
 * halo (or first owned layer when @c halo_z == 0), measured in global @c z coordinates.
 *
 * Stores a shallow copy of @p system (see @ref System); all copies share the same underlying
 * devices, stream/event state, and (when present) @c Distributed handle.
 */
class CARBON_API Dense {
public:
  /** Slab layout for one GPU (alias of @c detail::dense::Span). */
  using Span = detail::dense::Span;
  /** Cell index for dense slab kernels (alias of @c detail::dense::Idx). */
  using Idx = detail::dense::Idx;

  /**
   * Per-GPU device-pointer view of a slab for one @ref Field, alias of
   * @c detail::dense::Partition<T, Components>. Kernel code reads cells via
   * @c partition.read(span, idx, component_id); see @ref detail::dense::Partition for the
   * SoA layout. Mirrors the @ref Field alias so users don't reach into @c detail/dense/ for
   * the partition view type either.
   */
  template <typename T, int Components>
  using Partition = detail::dense::Partition<T, Components>;

  /**
   * @brief Builds a multi-GPU z-slab layout over @p system for the given @p global_extent.
   * Computes per-GPU spans, linear counts, and CUDA launch grid/block shapes (see @ref gpu_grid).
   * Throws if extents are invalid or launch-shape validation fails for any device.
   *
   * @param stencil       Optional stencil describing neighbour accesses over this grid. Stored
   *                      by value; retrievable via @ref stencil(). Defaults to an empty stencil
   *                      (radius 0, i.e. no halo dependencies). Drives the halo thickness on
   *                      each per-GPU @c Span.
   * @param boundary_mode Boundary semantics applied to the grid (propagated into every per-GPU
   *                      @c Span and consulted by @c Partition::neighbour and the halo
   *                      exchange). Defaults to @ref BoundaryMode::Open (non-toroidal);
   *                      @ref BoundaryMode::PeriodicZ is honoured end-to-end (z-axis wrap in
   *                      both @c Partition::neighbour and @c Halo).
   * @param origin        Physical-space origin at global integer cell @c (0,0,0) (cell-corner
   *                      convention with @c Span::global_location).
   * @param spacing       Isotropic cell spacing; must be finite and strictly positive.
   * @param block_override When set, used as the CUDA/SYCL launch block for every local GPU
   *                      instead of @c dense_make_kernel_block(). The matching launch grid
   *                      is derived via @c compute_dense_grid for each @ref DataView.
   */
  Dense(System system,
        int64_3d global_extent,
        Stencil stencil = Stencil{},
        BoundaryMode boundary_mode = BoundaryMode::Open,
        float32_3d origin = {},
        float spacing = 1.0F,
        std::optional<LaunchShape> block_override = std::nullopt);

  /**
   * @brief Default-construct an empty grid: zero global extent, default
   *        @ref System (no devices), no slabs. Useful as a placeholder
   *        member or before an assignment from a real-args ctor; every
   *        accessor returns its zero value (e.g. @c global_extent() ==
   *        @c {0, 0, 0} , @c spans().empty() ). Required to make the
   *        default constructor of @c detail::dense::Field<T, C> well-
   *        formed (and through that, @c carbon::Halo<T, C> as well —
   *        both store a @c Dense by value as the "I'm a placeholder"
   *        state). Trivial implementation; all data members carry
   *        in-class default initializers.
   */
  Dense() = default;

  /**
   * @brief Global @c (x,y,z) extent shared by all slabs (before z splitting across GPUs).
   * @return Reference to the extent stored at construction; valid for the lifetime of this @c Dense.
   * @see local_extent for per-GPU subgrid extents.
   */
  [[nodiscard]] auto
  global_extent() const noexcept -> const int64_3d&;

  /**
   * @brief Stencil this grid was constructed with (empty by default).
   * @return Reference to the stored @ref Stencil; valid for the lifetime of this @c Dense.
   */
  [[nodiscard]] auto
  stencil() const noexcept -> const Stencil&;

  /**
   * @brief Boundary mode this grid was constructed with (default @c BoundaryMode::Open).
   *        Propagated to every entry of @ref spans() and consulted by @c Partition::neighbour
   *        and the halo exchange.
   */
  [[nodiscard]] auto
  boundary_mode() const noexcept -> BoundaryMode;

  /** Physical-space origin at global cell @c (0,0,0); same value on every @ref spans entry. */
  [[nodiscard]] auto
  origin() const noexcept -> float32_3d;

  /** Isotropic cell spacing carried on every @ref spans slab. */
  [[nodiscard]] auto
  spacing() const noexcept -> float;

  /**
   * @brief Mutable access to the backing @ref System (devices, streams, events).
   * @return Reference to the shared @c System state; copies of @c Dense alias the same state.
   * @see system() const for read-only access.
   */
  [[nodiscard]] auto
  system() noexcept -> System&;

  /**
   * @brief Read-only access to the backing @ref System (devices, streams, events).
   * @return Const reference to the shared @c System state.
   * @see system() for non-const access.
   */
  [[nodiscard]] auto
  system() const noexcept -> const System&;

  /**
   * @brief Number of @b local GPUs participating in this layout (same as
   *        @c system().gpu_group().size()). In distributed mode this is always @c 1 (carbon's
   *        one-rank-per-GPU model); in single-process mode it is the visible-GPU count.
   * @return Zero if the @c System has no devices; otherwise the local-GPU count.
   * @see gpu_grid and gpu_block for per-GPU launch metadata.
   * @see global_slab_count for the cross-rank slab total.
   */
  [[nodiscard]] auto
  gpu_count() const noexcept -> size_t;

  /**
   * @brief Total number of z-slabs across @b all ranks.
   * - Single-process: equals @ref gpu_count().
   * - Distributed: equals @c system().distributed()->world_size() (one rank per GPU).
   */
  [[nodiscard]] auto
  global_slab_count() const noexcept -> size_t;

  /**
   * @brief Global slab index of @b this rank's local slab @c 0. Equals @c 0 in single-process
   *        mode and @c distributed().world_rank() when distributed. Local slab @c i maps to
   *        global slab @c local_slab_offset() + i.
   */
  [[nodiscard]] auto
  local_slab_offset() const noexcept -> size_t;

  /**
   * @brief Local @b owned @c z extent on GPU @p gpu (excludes halo layers when the grid carries a
   *        non-trivial @ref Stencil; equals @c spans()[gpu].logic_extent().z).
   * @param gpu Must be less than @ref gpu_count(); otherwise throws.
   * @return Number of owned z layers on that device after splitting the global z extent.
   */
  [[nodiscard]] auto
  local_z(size_t gpu) const -> int64_t;

  /**
   * @brief Starting global z index of the @b owned region on GPU @p gpu (the slab's first
   *        non-halo layer; equals @c spans()[gpu].z_offset() + spans()[gpu].halo_z()).
   * @param gpu Must be less than @ref gpu_count(); otherwise throws.
   * @return Global z offset so owned cells on this GPU lie in @c [offset, offset + local_z(gpu)).
   */
  [[nodiscard]] auto
  z_offset_global(size_t gpu) const -> int64_t;

  /**
   * @brief Subgrid @b owned extent on GPU @p gpu: full global @c x,y; @c z equals
   *        @ref local_z(gpu) (excludes halos).
   * @param gpu Must be less than @ref gpu_count(); otherwise throws.
   * @return int64_3d describing the local owned Cartesian slab for that device.
   */
  [[nodiscard]] auto
  local_extent(size_t gpu) const -> int64_3d;

  /**
   * @brief Owned (non-haloed) element count on GPU @p gpu:
   *        @c global.x * global.y * local_z(gpu). For the haloed @b storage size used by
   *        @ref newField, see @c spans()[gpu].linear_storage_elements().
   * @param gpu Must be less than @ref gpu_count(); otherwise throws.
   * @return Number of owned scalar cells in the slab.
   */
  [[nodiscard]] auto
  linear_elements(size_t gpu) const -> size_t;

  /**
   * @brief One @b owned element count per GPU (computed from @ref spans). Use
   *        @c spans()[i].linear_storage_elements() when sizing buffers that include halos.
   * @return Vector parallel to @c system().gpu_group().devices() and @ref spans().
   * @see linear_elements for a single index.
   */
  [[nodiscard]] auto
  linear_counts_per_gpu() const -> std::vector<size_t>;

  /**
   * @brief One @ref Span per GPU (same order as @ref system), each describing the @b storage
   *        (haloed) layout of its slab: @c xy_slab_elements for pitch, @c z_offset for the
   *        global z-index of buffer-local @c z=0, and @c halo_z for the halo thickness derived
   *        from @ref stencil().
   *
   * When the grid has no stencil, @c halo_z() == 0 and @c haloed_extent() equals
   * @c logic_extent(). Use @c logic_extent() to iterate only owned cells; use @c haloed_extent()
   * for pointer math over the buffer.
   *
   * @see local_extent for the same owned-cell extents in @c int64_3d form.
   */
  [[nodiscard]] auto
  spans() const noexcept -> const std::vector<Span>& {
    return spans_;
  }

  /**
   * @brief Per-GPU launch grid for @c dense_grid_kernel under @p view.
   * @param view Iteration view (Standard / Internal / Boundary); see @ref DataView and
   *             @c Span::iter_extent. Defaults to @c DataView::Standard.
   * @return Computed in the constructor from @c spans()[i].iter_extent(view),
   *         @c dense_make_kernel_block(), and each device's @c max_threads_per_block.
   *         Returned as @ref LaunchShape so the public API does not depend on any
   *         backend-specific 3D type (the CUDA backend converts to @c dim3 internally;
   *         the future SYCL backend will convert to @c sycl::range<3>).
   * @see gpu_block for the matching block dimensions.
   */
  [[nodiscard]] auto
  gpu_grid(DataView view = DataView::Standard) const noexcept -> const std::vector<LaunchShape>& {
    return gpu_grid_per_view_[static_cast<size_t>(view)];
  }

  /**
   * @brief Per-GPU launch block for @p view (same shape repeated for each device today).
   *
   * Block dimensions are independent of @p view today (every view uses
   * @c dense_make_kernel_block()), but the API takes the view for symmetry with
   * @ref gpu_grid and to leave room for view-specific block tuning later.
   *
   * @return Vector parallel to @ref gpu_grid(view); each entry matches @c dense_make_kernel_block().
   * @see carbon/detail/dense/launch_shape.h for how the default block is chosen and validated.
   */
  [[nodiscard]] auto
  gpu_block(DataView view = DataView::Standard) const noexcept -> const std::vector<LaunchShape>& {
    return gpu_block_per_view_[static_cast<size_t>(view)];
  }

  /**
   * @brief Builds a deferred *device-only* compute @ref Kernel bound to every GPU in
   *        @ref system().
   *
   * The cheapest factory profile: only the device launch path is instantiated, and
   * the returned @c Kernel throws @c std::logic_error on @c launch_on_host. The
   * loading lambda is invoked with a @c carbon::KernelLoader @c & whose runtime
   * @c target() is @c KernelTarget::Device, so @c loader(field) returns the device
   * partition. If you also need a host (OpenMP) launch path on the same handle, use
   * @ref make_universal_kernel.
   *
   * Returns a value-typed @c Kernel handle (a thin shared_ptr wrapper around the
   * polymorphic @c detail::KernelImpl base) so the result can be stored alongside
   * reductions and halo exchanges in a @c std::vector<Kernel>. The @ref DataView is
   * not stored on the kernel — it is purely a launch-time choice, picked via
   * @ref Kernel::launch (defaults to @c DataView::Standard). Per-view launch shapes
   * are precomputed in the @c Dense constructor (see @ref gpu_grid / @ref gpu_block).
   * Definition lives in @c carbon/kernel.h.
   */
  template <typename L>
  [[nodiscard]] auto
  make_kernel(L&& loading_fn,
              size_t shared_mem = 0U) const -> Kernel;

  /**
   * @brief Builds a deferred *dual-target* (universal) compute @ref Kernel bound to
   *        every GPU in @ref system(). The returned kernel can be launched on either
   *        the device (CUDA / SYCL) or the host (OpenMP) from the same handle.
   *
   * Both launch paths are instantiated when this factory is called, so the consumer
   * pays a slightly higher compile-time cost than @ref make_kernel — earn it back
   * by using the same loading lambda for in-process device/host parity tests, or for
   * adaptive dispatch where the runtime picks the side per call. The loading lambda
   * sees a @c carbon::KernelLoader @c & on every launch (the same non-template type
   * on both paths); @c loader.target() differs at runtime so @c loader(field)
   * automatically routes to the device or host buffer, and the user writes one
   * concrete lambda body that nvcc accepts without the generic-lambda HD-trap.
   *
   * @par Default launch behaviour
   * @c Kernel::launch(stream) on the result dispatches to the device path so existing
   * call sites that iterate a @c std::vector<Kernel> keep their device behaviour.
   * @c Kernel::launch_on_host(stream) runs the OpenMP body on the same handle —
   * useful for debugging, validation against a CPU reference, or adaptive dispatch.
   *
   * @par Field side selection
   * The @c KernelLoader passed to the loading lambda routes @c loader(field) based on
   * the active launch target: device launches see @c Field::partitions, host launches
   * see @c Field::host_partitions. Fields touched by a host launch must therefore have
   * been built with @c MemorySide::Host or @c MemorySide::HostDevice.
   */
  template <typename L>
  [[nodiscard]] auto
  make_universal_kernel(L&& loading_fn,
                        size_t shared_mem = 0U) const -> Kernel;

  /**
   * @brief Builds a deferred *device-only* grid-wide reduction kernel bound to this
   *        @c Dense slab.
   *
   * The user's per-cell lambda returns @p T (instead of @c void); the launch pipeline
   * reduces those contributions to a single scalar @ref ReductionResult destination.
   * Multi-GPU and multi-rank are both supported through a three-stage pipeline
   * (per-GPU stage 1 via CUB / @c sycl::reduction , intra-process combine on the
   * root GPU as stage 2, cross-rank @c ncclAllReduce / @c MPI_Allreduce as stage 3
   * — see @c include/carbon/detail/reduce_stages.h ). Launches use
   * @c DataView::Standard ; the Internal / Boundary split is the universal-reduce
   * kernel's job. The loading lambda receives a @c carbon::KernelLoader @c & whose
   * @c target() is @c KernelTarget::Device. For a dual-target reduction kernel,
   * see @ref make_universal_reduce_kernel.
   *
   * The destination @ref ReductionResult is captured by reference at construction
   * time so the polymorphic @c Kernel::launch signature stays uniform with every
   * other kernel kind. Lifetime contract: @p dst must outlive the returned @c Kernel
   * handle.
   *
   * Definition lives in @c carbon/kernel.h.
   *
   * @tparam T   Reduction value type returned by the per-cell lambda.
   * @tparam L   Type of the loading callable (Neon-style: takes @ref KernelLoader,
   *             returns the per-cell lambda).
   * @tparam Op  Reduction operator with @c CARBON_HD @c identity() and
   *             @c operator()(T, T) -> T. Defaults to @c carbon::reduce::Sum<T>.
   *
   * @param loading_fn  Loading callable.
   * @param dst         Reduction destination; lifetime must extend beyond every
   *                    @c launch on the returned @c Kernel.
   * @param op          Reduction operator instance (default-constructed).
   * @param shared_mem  Dynamic shared memory in bytes (default 0).
   * @param options     Optional @ref reduce::ReduceOptions .
   */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_reduce_kernel(L&& loading_fn,
                     ReductionResult<T>& dst,
                     Op op = {},
                     size_t shared_mem = 0U,
                     reduce::ReduceOptions options = {}) const -> Kernel;

  /**
   * @brief Builds a deferred *dual-target* (universal) grid-wide reduction kernel.
   *
   * Same dual semantics as @ref make_universal_kernel but for a grid-wide reduction:
   * @c Kernel::launch(stream) defaults to the device path (CUB / sycl::reduction +
   * async D→H copy; user calls @c dst.sync(stream) before reading
   * @c dst.host_value()). @c Kernel::launch_on_host(stream) runs the OpenMP per-thread
   * partials path and writes directly to @c dst.host_data — no async copy, no sync
   * needed. The loading lambda must be generic (see @ref make_universal_kernel).
   *
   * Definition lives in @c carbon/kernel.h.
   */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_universal_reduce_kernel(L&& loading_fn,
                               ReductionResult<T>& dst,
                               Op op = {},
                               size_t shared_mem = 0U) const -> Kernel;

  /**
   * @brief Builds a deferred host (CPU / OpenMP) compute @ref Kernel bound to every GPU
   *        slab in @ref system().
   *
   * Twin of @ref make_kernel that walks the iter extent through an OpenMP nested-for
   * (z slowest, x fastest) instead of a CUDA / SYCL submit. The user's loading lambda
   * has the same shape; the per-cell @c (Span, Idx) -> void closure can be the same
   * @c CARBON_HOST_DEVICE-annotated body fed to @ref make_kernel. The loader passed in
   * is @c carbon::KernelLoader @c & with @c target() == @c KernelTarget::Host, so
   * @c loader(field) hands back @c Field::host_partitions instead of
   * @c Field::partitions — every field touched must have been built with
   * @c MemorySide::Host or @c MemorySide::HostDevice.
   *
   * Definition lives in @c carbon/kernel.h.
   *
   * @note Stream argument on @c Kernel::launch(stream) is waited-on first (so a host
   *       kernel sitting after a device kernel sees the device's writes flushed to the
   *       CPU mirror via a prior Transfer), then the OpenMP loop runs synchronously
   *       inline. There is no per-kernel shared-memory parameter: OpenMP has no analog.
   */
  template <typename L>
  [[nodiscard]] auto
  make_host_kernel(L&& loading_fn) const -> Kernel;

  /**
   * @brief Builds a deferred host (CPU / OpenMP) grid-wide reduction kernel.
   *
   * Twin of @ref make_reduce_kernel; the OpenMP path uses one accumulator per thread
   * (initialised to @c op.identity()) and combines the partials sequentially through
   * @p op so user-defined operators (not just @c reduce::Sum) work without an explicit
   * @c omp declare reduction. Result is written directly into @c dst.host_data() —
   * @c dst.sync(stream) after launch is still safe but a no-op for the host path.
   *
   * Definition lives in @c carbon/kernel.h.
   *
   * @throws std::invalid_argument when @c gpu_count() == 0 (the System carries no
   *         GPUs / slabs to drive the OpenMP loop). Multi-GPU and multi-rank are
   *         both supported, mirroring the device reducer's three-stage pipeline.
   */
  template <typename T,
            typename L,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] auto
  make_host_reduce_kernel(L&& loading_fn,
                          ReductionResult<T>& dst,
                          Op op = {}) const -> Kernel;

  /**
   * @brief Per-GPU device field storage (buffers + @ref Partition views) for this layout.
   * @tparam T          Element type per grid cell.
   * @tparam Components Number of components per cell (1 = scalar, 3 = vector, ...). Same value
   *                    flows through to @ref detail::dense::Partition.
   */
  template <typename T, int Components>
  using Field = detail::dense::Field<T, Components>;

  /**
   * @brief Allocates per-slab field storage for this @c Dense layout.
   *
   * @p side selects which sides of the field are reserved (see @ref MemorySide):
   *  - @c Device       : one device @ref Buffer per GPU (sized by
   *                      @c spans()[gpu].linear_storage_elements()); host buffers are empty.
   *  - @c Host         : one host @ref Buffer per GPU (same per-GPU sizes); device buffers are
   *                      empty. Partitions are bound to the host pointers so callers can use
   *                      the same @ref Partition / @ref Span APIs on the CPU side.
   *  - @c HostDevice   : both sides are allocated; partitions are bound to the @b device side
   *                      (matching the previous behaviour) so kernel launches keep working
   *                      unchanged. Use @ref Transfer to move data between sides.
   *
   * The default is @c MemorySide::Device, matching the original device-only behaviour so
   * existing callers keep working without changes.
   *
   * @throws std::runtime_error if @ref gpu_count() is zero (after @c CARBON_LOG_ERROR).
   */
  /**
   * @overload
   *
   * Like @ref newField above but also stamps the resulting field with a
   * diagnostic @p name (forwarded to @c Field::set_name ). The name is
   * optional — pass an empty string for unnamed fields. Used by the carbon
   * @c Skeleton 's DAG export to label edges and node tooltips, and propagated
   * through @c KernelLoader so token records carry the name to the analysis
   * stage. Identity (the @c BufferGroup uid) is independent of the name —
   * two fields with the same name are still distinct fields, and changing the
   * name post-hoc via @c Field::set_name does not affect any already-recorded
   * tokens (those carry the name they saw at @c make_kernel time).
   */
  template <typename T,
            int Components>
  [[nodiscard]] auto
  newField(MemorySide side = MemorySide::Device,
           std::string name = {}) const -> Field<T,
                                                 Components>;

private:
  int64_3d global_{};
  System system_;
  Stencil stencil_{};
  BoundaryMode boundary_mode_{BoundaryMode::Open};
  float32_3d origin_{};
  float spacing_{1.0F};
  // Cross-rank partitioning state. See class doxygen for the model. In single-process mode,
  // global_slab_count_ == gpu_count() and local_slab_offset_ == 0.
  size_t global_slab_count_{};
  size_t local_slab_offset_{};
  // spans_ is the single source of truth for per-GPU slab geometry: owned/haloed extent,
  // global z origin (with halo), halo thickness, and storage pitch. Owned counts and offsets
  // are derived from it on access; only the launch shapes (gpu_grid_/gpu_block_) are kept as
  // separate vectors because they depend on per-device launch caps, not just the layout.
  std::vector<Span> spans_{};
  // Per-DataView per-GPU launch shapes. Indexed by static_cast<size_t>(DataView): 0=Standard,
  // 1=Internal, 2=Boundary. Each inner vector parallels @ref spans() and @ref system(). Stored
  // as backend-neutral @ref LaunchShape; converted to the active backend's 3D type
  // (@c dim3 / @c sycl::range<3>) at the launch site inside @c carbon::detail.
  std::array<std::vector<LaunchShape>, data_view_count> gpu_grid_per_view_{};
  std::array<std::vector<LaunchShape>, data_view_count> gpu_block_per_view_{};
};

}  // namespace carbon

#include <carbon/detail/dense/dense.inl>
