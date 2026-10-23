#pragma once

/**
 * Public ::carbon::Kernel — the type-erased, value-semantic, polymorphic handle for a
 * runnable unit on a Carbon @c System. Mirrors @c Neon::set::Container : a
 * @c std::shared_ptr<KernelImpl> wrapped in a copyable value type, with concrete impls
 * for compute (device kernel over a dense grid), reduction, halo exchange, and graph
 * anchor.
 *
 * @par Why a single value type
 * Every kernel kind exposes the same launch signature
 * (@c launch(stream, view = DataView::Standard)), which lets users collect a heterogeneous
 * pipeline in a @c std::vector<Kernel> and run it in order — the same idiom Neon's
 * Skeleton uses (see @c std::vector<Container> in @c MultiXpuGraph). The future
 * carbon skeleton / graph layer will consume exactly this vector type and turn it into a
 * dependency graph; nothing in the user-facing API changes when that layer lands.
 *
 * @par Factory entry points
 * - @ref Kernel::compute — device kernel from a Neon-style loading lambda.
 * - @ref Kernel::reduce  — grid-wide reduction over @c DataView::Standard
 *                         (multi-GPU + multi-rank via the three-stage pipeline
 *                         in @c detail/reduce_stages.h ; the Internal / Boundary
 *                         split is the universal-reduce kernel's responsibility).
 * - @ref Kernel::halo    — wraps an existing @ref Halo<T, Components>.
 * - @ref Kernel::anchor  — no-op placeholder for graph synchronisation.
 *
 * Convenience members @c Dense::make_kernel and @c Dense::make_reduce_kernel forward to
 * the static factories so the existing two-argument call shape stays identical.
 *
 * @par Loading lambda
 * Compute and reduction kernels are constructed from a @e loading lambda — a host
 * callable invoked once per launch with a @ref KernelLoader (carrying the relative GPU
 * id) that returns the per-cell device callable. This is Neon's
 * @c std::function<UserComputeLambdaT(Loader&)> shape, preserved through the type-
 * erased wrapper because each concrete impl stores the lambda type internally.
 *
 * @par CUDA gating
 * The public @c Kernel type itself is host-only and trivially compilable without nvcc.
 * Concrete impls that need nvcc (compute, reduce) gate their launch bodies on
 * @c __CUDACC__; instantiating those factories outside an nvcc TU triggers a
 * @c static_assert at the call site rather than at @c launch time.
 */

#include <carbon/access_token.h>
#include <carbon/block.h>
#include <carbon/data_view.h>
#include <carbon/dense.h>
#include <carbon/dense_disg.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/detail/nvtx_scope.h>
#include <carbon/halo.h>
#include <carbon/halo_block.h>
#include <carbon/halo_disg.h>
#include <carbon/logging.h>
#include <carbon/mgrid.h>
#include <carbon/reduce.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbon {

/**
 * Tag identifying which buffer side a @ref KernelLoader hands back to the user's
 * loading lambda. Promoted to a namespace-scope enum (was nested in @c KernelLoader)
 * so the same enum names are usable in user-facing API surfaces without going through
 * @c KernelLoader::Target — historical aliasing kept for source compatibility.
 *
 * - @c Device: device-only / dual-target kernels feed @c field.partitions() (device
 *   USM / CUDA pointers).
 * - @c Host: host-only / dual-target kernels feed @c field.host_partitions() (CPU
 *   mirror; field must have been built with @c MemorySide::Host or @c HostDevice).
 *
 * @note We considered making @c KernelLoader a template @c <KernelTarget T> so the
 *       device-loader and host-loader types are statically distinct, but nvcc rejects
 *       extended @c __host__ @c __device__ lambdas defined inside a generic lambda,
 *       and a universal kernel's loading lambda would need to be generic to accept
 *       both specializations. The runtime byte stays in @c KernelLoader so the same
 *       loading lambda — non-generic, with @c CARBON_HOST_DEVICE per-cell body — can
 *       feed every factory shape without tripping that compiler bug.
 */
enum class KernelTarget : unsigned char {
  Device = 0,
  Host = 1,
};

/**
 * Whether a @ref KernelLoader is being driven for one-shot dependency
 * *discovery* or for a hot-path *launch*.
 *
 * The loading lambda has two jobs bundled into one call: (1) it records access
 * tokens (and, for @ref KernelLoader::read_stencil , snapshots a
 * @c HaloFactoryBase capturing a full @c Field copy) for the @ref Skeleton's
 * dependency analysis, and (2) it returns the per-cell device/host callable that
 * actually runs. Job (1) is needed exactly once, at kernel construction; job (2)
 * is needed on every launch. Re-running job (1) on every launch (per GPU!) meant
 * every @c launch() paid a @c std::make_shared + @c Field copy + token-vector /
 * string allocations that were immediately discarded — pure host-side overhead
 * that starved the GPU.
 *
 * - @c Discovery: record tokens + snapshot halo factories (the one-shot
 *   @c discover_tokens_ pass). This is the default so any un-updated construction
 *   site stays correct (dependency analysis never silently loses a token).
 * - @c Launch: skip all metadata; build only the per-cell callable. Used by every
 *   per-GPU launch loop.
 */
enum class KernelLoaderPhase : unsigned char {
  Discovery = 0,
  Launch = 1,
};

/**
 * Context passed to the loading lambda (Neon @c Loader analog). Identifies which GPU
 * slab @c Kernel::launch is configuring; the same handle is the entry point for
 * partition lookup via @c loader(field).
 *
 * Today @c KernelLoader is a thin convenience over @c field.partitions()[gpu_i] — the
 * kernel writer no longer captures per-GPU partition vectors and indexes them by hand:
 *
 * @code
 * const auto kernel = dense.make_kernel(
 *     [&u_in, &u_out](carbon::KernelLoader& loader) {
 *       auto in  = loader(u_in);   // Partition<T, Components>
 *       auto out = loader(u_out);
 *       return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
 *         out.write(s, idx, 0, in.read(s, idx, 0));
 *       };
 *     },
 *     0U);
 * @endcode
 *
 * It is also the place the @e loader will grow into a real "register a read / write /
 * stencil-read on this field" interface when the skeleton / dependency-analysis layer
 * lands. Writing call sites as @c loader(field) today already lines them up for that
 * future evolution.
 *
 * @warning @b Lifetime: the loading lambda is stored by value inside the
 * @ref Kernel impl and re-invoked on every @c launch() (and once at
 * construction for token discovery). Any object the lambda touches inside the
 * loader closure must outlive the @ref Kernel itself. The canonical idiom is
 * to capture @ref Dense::Field (and equivalents) @b by-value: carbon's
 * @c Field is a cheap reference-counted handle over @c BufferGroup<T> 's
 * @c shared_ptr<Allocation<T>> internals, so a by-value capture is identical
 * in cost to a reference capture and removes the dangling-reference footgun.
 * Capturing @c Field& / @c ReductionResult<T>& by reference is allowed when
 * the storage outlives the kernel — typical for a kernel built inside the
 * same function that holds the fields, and used before that function returns
 * — but is a silent UAF on the first @c launch when the field has been
 * destroyed in between. The same applies to @ref ReductionResult bound to a
 * reduce-kernel: capture by value into the loading lambda even though the
 * impl holds its own reference for the result slot.
 *
 * @par Why not a template parameter on the target
 * See the note on @ref KernelTarget — nvcc rejects extended HD lambdas inside generic
 * lambdas, which forced this to stay a non-template struct with a runtime byte.
 */
struct KernelLoader {
  /** Backward-compat alias — prefer naming @c carbon::KernelTarget directly in new code. */
  using Target = KernelTarget;

  /**
   * Bind the loader to @p relative_gpu_id (same index as @c Dense::spans()).
   *
   * @p phase defaults to @c Discovery so a bare @c KernelLoader{i} records tokens
   * (the correctness-preserving default). Per-GPU launch loops pass
   * @c KernelLoaderPhase::Launch to skip the discovery-only metadata work.
   */
  explicit KernelLoader(size_t relative_gpu_id,
                        KernelTarget target = KernelTarget::Device,
                        KernelLoaderPhase phase = KernelLoaderPhase::Discovery) noexcept
      : relative_gpu_id_(relative_gpu_id),
        target_(target),
        phase_(phase) {}

  /** Index into @c Dense::spans() and @c System::gpu_group() for the current launch. */
  [[nodiscard]] auto
  relative_gpu_id() const noexcept -> size_t {
    return relative_gpu_id_;
  }

  /** Which buffer side @c operator() will route to. */
  [[nodiscard]] auto
  target() const noexcept -> KernelTarget {
    return target_;
  }

  /** Whether this loader records dependency metadata (@c Discovery) or not (@c Launch). */
  [[nodiscard]] auto
  phase() const noexcept -> KernelLoaderPhase {
    return phase_;
  }

  /**
   * @brief Hand back the per-GPU @c Partition view of @p field for this loader's GPU
   *        and record a conservative @c AccessMode::ReadWrite token for the future
   *        skeleton's dependency analysis.
   *
   * Routes to @c field.partitions() under @c KernelTarget::Device (device kernels) and
   * to @c field.host_partitions() under @c KernelTarget::Host (OpenMP kernels).
   * Refine the recorded mode by calling @ref read , @ref write , or @ref read_write
   * instead — those record the matching @ref AccessMode and are otherwise identical.
   *
   * @tparam Field Duck-typed: any type providing a @c partitions() , a
   *               @c host_partitions() , and a @c gpu_buffers() member. For
   *               @c Dense::Field<T, Components> @c gpu_buffers() returns the
   *               @c BufferGroup<T> whose @c get_uid() identifies the field across
   *               kernels; sparse / block grid families plug in their own
   *               @c Field / @c Partition with the same shape.
   *
   * @throws std::invalid_argument from the host branch when
   *         @c field.host_partitions() is empty (i.e. the field was built with
   *         @c MemorySide::Device and has no CPU mirror to bind to). Device branch
   *         is @c noexcept beyond the token-vector push.
   */
  template <typename Field>
  [[nodiscard]] auto
  operator()(const Field& field) const {
    record_token_(field, AccessMode::ReadWrite);
    return lookup_partition_(field);
  }

  /**
   * @brief Read-only access — records @c AccessMode::Read and returns the same
   *        @c Partition that @c operator() does.
   *
   * Use when the per-cell body only @e reads the field. Pairs of @c read tokens on
   * the same field do not conflict, so the @ref Skeleton can run two read-only
   * kernels concurrently on the same input.
   */
  template <typename Field>
  [[nodiscard]] auto
  read(const Field& field) const {
    record_token_(field, AccessMode::Read);
    return lookup_partition_(field);
  }

  /**
   * @brief Write-only access — records @c AccessMode::Write . Conflicts with any
   *        other access mode on the same field (RAW / WAR / WAW).
   *
   * Use when the per-cell body strictly writes the field (no read of the existing
   * value).
   */
  template <typename Field>
  [[nodiscard]] auto
  write(const Field& field) const {
    record_token_(field, AccessMode::Write);
    return lookup_partition_(field);
  }

  /**
   * @brief Read-write access — records @c AccessMode::ReadWrite . Same conflict
   *        semantics as @ref write (any other access conflicts), but documents that
   *        the per-cell body reads and writes the same field. Equivalent to the
   *        bare @c operator() call in dependency terms.
   */
  template <typename Field>
  [[nodiscard]] auto
  read_write(const Field& field) const {
    record_token_(field, AccessMode::ReadWrite);
    return lookup_partition_(field);
  }

  /**
   * @brief Stencil read — records @c AccessMode::Read with
   *        @c AccessPattern::Stencil and snapshots a @ref HaloFactoryBase for
   *        the field so the @c Skeleton can auto-insert a halo update before
   *        this kernel.
   *
   * Use when the per-cell body reads neighbour cells of @p field (anything
   * beyond the cell at the kernel's @c Idx ). The Skeleton uses the recorded
   * @c Stencil pattern to:
   *
   *   1. Auto-insert a halo update before the kernel when there is no
   *      preceding halo on @p field since the last write to @p field
   *      (one halo per write→stencil chain — see @c Skeleton::Options ). With
   *      @c Options::strict_halo set, the missing halo throws at @c compile
   *      time instead.
   *   2. Mark this kernel as a stencil node for the OCC rewrites
   *      (Standard / Extended / TwoWayExtended in @c carbon::OverlapMode ),
   *      which split the stencil into Internal+Boundary and overlap
   *      compute with the halo exchange.
   *
   * Stencil writes are not supported (matches Neon — see Loader_imp.h:116).
   * Use @ref read_stencil only on the @e read side; the write side stays
   * a regular @ref write or @ref read_write .
   *
   * Definition is out-of-line at the bottom of this header so the type-erased
   * factory wrapper @c detail::TypedHaloFactory<Field> is visible (it lives
   * after @c Kernel is fully defined).
   *
   * @tparam Field Same duck-typed contract as @ref read / @ref write — the
   *               field type must additionally provide a
   *               @c make_halo_kernel() member returning a @c Kernel
   *               (carbon's general @c HaloFactory trait; @c Dense::Field
   *               satisfies it today, future grid families will too).
   */
  template <typename Field>
  [[nodiscard]] auto
  read_stencil(const Field& field) const;

  /**
   * @brief Stencil read with explicit halo semantic — same as @ref read_stencil but
   *        the auto-inserted halo uses @p stencil and @p semantic (e.g.
   *        @c StencilSemantic::Lattice for LBM populations). Mirrors Neon's
   *        @c Loader::load(field, Pattern::STENCIL, stencilSemantic) .
   *
   * @p stencil must be index-aligned with the field's @c Components when
   * @p semantic is @c StencilSemantic::Lattice (population @c q pairs with
   * @c stencil.points()[q] ). The field type must provide
   * @c make_halo_kernel(name, stencil, semantic) — @c Dense::Field and
   * @c DenseDisg::Field today. @c Block::Field transfers whole bricks (all
   * components fused) so it has no per-population lattice filtering and only
   * exposes the plain @c make_halo_kernel(name) — @c read_stencil(field, stencil,
   * semantic) on a Block field is a compile error by design.
   */
  template <typename Field>
  [[nodiscard]] auto
  read_stencil(const Field& field,
               Stencil stencil,
               StencilSemantic semantic) const;

  /**
   * @brief Snapshot of the access tokens recorded since this loader was constructed.
   *
   * Populated by @ref operator(), @ref read, @ref write, and @ref read_write. Used
   * by every concrete @c KernelImpl to invoke the user's loading lambda once at
   * construction with a fresh loader and stash the resulting @c tokens() for the
   * @ref Skeleton's dependency analysis. The token vector is per-loader, not
   * per-process — concurrent launches creating their own @c KernelLoader stacks do
   * not race.
   */
  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& {
    return tokens_;
  }

private:
  template <typename Field>
  [[nodiscard]] auto
  lookup_partition_(const Field& field) const {
    if (target_ == KernelTarget::Host) {
      const auto& parts = field.host_partitions();
      if (parts.empty()) {
        log::throw_with_log<std::invalid_argument>(
            "KernelLoader: host kernel requires a host mirror (MemorySide::Host or "
            "MemorySide::HostDevice); field was built without one (host_partitions() is "
            "empty).");
      }
      return parts[relative_gpu_id_];
    }
    return field.partitions()[relative_gpu_id_];
  }

  template <typename Field>
  auto
  record_token_(const Field& field,
                AccessMode mode,
                AccessPattern pattern = AccessPattern::Map,
                std::shared_ptr<detail::HaloFactoryBase> halo_factory = {}) const -> void {
    // Launch phase: tokens are discovery-only metadata, so skip the push_back
    // (and the field.name() std::string copy it entails) entirely on the hot
    // path. Any halo_factory handed in has already been suppressed by the
    // read_stencil guard, so nothing is leaked here.
    if (phase_ == KernelLoaderPhase::Launch) {
      return;
    }
    // gpu_buffers() is the canonical field identity probe: always defined (even on
    // Host-only fields the BufferGroup is default-constructed but its get_uid stays
    // stable per Field instance), and unlike cpu_buffers() / host_partitions() it
    // never triggers lazy host-mirror allocation on a HostDevice field. Discovery is
    // therefore allocation-free regardless of MemorySide.
    //
    // field.name() is captured by value alongside the uid so the Skeleton's DOT /
    // diagnostics layer has a stable label even after the field goes out of scope
    // (the kernel keeps the field alive through the captures in its loading lambda,
    // but the name is read during construction-time discovery and snapshotted —
    // a later set_name does not retroactively rewrite already-recorded tokens).
    tokens_.push_back(AccessToken{field.gpu_buffers().get_uid(), ResourceKind::Field, mode, pattern,
                                  field.name(), std::move(halo_factory)});
  }

  size_t relative_gpu_id_{};
  KernelTarget target_{KernelTarget::Device};
  KernelLoaderPhase phase_{KernelLoaderPhase::Discovery};
  // Mutable so the const-qualified loader methods (which match the historical
  // operator() signature) can append while staying const-correct from the user's
  // perspective. The vector is local to each KernelLoader instance, so there is no
  // shared-state concern.
  mutable std::vector<AccessToken> tokens_{};
};

/**
 * Type-erased, value-semantic handle for a runnable unit. See file-level doc above.
 *
 * Default-constructed @c Kernel objects are empty (@c empty() returns @c true);
 * @c launch on an empty kernel throws @c std::logic_error. Copies are cheap
 * (shared-pointer bump); two copies refer to the same underlying impl instance, so
 * configuration is shared — matching @c Neon::set::Container's semantics.
 */
class Kernel {
public:
  /** Empty handle; @c launch throws @c std::logic_error until a factory is used. */
  Kernel() noexcept = default;

  /**
   * @brief Forward to the underlying impl's launch using its *natural* target.
   *
   * Each kind picks a sensible default: device-only kinds (Compute / Reduction / Halo)
   * run on device; host-only kinds (ComputeHost / ReductionHost) run on host;
   * dual-target kinds (Universal / UniversalReduction) default to device — the common
   * call shape "iterate a @c std::vector<Kernel> via @c launch(stream)" therefore
   * lands on device, matching what @c Dense::make_universal_kernel users typically
   * mean by "default launch". Anchor is a no-op on either side.
   *
   * Use @ref launch_on_device or @ref launch_on_host on a Universal kernel to force a
   * specific side; on a single-target kernel, calling the wrong side throws.
   *
   * @throws std::logic_error if this kernel is empty.
   * @throws std::invalid_argument / std::runtime_error from the impl on misuse.
   */
  auto
  launch(const Stream& stream,
         DataView view = DataView::Standard) const -> void {
    if (!impl_) {
      log::throw_with_log<std::logic_error>(
          "Kernel::launch: kernel is empty (default-constructed)");
    }
    detail::NvtxPushPopScope nvtx_scope(name());
    impl_->launch(stream, view);
  }

  /** Phased reduce launch for skeleton OCC splits (@ref reduce::ReduceLaunchMode ). */
  auto
  launch_reduction_phase(const Stream& stream,
                         DataView view,
                         reduce::ReduceLaunchMode mode) const -> void {
    if (!impl_) {
      log::throw_with_log<std::logic_error>(
          "Kernel::launch_reduction_phase: kernel is empty (default-constructed)");
    }
    detail::NvtxPushPopScope nvtx_scope(name());
    impl_->launch_reduction_phase(stream, view, mode);
  }

  /**
   * @brief Force device-side launch.
   *
   * For Universal / UniversalReduction kernels: same as the device path on an explicit
   * @c Dense::make_kernel-built kernel (which is now device-only by default). For
   * single-target host kernels: throws @c std::logic_error. For Anchor: no-op.
   *
   * @throws std::logic_error if this kernel is empty or has no device launch.
   * @throws std::invalid_argument / std::runtime_error from the impl on misuse.
   */
  auto
  launch_on_device(const Stream& stream,
                   DataView view = DataView::Standard) const -> void {
    if (!impl_) {
      log::throw_with_log<std::logic_error>(
          "Kernel::launch_on_device: kernel is empty (default-constructed)");
    }
    detail::NvtxPushPopScope nvtx_scope(name());
    impl_->launch_on_device(stream, view);
  }

  /**
   * @brief Force host-side launch (OpenMP).
   *
   * For Universal / UniversalReduction kernels: runs the OpenMP body after waiting on
   * @p stream. For single-target device kernels: throws @c std::logic_error. For
   * Anchor: no-op.
   *
   * @throws std::logic_error if this kernel is empty or has no host launch.
   * @throws std::invalid_argument / std::runtime_error from the impl on misuse.
   */
  auto
  launch_on_host(const Stream& stream,
                 DataView view = DataView::Standard) const -> void {
    if (!impl_) {
      log::throw_with_log<std::logic_error>(
          "Kernel::launch_on_host: kernel is empty (default-constructed)");
    }
    detail::NvtxPushPopScope nvtx_scope(name());
    impl_->launch_on_host(stream, view);
  }

  /**
   * @brief Whether @ref launch_on_device is supported on this handle. Always @c false
   *        for an empty / default-constructed kernel.
   */
  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool {
    return impl_ && impl_->runs_on_device();
  }

  /**
   * @brief Whether @ref launch_on_host is supported on this handle. Always @c false
   *        for an empty / default-constructed kernel.
   */
  [[nodiscard]] auto
  runs_on_host() const noexcept -> bool {
    return impl_ && impl_->runs_on_host();
  }

  /** Coarse classification — see @ref KernelKind. Returns @c Anchor for an empty kernel. */
  [[nodiscard]] auto
  kind() const noexcept -> KernelKind {
    return impl_ ? impl_->kind() : KernelKind::Anchor;
  }

  /** Optional human-readable name; empty string for a default-constructed kernel. */
  [[nodiscard]] auto
  name() const noexcept -> std::string_view {
    return impl_ ? impl_->name() : std::string_view{};
  }

  /** True iff this is a default-constructed handle (no impl attached). */
  [[nodiscard]] auto
  empty() const noexcept -> bool {
    return !impl_;
  }

  /**
   * Underlying impl — exposed for the future skeleton/graph layer (token introspection,
   * DAG node identity, etc.). User code should not rely on this except through the
   * skeleton API once it lands.
   */
  [[nodiscard]] auto
  impl() const noexcept -> const std::shared_ptr<detail::KernelImpl>& {
    return impl_;
  }

  // ----- factories -----

  /**
   * @brief Build a compute kernel that runs @p loading 's per-cell device callable on
   *        every GPU of @p grid.
   *
   * Grid-family agnostic: every grid binds to the single
   * @c detail::DeviceKernelImpl ; per-family launch-helper variation lives
   * inside @c detail::DeviceLaunchTraits<Grid> (Dense / DenseDisg run a
   * Cartesian launch over @c span.iter_extent ; @c Block runs a
   * @c (block_count, 1, 1) thread-block grid — one CUDA block per retained
   * brick, one thread per voxel — and @c span.set_and_validate filters
   * inactive voxels).
   *
   * @param grid        Grid layout (e.g. @ref Dense or @ref Block); copied into the impl.
   * @param loading     Loading callable @c (KernelLoader&) -> F where @c F is a device
   *                    callable @c (Span, Idx) -> void. Invoked once per GPU per launch.
   * @param name        Optional human-readable name (used by logs and the skeleton's
   *                    DAG node labels).
   * @param shared_mem  Dynamic shared memory in bytes per launch; default 0.
   *
   * @note Must be called from a @c .cu translation unit (or any TU compiled with nvcc)
   *       under the CUDA backend; the impl's launch body is gated on @c __CUDACC__.
   *       Under SYCL, plain @c .cpp TUs work.
   */
  template <typename Grid,
            typename LoadingFn>
  [[nodiscard]] static auto
  compute(const Grid& grid,
          LoadingFn&& loading,
          std::string name = {},
          size_t shared_mem = 0U) -> Kernel;

  /**
   * @brief Build a grid-wide reduction kernel writing into @p dst.
   *
   * @tparam T   Reduction value type returned by the per-cell device lambda.
   *
   * @param grid        Grid layout. Multi-GPU and multi-rank are both supported via
   *                    the three-stage pipeline in @c detail/reduce_stages.h .
   * @param loading     Loading callable; same shape as @ref compute, but @c F returns
   *                    @c T (the per-cell contribution) instead of @c void.
   * @param dst         Destination held by reference for the lifetime of the returned
   *                    @c Kernel. Each @c launch overwrites the device scalar (no
   *                    accumulation across launches) and asynchronously copies it into
   *                    @c dst.host_data(); call @c dst.sync(stream) before reading
   *                    @c dst.host_value(). The destination's per-GPU partials must
   *                    have been built on the same @c System as @p grid (the ctor
   *                    asserts @c dst.gpu_count() == grid.gpu_count() ).
   * @param op          Reduction operator (default-constructed).
   * @param name        Optional human-readable name.
   * @param shared_mem  Dynamic shared memory in bytes (default 0).
   * @param options     Optional @ref reduce::ReduceOptions (e.g. @c BlockAtomic finalize).
   *
   * The loading lambda may fuse a map/stencil update with the reduction: call
   * @c KernelLoader::read_stencil / @c write (or @c read_write) in the loader and return
   * a per-cell @c T contribution from the same launch. Such kernels stay
   * @c KernelKind::Reduction and are not OCC-split into Internal/Boundary on multi-GPU.
   *
   * @throws std::invalid_argument when @c grid.gpu_count() == 0 or when
   *         @c dst.gpu_count() does not match @c grid.gpu_count() .
   */
  template <typename T,
            typename Grid,
            typename LoadingFn,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] static auto
  reduce(const Grid& grid,
         LoadingFn&& loading,
         ReductionResult<T>& dst,
         Op op = Op{},
         std::string name = {},
         size_t shared_mem = 0U,
         reduce::ReduceOptions options = {}) -> Kernel;

  /**
   * @brief D→H copy of the device reduction scalar into pinned host memory.
   *
   * Inserted automatically by @ref Skeleton when @c ReduceOptions::host_copy is
   * @c Deferred ; may also be added manually. Runs on the skeleton's detached copy
   * stream so compute can overlap the transfer.
   */
  template <typename T>
  [[nodiscard]] static auto
  scalar_readback(ReductionResult<T>& dst,
                  std::string name = {}) -> Kernel;

  /**
   * @brief Wrap an existing @ref Halo<T, Components> as a @c Kernel.
   *
   * The halo is moved into the impl; subsequent calls to @c launch invoke
   * @c halo(stream). The @c view argument is ignored — a halo update has no iteration
   * sub-domain.
   */
  template <typename T,
            int Components>
  [[nodiscard]] static auto
  halo(Halo<T,
            Components> h,
       std::string name = {}) -> Kernel;

  /**
   * @brief Wrap a @c HaloBlock<T, Components, SBlock> into a @c Kernel . The
   *        block-sparse twin of @ref halo above: forwards @ref launch to the
   *        halo's @c operator()(stream) ; the @c DataView argument is ignored.
   *        Used by @c Block::Field::make_halo_kernel when the field has a
   *        back-reference to its owning @c Block (see
   *        @ref detail::block::Field::set_block ).
   */
  template <typename T,
            int Components,
            typename SBlock>
  [[nodiscard]] static auto
  halo_block(HaloBlock<T,
                       Components,
                       SBlock> h,
             std::string name = {}) -> Kernel;

  /**
   * @brief Wrap a @c HaloDisg<T, Components> into a @c Kernel . The
   *        disaggregated-dense twin of @ref halo above: forwards @ref launch to the
   *        halo's @c operator()(stream) ; the @c DataView argument is ignored.
   *        Used by @c DenseDisg::Field::make_halo_kernel .
   */
  template <typename T,
            int Components>
  [[nodiscard]] static auto
  halo_disg(HaloDisg<T,
                     Components> h,
            std::string name = {}) -> Kernel;

  /**
   * @brief Build a dual-target (universal) compute kernel.
   *
   * The same loading lambda compiles into BOTH a device launch path (CUDA / SYCL) and a
   * host launch path (OpenMP). Default @c launch dispatches to device — call
   * @ref Kernel::launch_on_host on the resulting handle to take the host path on the
   * same kernel definition.
   *
   * @param grid        Grid layout (e.g. @ref Dense); copied into the impl.
   * @param loading     Loading callable @c (KernelLoader&) -> F where @c F is invocable
   *                    as @c (Span, Idx) -> void on both targets — practically a
   *                    @c CARBON_HOST_DEVICE-annotated closure.
   * @param name        Optional human-readable name.
   * @param shared_mem  Dynamic shared memory in bytes; honoured by the device path,
   *                    ignored by the host path.
   *
   * @note Must be called from a @c .cu translation unit (or any TU compiled with nvcc)
   *       under the CUDA backend. The host path itself does not require nvcc, but the
   *       device path's launch body is gated on @c __CUDACC__; instantiating this
   *       factory in a plain @c .cpp under CUDA triggers a @c static_assert at the
   *       call site. Under the SYCL backend, plain @c .cpp TUs work.
   */
  template <typename Grid,
            typename LoadingFn>
  [[nodiscard]] static auto
  compute_universal(const Grid& grid,
                    LoadingFn&& loading,
                    std::string name = {},
                    size_t shared_mem = 0U) -> Kernel;

  /**
   * @brief Build a dual-target (universal) reduction kernel.
   *
   * Same dual semantics as @ref compute_universal, scaled up to a grid-wide reduction:
   * the device path goes through CUB / sycl::reduction (same code as
   * @ref Kernel::reduce), the host path goes through OpenMP per-thread partials
   * combined sequentially through @p op (same code as @ref Kernel::host_reduce).
   *
   * Default @c launch runs the device path (writes to @c dst.device_data and async-
   * copies to @c dst.host_data; user calls @c dst.sync(stream) before reading).
   * @ref Kernel::launch_on_host writes directly to @c dst.host_data and skips the
   * async copy entirely (subsequent @c dst.sync is then a no-op).
   *
   * Multi-GPU and multi-rank are both supported, sharing the three-stage pipeline
   * with @ref Kernel::reduce . The Internal / Boundary @c DataView split is the
   * universal-reduce kernel's distinguishing capability: launches passed
   * @c DataView::Internal or @c DataView::Boundary skip stage 3 (cross-rank
   * combine) and merge with the matching halo-update kernel on the next compile.
   *
   * @throws std::invalid_argument when @c grid.gpu_count() == 0 or when
   *         @c dst.gpu_count() does not match @c grid.gpu_count() .
   */
  template <typename T,
            typename Grid,
            typename LoadingFn,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] static auto
  reduce_universal(const Grid& grid,
                   LoadingFn&& loading,
                   ReductionResult<T>& dst,
                   Op op = Op{},
                   std::string name = {},
                   size_t shared_mem = 0U,
                   reduce::ReduceOptions options = {}) -> Kernel;

  /**
   * @brief Build a host (CPU / OpenMP) compute kernel that runs @p loading 's per-cell
   *        callable inside an OpenMP @c parallel for over the iter extent of @p grid.
   *
   * Twin of @ref compute. The user-facing surface is identical: the same loading lambda
   * (one returning a per-cell @c (Span, Idx) -> void closure) feeds either factory, and
   * the resulting @c Kernel value is the same type. What changes is the @c launch path
   * (OpenMP nested-for instead of @c cudaLaunchKernel / @c sycl::queue::parallel_for) and
   * which buffer side @c loader(field) routes to (@c host_partitions instead of
   * @c partitions).
   *
   * @param grid     Grid layout (e.g. @ref Dense); copied into the impl.
   * @param loading  Loading callable @c (KernelLoader&) -> F where @c F is invocable as
   *                 @c (Span, Idx) -> void.
   * @param name     Optional human-readable name.
   *
   * @note Compatible with both the CUDA and SYCL backends — runs purely on the host
   *       toolchain and does not require an nvcc / acpp TU. The @c CARBON_HOST_DEVICE
   *       macro the user sprinkles on the per-cell lambda is a no-op on host TUs, so
   *       a body authored for the device kernel works as-is here.
   */
  template <typename Grid,
            typename LoadingFn>
  [[nodiscard]] static auto
  host_compute(const Grid& grid,
               LoadingFn&& loading,
               std::string name = {}) -> Kernel;

  /**
   * @brief Build a host (CPU / OpenMP) reduction kernel writing into @p dst.
   *
   * Twin of @ref reduce. Uses thread-local partials combined sequentially through
   * @p op so arbitrary @c reduce::* operators (not just @c reduce::Sum) work without
   * an explicit @c omp declare reduction. The result lands in @c dst.host_data()
   * directly; @c dst.sync(stream) is still safe but is effectively a no-op (the device
   * scalar is not touched).
   *
   * @tparam T   Reduction value type returned by the per-cell host lambda.
   *
   * @param grid     Grid layout. Multi-GPU and multi-rank are both supported,
   *                 mirroring the device reducer's three-stage pipeline (per-thread
   *                 partials per GPU slab in stage 1, intra-process combine on the
   *                 root GPU in stage 2, cross-rank @c MPI_Allreduce in stage 3).
   * @param loading  Loading callable; same shape as @ref host_compute, but @c F returns
   *                 @c T instead of @c void.
   * @param dst      Destination held by reference for the lifetime of the returned
   *                 @c Kernel. Per-GPU partials must have been built on the same
   *                 @c System as @p grid .
   * @param op       Reduction operator (default-constructed).
   * @param name     Optional human-readable name.
   *
   * @throws std::invalid_argument when @c grid.gpu_count() == 0 or when
   *         @c dst.gpu_count() does not match @c grid.gpu_count() .
   */
  template <typename T,
            typename Grid,
            typename LoadingFn,
            typename Op = reduce::Sum<T>>
  [[nodiscard]] static auto
  host_reduce(const Grid& grid,
              LoadingFn&& loading,
              ReductionResult<T>& dst,
              Op op = Op{},
              std::string name = {}) -> Kernel;

  /**
   * @brief Build a no-op anchor kernel (graph synchronisation placeholder). @c launch
   *        is a no-op; @c kind() returns @ref KernelKind::Anchor.
   */
  [[nodiscard]] static auto
  anchor(std::string name = "anchor") -> Kernel;

  /**
   * @brief Adopt a custom @c KernelImpl (libcarbonpy Warp foreign-launch path).
   */
  [[nodiscard]] static auto
  from_impl(std::shared_ptr<detail::KernelImpl> impl) -> Kernel {
    return Kernel(std::move(impl));
  }

private:
  explicit Kernel(std::shared_ptr<detail::KernelImpl> impl) noexcept
      : impl_(std::move(impl)) {}

  std::shared_ptr<detail::KernelImpl> impl_{};
};

}  // namespace carbon

// Templated impl headers must be visible before the inline factory definitions below.
// They depend only on detail::KernelImpl (already pulled in above), so include order is
// not circular.
#include <carbon/detail/block_halo_kernel_impl.h>
#include <carbon/detail/dense_host_kernel_impl.h>
#include <carbon/detail/dense_kernel_impl.h>
#include <carbon/detail/halo_disg_kernel_impl.h>
#include <carbon/detail/halo_kernel_impl.h>
#include <carbon/detail/reduce_host_kernel_impl.h>
#include <carbon/detail/reduce_kernel_impl.h>
#include <carbon/detail/scalar_readback_kernel_impl.h>
#include <carbon/detail/universal_kernel_impl.h>
#include <carbon/detail/universal_reduce_kernel_impl.h>

namespace carbon {

template <typename Grid,
          typename LoadingFn>
auto
Kernel::compute(const Grid& grid,
                LoadingFn&& loading,
                std::string name,
                size_t shared_mem) -> Kernel {
  // Single concrete impl across all grid families — per-family launch-helper
  // variation lives inside @c detail::DeviceLaunchTraits<Grid> (see
  // @c detail/launch_traits.h ). Block / Dense / DenseDisg all bind to
  // @c DeviceKernelImpl ; the trait absorbs the helper choice and the
  // parameter-shape variation (Dense / DenseDisg unpack @c gpu_grid[i] /
  // @c gpu_block[i] ; Block derives the dims internally from @c SBlock ).
  using Impl = detail::DeviceKernelImpl<Grid, std::decay_t<LoadingFn>>;
  return Kernel(
      std::make_shared<Impl>(grid, std::forward<LoadingFn>(loading), shared_mem, std::move(name)));
}

template <typename T,
          typename Grid,
          typename LoadingFn,
          typename Op>
auto
Kernel::reduce(const Grid& grid,
               LoadingFn&& loading,
               ReductionResult<T>& dst,
               Op op,
               std::string name,
               size_t shared_mem,
               reduce::ReduceOptions options) -> Kernel {
  // Single concrete impl across all grid families — per-family stage-1
  // launch variation lives inside @c detail::ReduceLaunchTraits<Grid> (see
  // @c detail/launch_traits.h ). Block / Dense / DenseDisg all bind to
  // @c ReduceKernelImpl ; the trait absorbs the stage-1 kernel choice,
  // block-dim Z (kReduceBlockThreads vs SBlock::VoxelCount), and the
  // SYCL launcher (launch_dense_reduce vs launch_block_reduce).
  using Impl = detail::ReduceKernelImpl<Grid, std::decay_t<LoadingFn>, T, std::decay_t<Op>>;
  return Kernel(std::make_shared<Impl>(grid, std::forward<LoadingFn>(loading), dst, std::move(op),
                                       shared_mem, std::move(name), options));
}

template <typename T>
auto
Kernel::scalar_readback(ReductionResult<T>& dst,
                        std::string name) -> Kernel {
  using Impl = detail::ScalarReadbackKernelImpl<T>;
  return Kernel(std::make_shared<Impl>(dst, std::move(name)));
}

template <typename T>
auto
detail::TypedScalarReadbackFactory<T>::make() const -> Kernel {
  return Kernel::scalar_readback<T>(*dst_, "auto_scalar_readback");
}

template <typename T,
          int Components>
auto
Kernel::halo(Halo<T,
                  Components> h,
             std::string name) -> Kernel {
  using Impl = detail::HaloKernelImpl<T, Components>;
  return Kernel(std::make_shared<Impl>(std::move(h), std::move(name)));
}

template <typename T,
          int Components,
          typename SBlock>
auto
Kernel::halo_block(HaloBlock<T,
                             Components,
                             SBlock> h,
                   std::string name) -> Kernel {
  using Impl = detail::BlockHaloKernelImpl<T, Components, SBlock>;
  return Kernel(std::make_shared<Impl>(std::move(h), std::move(name)));
}

template <typename T,
          int Components>
auto
Kernel::halo_disg(HaloDisg<T,
                           Components> h,
                  std::string name) -> Kernel {
  using Impl = detail::HaloDisgKernelImpl<T, Components>;
  return Kernel(std::make_shared<Impl>(std::move(h), std::move(name)));
}

inline auto
Kernel::anchor(std::string name) -> Kernel {
  return Kernel(std::make_shared<detail::AnchorKernelImpl>(std::move(name)));
}

template <typename Grid,
          typename LoadingFn>
auto
Kernel::host_compute(const Grid& grid,
                     LoadingFn&& loading,
                     std::string name) -> Kernel {
  // Single concrete impl across all grid families — per-family iteration-
  // shape variation lives inside @c detail::HostLaunchTraits<Grid> (see
  // @c detail/launch_traits.h ). Block / Dense / DenseDisg all bind to
  // @c DenseHostKernelImpl (the name is historical; the impl is now grid-
  // family agnostic).
  using Impl = detail::DenseHostKernelImpl<Grid, std::decay_t<LoadingFn>>;
  return Kernel(std::make_shared<Impl>(grid, std::forward<LoadingFn>(loading), std::move(name)));
}

template <typename Grid,
          typename LoadingFn>
auto
Kernel::compute_universal(const Grid& grid,
                          LoadingFn&& loading,
                          std::string name,
                          size_t shared_mem) -> Kernel {
  // Single concrete impl across all grid families — both per-family launch
  // variations (device-launch helper + iteration shape) live inside
  // @c detail::DeviceLaunchTraits<Grid> and @c detail::HostLaunchTraits<Grid>
  // (see @c detail/launch_traits.h ).
  using Impl = detail::UniversalKernelImpl<Grid, std::decay_t<LoadingFn>>;
  return Kernel(
      std::make_shared<Impl>(grid, std::forward<LoadingFn>(loading), shared_mem, std::move(name)));
}

template <typename T,
          typename Grid,
          typename LoadingFn,
          typename Op>
auto
Kernel::reduce_universal(const Grid& grid,
                         LoadingFn&& loading,
                         ReductionResult<T>& dst,
                         Op op,
                         std::string name,
                         size_t shared_mem,
                         reduce::ReduceOptions options) -> Kernel {
  // Single concrete impl across all grid families — per-family launch
  // variation lives inside @c detail::ReduceLaunchTraits<Grid> (device
  // stage-1) and @c detail::HostLaunchTraits<Grid>::iterate_slab_reduce
  // (host fold). See @c detail/launch_traits.h .
  using Impl =
      detail::UniversalReduceKernelImpl<Grid, std::decay_t<LoadingFn>, T, std::decay_t<Op>>;
  return Kernel(std::make_shared<Impl>(grid, std::forward<LoadingFn>(loading), dst, std::move(op),
                                       shared_mem, std::move(name), options));
}

template <typename T,
          typename Grid,
          typename LoadingFn,
          typename Op>
auto
Kernel::host_reduce(const Grid& grid,
                    LoadingFn&& loading,
                    ReductionResult<T>& dst,
                    Op op,
                    std::string name) -> Kernel {
  // Single concrete impl across all grid families — per-family iteration-
  // shape variation lives inside @c detail::HostLaunchTraits<Grid>::iterate_slab_reduce
  // (see @c detail/launch_traits.h ).
  using Impl = detail::ReduceHostKernelImpl<Grid, std::decay_t<LoadingFn>, T, std::decay_t<Op>>;
  return Kernel(std::make_shared<Impl>(grid, std::forward<LoadingFn>(loading), dst, std::move(op),
                                       std::move(name)));
}

// Per-grid-family convenience members (Dense / Block / DenseDisg). Split into
// companion .inl tail-includes to keep this header focused on the cross-grid
// Kernel / KernelLoader machinery + Kernel's static factories. Each .inl file
// is included exactly once, in this order, and is not a standalone header
// (the contents need Kernel + Halo + Block + DenseDisg + Dense fully visible,
// which the includes above guarantee).
#include <carbon/detail/kernel_block.inl>
#include <carbon/detail/kernel_dense.inl>
#include <carbon/detail/kernel_dense_disg.inl>
#include <carbon/detail/kernel_mgrid.inl>

// ----- HaloFactoryBase + TypedHaloFactory<FieldT> -----
//
// Type erasure for the "field can produce its own halo kernel on demand" trait.
// Forward-declared on AccessToken (carbon/access_token.h); defined here so the
// concrete impls have Kernel fully visible. The trait is satisfied by any type
// providing a Kernel-returning make_halo_kernel() member — Dense::Field<T, C>
// today, future grid families (sparse / multi-resolution) the same way.
namespace detail {

class HaloFactoryBase {
public:
  HaloFactoryBase() = default;
  HaloFactoryBase(const HaloFactoryBase&) = delete;
  HaloFactoryBase(HaloFactoryBase&&) = delete;
  auto
  operator=(const HaloFactoryBase&) -> HaloFactoryBase& = delete;
  auto
  operator=(HaloFactoryBase&&) -> HaloFactoryBase& = delete;
  virtual ~HaloFactoryBase() = default;

  /**
   * @brief Materialise a fresh halo @c Kernel for the captured field handle.
   *
   * Invoked by @c Skeleton::compile during the auto-halo insertion pass, at
   * most once per @c (field_uid, write→stencil chain) thanks to the
   * compile-time dedup cache. Implementations forward to the field's own
   * @c make_halo_kernel() so the produced @c Kernel is indistinguishable
   * from a halo the user added manually.
   */
  [[nodiscard]] virtual auto
  make() const -> Kernel = 0;
};

template <typename FieldT>
class TypedHaloFactory final : public HaloFactoryBase {
public:
  explicit TypedHaloFactory(FieldT field) noexcept(std::is_nothrow_move_constructible_v<FieldT>)
      : field_(std::move(field)) {}

  [[nodiscard]] auto
  make() const -> Kernel override {
    // Deliberately leaves the auto-inserted halo's name empty so a renderer
    // (export_dot / log) can show "<auto>" or similar; the Skeleton labels
    // it via the Kernel returned by make_halo_kernel.
    return field_.make_halo_kernel("auto_halo_" + field_.name());
  }

private:
  FieldT field_;
};

template <typename FieldT>
class TypedLatticeHaloFactory final : public HaloFactoryBase {
public:
  TypedLatticeHaloFactory(
      FieldT field,
      Stencil stencil,
      StencilSemantic semantic) noexcept(std::is_nothrow_move_constructible_v<FieldT>)
      : field_(std::move(field)),
        stencil_(std::move(stencil)),
        semantic_(semantic) {}

  [[nodiscard]] auto
  make() const -> Kernel override {
    return field_.make_halo_kernel("auto_halo_" + field_.name(), stencil_, semantic_);
  }

private:
  FieldT field_;
  Stencil stencil_;
  StencilSemantic semantic_;
};

}  // namespace detail

template <typename Field>
auto
KernelLoader::read_stencil(const Field& field) const {
  // Launch phase: no token, no halo factory — the make_shared + Field copy below
  // is exactly the per-launch overhead this split exists to eliminate. Return the
  // partition straight away (the only launch-relevant output of this call).
  if (phase_ != KernelLoaderPhase::Launch) {
    // Build the type-erased factory snapshot first so the token can hold it.
    // FieldT is captured by value (cheap shallow handle on Dense::Field — refs
    // the BufferGroup / host storage / span vector through shared pointers) so
    // the factory survives the loader and the user's loading lambda's stack
    // frame, all the way to Skeleton::compile.
    auto factory = std::make_shared<detail::TypedHaloFactory<Field>>(field);
    record_token_(field, AccessMode::Read, AccessPattern::Stencil, std::move(factory));
  }
  return lookup_partition_(field);
}

template <typename Field>
auto
KernelLoader::read_stencil(const Field& field,
                           Stencil stencil,
                           StencilSemantic semantic) const {
  if (phase_ != KernelLoaderPhase::Launch) {
    auto factory = std::make_shared<detail::TypedLatticeHaloFactory<Field>>(
        field, std::move(stencil), semantic);
    record_token_(field, AccessMode::Read, AccessPattern::Stencil, std::move(factory));
  }
  return lookup_partition_(field);
}

}  // namespace carbon
