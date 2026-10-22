#pragma once

/**
 * Device compute kernel concrete impl behind ::carbon::Kernel. The templated leaf
 * (parameterised on @c Grid and @c LoadingFn) that previously lived as
 * @c carbon::Kernel<Grid, LoadingFn> is now a private @c carbon::detail::DeviceKernelImpl;
 * the public @c Kernel value type forwards @c launch to it through virtual dispatch.
 *
 * Mirrors @c Neon::set::internal::DeviceContainer<Iter, LoadingLambda>: a templated
 * subclass of the polymorphic kernel base that stores the Neon-style loading callable
 * (@c (KernelLoader&) -> per-cell-device-lambda) and orchestrates the per-GPU launch
 * loop via the grid's @c launch_dense_grid friend hook.
 *
 * Include from @c .cu translation units (or any TU compiled with nvcc) when the
 * factory @c carbon::Kernel::compute / @c Dense::make_kernel is instantiated. Host-only
 * TUs that merely include @c <carbon/kernel.h> stay clean — @ref launch's CUDA body is
 * gated on @c __CUDACC__, and instantiation outside an nvcc TU triggers a
 * @c static_assert at the call site rather than a runtime exception.
 */

#include <carbon/data_view.h>
#include <carbon/detail/backend/select.h>
#include <carbon/detail/dense/kernel.h>
#include <carbon/detail/grid_concepts.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/detail/launch_traits.h>
#include <carbon/launch_shape.h>
#include <carbon/logging.h>
#include <carbon/system.h>

// All per-family launch-helper variation (dense / disg / block) lives behind
// @c carbon::detail::DeviceLaunchTraits<Grid> ; this header pulls it in once.
// The trait absorbs both the helper choice (launch_dense_grid vs
// launch_disg_grid vs launch_block_grid) and the parameter-shape variation
// (Dense / DenseDisg pass @c gpu_grid[i] / @c gpu_block[i] ; Block derives its
// dims internally from @c SBlock::Mem{X,Y,Z} ). See @c detail/launch_traits.h
// for the full surface.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <carbon/detail/backend/cuda_like.h>

namespace carbon {

struct KernelLoader;

namespace detail {

/**
 * Dependent-false constant for static_assert inside DeviceKernelImpl::launch's non-CUDA
 * branch. Keeps the assert from firing at definition time so simply including
 * @c <carbon/kernel.h> in a host-only TU stays clean — the diagnostic only triggers when
 * user code actually instantiates the launch path outside an nvcc-compiled TU.
 */
template <typename>
inline constexpr bool device_kernel_launch_requires_cudacc_ = false;

/**
 * Concrete @ref KernelImpl that runs the per-cell device callable produced by a
 * Neon-style loading lambda over every GPU in @c Grid::system().
 *
 * @tparam Grid       Layout type stored by value (e.g. @ref Dense). Must provide
 *                    @c gpu_count(), @c gpu_grid(view), @c gpu_block(view), and
 *                    @c launch_dense_grid(...). @c Dense friends this template so its
 *                    private @c launch_dense_grid is reachable.
 * @tparam LoadingFn  Callable @c (KernelLoader&) -> F where @c F satisfies
 *                    @c detail::dense::dense_kernel_fn at instantiation time.
 *                    No top-level @c requires clause — nvcc has historically had bugs
 *                    deducing @c std::invoke_result_t on extended lambdas.
 */
template <typename Grid, typename LoadingFn>
class DeviceKernelImpl final : public KernelImpl {
public:
  /**
   * @brief Stores a copy of @p g, the loading callable, the dynamic shared memory size,
   *        and an optional human-readable @p name.
   *
   * Self-construction (forwarding another @c DeviceKernelImpl as @p loading) is already
   * blocked by the @c const Grid& first parameter — a @c DeviceKernelImpl is not a
   * @c Grid — and copy/move on the base @ref KernelImpl are deleted, so no
   * @c requires-clause guard is needed here.
   *
   * @warning The loading callable is stored @b by-value and re-invoked on every
   * @c launch . Any state it captures must outlive this kernel — capture
   * @c Dense::Field and @c ReductionResult by value (their handles are cheap
   * ref-counted views) rather than by reference. See @ref KernelLoader for the
   * full lifetime contract.
   */
  template <typename L>
  DeviceKernelImpl(const Grid& g,
                   L&& loading,
                   size_t shared_mem,
                   std::string name)
      : KernelImpl(std::move(name)),
        grid_(g),
        loading_fn_(std::forward<L>(loading)),
        shared_mem_(shared_mem) {
    // Anchor the cross-grid contract documented in
    // include/carbon/detail/grid_concepts.h. A grid family whose Span /
    // Partition does not satisfy the matching concept lands on a clean
    // concept diagnostic here rather than a deep launch template
    // instantiation stack inside launch_dense_grid / launch_disg_grid /
    // launch_block_grid. Both branches of @c grid_has_dense_iteration_v
    // anchor their respective Span concept *and* the shared
    // @c PartitionConcept (closes P1.β of
    // @c docs/audit_2026_05_third_pass.md ). The PartitionConcept anchor
    // uses @c Grid::template Partition<int, 1> as the representative
    // @c (T, Components) — every grid family exposes the
    // @c template <typename T, int Components> using Partition = ...
    // alias, so this instantiation is well-formed for all three families.
    if constexpr (grid_has_dense_iteration_v<Grid>) {
      static_assert(DenseLikeSpanConcept<typename Grid::Span, typename Grid::Idx>,
                    "Grid::Span must satisfy carbon::detail::DenseLikeSpanConcept "
                    "(see include/carbon/detail/grid_concepts.h for the contract).");
      static_assert(PartitionConcept<typename Grid::template Partition<int, 1>, typename Grid::Span,
                                     typename Grid::Idx, int>,
                    "Grid::Partition<T, Components> must satisfy carbon::detail::"
                    "PartitionConcept (read / write / neighbour with .value + .valid). "
                    "See include/carbon/detail/grid_concepts.h.");
    } else {
      static_assert(BlockLikeSpanConcept<typename Grid::Span, typename Grid::Idx>,
                    "Grid::Span must satisfy carbon::detail::BlockLikeSpanConcept "
                    "(brick-iteration contract: iter_block_count(view), "
                    "view_block_idx, block_count, n_internal, set_and_validate(b, "
                    "ix, iy, iz), linear_offset(idx, components, c), "
                    "per_card_elements, linear_storage_elements(int), "
                    "global_location(idx)). See include/carbon/detail/grid_concepts.h.");
      static_assert(PartitionConcept<typename Grid::template Partition<int, 1>, typename Grid::Span,
                                     typename Grid::Idx, int>,
                    "Grid::Partition<T, Components> must satisfy carbon::detail::"
                    "PartitionConcept (read / write / neighbour with .value + .valid). "
                    "See include/carbon/detail/grid_concepts.h.");
    }
    // Anchor the loading-lambda shape early. KernelLoader is complete by the
    // time any DeviceKernelImpl instantiation runs (kernel.h is the only
    // entry point and defines KernelLoader before pulling this header). A
    // user who hands compute() the per-cell device lambda instead of the
    // outer loading lambda lands on this assert rather than a deep template
    // instantiation tree from discover_tokens_'s loading_fn_(ctx) call.
    static_assert(std::is_invocable_v<LoadingFn&, ::carbon::KernelLoader&>,
                  "Kernel::compute / Dense::make_kernel: the loading callable must be invocable "
                  "as `loader(KernelLoader&) -> F`, where F is the per-cell device lambda. The "
                  "common mistake is to hand the per-cell `[=] CARBON_HD (Span, Idx)` lambda "
                  "directly to compute(); wrap it in the loader-shape outer lambda. See the "
                  "KernelLoader doc comment in <carbon/kernel.h> for a worked example.");
    discover_tokens_();
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::Compute;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& override {
    return tokens_;
  }

  /**
   * @brief Device launch entry point.
   *
   * For each GPU @c i in @c grid.gpu_count(), invokes
   * @c loading_fn(KernelLoader{i}) and launches @c detail::dense::dense_grid_kernel using
   * @c grid.gpu_grid(view)[i] / @c grid.gpu_block(view)[i].
   *
   * The polymorphic @ref launch dispatches to this method (device is the natural
   * target for a device-only kernel); @ref launch_on_host throws — an explicit cast
   * to a host-capable kernel is the only way to run device-built kernels on CPU.
   *
   * @throws std::logic_error on grid / block / stream-size mismatches.
   * @throws std::runtime_error on CUDA failure.
   * @throws via @c static_assert when instantiated outside an nvcc TU.
   */
  auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void override {
    this->launch_on_device(stream, view);
  }

  [[nodiscard]] auto
  runs_on_device() const noexcept -> bool override {
    return true;
  }

  auto
  launch_on_device(const ::carbon::Stream& stream,
                   DataView view) -> void override {
#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
    using Traits = DeviceLaunchTraits<Grid>;
    const size_t n = Traits::preflight(grid_, view);
    if (n > 0 && stream.size() != n) {
      log::throw_with_log<std::logic_error>(
          "Kernel::launch: stream group size {} must equal gpu_count() {}", stream.size(), n);
    }
    // Fan the per-GPU launches out across one host thread per GPU (mirrors
    // Neon's DevSet::helpLaunchLambdaOnSpanCUDA). Each launch carries a
    // gpu_set_device + launch-config cost, so a single-threaded loop
    // serialises that host-side latency and pins the GPUs to a staircase
    // start. The `if (n > 1)` clause keeps the single-GPU case on the plain
    // serial path (no thread-team spawn). Errors are collected per GPU and
    // rethrown after the region — exceptions cannot cross an OpenMP boundary.
    std::vector<gpu_error_t> launch_errs(n, gpu_success);
#pragma omp parallel for num_threads(static_cast<int>(n)) if (n > 1)
    for (size_t i = 0; i < n; ++i) {
      KernelLoader ctx{i, KernelTarget::Device, KernelLoaderPhase::Launch};
      auto device_fn = loading_fn_(ctx);
      const gpu_stream_t cuda_stream = stream[i].get();
      launch_errs[i] = Traits::cuda_launch(grid_, i, shared_mem_, cuda_stream, view, device_fn);
    }
    for (size_t i = 0; i < n; ++i) {
      if (launch_errs[i] != gpu_success) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::launch (compute): per-GPU launch failed for relative_gpu_id={}: {} "
            "(gpu error {})",
            i, gpu_error_string(launch_errs[i]), static_cast<int>(launch_errs[i]));
      }
    }
#elif CARBON_BACKEND_IS_SYCL
    // SYCL launch: same orchestration shape as the CUDA branch (per-GPU loop, KernelLoader,
    // preflight + stream group sizing checks), but the per-iteration body calls into
    // DeviceLaunchTraits<Grid>::sycl_launch . That helper submits a q.parallel_for over an
    // nd_range mirroring the CUDA grid * block; any sycl::exception escaping the submission
    // is repackaged into std::runtime_error so the user-facing error path is consistent
    // with the CUDA backend.
    {
      using Traits = DeviceLaunchTraits<Grid>;
      const size_t n = Traits::preflight(grid_, view);
      if (n > 0 && stream.size() != n) {
        log::throw_with_log<std::logic_error>(
            "Kernel::launch: stream group size {} must equal gpu_count() {}", stream.size(), n);
      }
      for (size_t i = 0; i < n; ++i) {
        KernelLoader ctx{i, KernelTarget::Device, KernelLoaderPhase::Launch};
        auto device_fn = loading_fn_(ctx);
        Traits::sycl_launch(grid_, i, shared_mem_, stream[i].queue(), view, device_fn);
      }
    }
#else
    static_assert(device_kernel_launch_requires_cudacc_<LoadingFn>,
                  "Kernel::launch (compute) requires a translation unit compiled with nvcc "
                  "(__CUDACC__). Move the call site to a .cu file (or build that TU with nvcc) "
                  "and include <carbon/kernel.h> from there. Only the call site needs to be "
                  "CUDA — the rest of your project can stay on the host compiler.");
    (void)stream;
    (void)view;
#endif
  }

  /** Grid layout (e.g. @ref Dense), copied at construction. */
  [[nodiscard]] auto
  grid() const noexcept -> const Grid& {
    return grid_;
  }

  /** User loading callable. */
  [[nodiscard]] auto
  loading_fn() const noexcept -> const LoadingFn& {
    return loading_fn_;
  }

  /** Dynamic shared memory in bytes forwarded to every per-GPU kernel launch. */
  [[nodiscard]] auto
  shared_mem() const noexcept -> size_t {
    return shared_mem_;
  }

private:
  /**
   * @brief One-shot token discovery: invoke @c loading_fn_ on a throwaway
   *        @c KernelLoader (relative_gpu_id=0, target=Device) and snapshot the
   *        recorded tokens. The per-cell device callable returned by the lambda is
   *        immediately discarded — this pass exists only to populate @c tokens_ for
   *        the future @c Skeleton's dependency analysis.
   *
   * Skipped on empty grids (no GPUs to bind a loader against). Invoked exactly once
   * from the constructor; subsequent launches build their own loaders unaware of
   * this discovery pass.
   */
  auto
  discover_tokens_() -> void {
    if (grid_.gpu_count() == 0U) {
      return;
    }
    KernelLoader ctx{0U, KernelTarget::Device};
    [[maybe_unused]] auto fn = loading_fn_(ctx);
    tokens_ = ctx.tokens();
  }

  Grid grid_{};
  LoadingFn loading_fn_{};
  size_t shared_mem_{};
  std::vector<AccessToken> tokens_{};
};

}  // namespace detail
}  // namespace carbon
