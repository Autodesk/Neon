#pragma once

/**
 * Host (CPU / OpenMP) compute kernel concrete impl behind ::carbon::Kernel. Twin of
 * @ref carbon::detail::DeviceKernelImpl but driven by an OpenMP @c parallel for over the
 * iter extent of each @c Dense slab; written so the user can feed exactly the same
 * loading lambda (and the same per-cell @c (Span, Idx) -> void closure) it feeds to
 * @ref carbon::Kernel::compute / @c Dense::make_kernel — only the loader target
 * (@c loader.target() == KernelTarget::Host) and the iteration backend differ.
 *
 * Mirrors @c Neon::set::internal::HostContainer<DataContainerT, LoadingLambda>: a
 * templated subclass of the polymorphic kernel base that stores the Neon-style loading
 * callable and, at @ref launch time, walks every slab with three nested @c #pragma omp
 * parallel for loops (z slowest, x fastest) calling @c span.set_and_validate + the user
 * lambda. Implemented header-only because the lambda type is captured as a template
 * parameter on @ref DenseHostKernelImpl; instantiation costs go through the same
 * "include from the .cu factory call site" route as the device impl.
 *
 * @par Stream semantics
 * @ref launch waits on the @p stream first (per the user-approved phase-2 design): a
 * host kernel sitting in a heterogeneous @c std::vector<Kernel> after a device kernel
 * needs the device kernel's writes to have landed in the CPU mirror before the host
 * loop reads them. After that, the OpenMP loop runs synchronously inline — there is no
 * notion of a host stream. The same uniform @c Kernel::launch(stream) signature is
 * preserved so the heterogeneous-vector use case keeps working unchanged.
 *
 * @par MemorySide requirements
 * Every field touched by the loading lambda must have been built with
 * @c MemorySide::Host or @c MemorySide::HostDevice. The check happens when
 * @c loader(field) is invoked inside the loading lambda: @ref KernelLoader's host
 * branch throws @c std::invalid_argument with the offending field's empty
 * @c host_partitions vector. Device-only fields hit the same throw.
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense/span.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/detail/launch_traits.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace carbon {

struct KernelLoader;

namespace detail {

/**
 * Concrete @ref KernelImpl that runs the user's per-cell host callable over every slab
 * of @c Grid::system() via OpenMP.
 *
 * @tparam Grid       Layout type stored by value (e.g. @ref Dense). Must provide
 *                    @c gpu_count() / @c spans() (one @c detail::dense::Span per slab),
 *                    same as the device impl.
 * @tparam LoadingFn  Callable @c (KernelLoader&) -> F where @c F is invocable as
 *                    @c (Span, Idx) -> void. Returning a closure with state captured by
 *                    value is fine — OpenMP threads each invoke @c F by value through a
 *                    shared @c const reference, so the closure does not need to be
 *                    thread-local.
 */
template <typename Grid, typename LoadingFn>
class DenseHostKernelImpl final : public KernelImpl {
public:
  /**
   * @brief Stores a copy of @p g, the loading callable, and an optional human-readable
   *        @p name. The shared-memory parameter the device impl carries is intentionally
   *        absent — OpenMP has no concept of dynamic shared memory and the kernel runs
   *        in the host's per-thread stack instead.
   */
  template <typename L>
  DenseHostKernelImpl(const Grid& g,
                      L&& loading,
                      std::string name)
      : KernelImpl(std::move(name)),
        grid_(g),
        loading_fn_(std::forward<L>(loading)) {
    discover_tokens_();
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::ComputeHost;
  }

  [[nodiscard]] auto
  runs_on_host() const noexcept -> bool override {
    return true;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& override {
    return tokens_;
  }

  /**
   * @brief Polymorphic launch — host is the natural target for a host-only kernel,
   *        so @c launch forwards to @ref launch_on_host. @c launch_on_device throws via
   *        the base class default.
   */
  auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void override {
    this->launch_on_host(stream, view);
  }

  /**
   * @brief Host launch order:
   *   1. Wait on @p stream so any prior device kernel's writes are visible on the CPU
   *      mirror (the user is expected to have queued an explicit Transfer<T> from the
   *      device buffer to the host mirror before the host kernel; @p stream.wait() then
   *      blocks until that transfer completes).
   *   2. For each slab @c g, build a host loader (@c KernelLoader{g, KernelTarget::Host}),
   *      call the loading lambda once to get the per-cell host callable, and run a 3-deep
   *      OpenMP parallel-for over @c span.iter_extent(view) — z slowest, x fastest, same
   *      ordering as the CUDA / SYCL launch's flat-index decoder.
   * @throws std::runtime_error on @p stream.wait() failure or any exception escaping the
   *         OpenMP loop body.
   */
  auto
  launch_on_host(const ::carbon::Stream& stream,
                 DataView view) -> void override {
    try {
      stream.wait();
    } catch (const std::exception& ex) {
      log::throw_with_log<std::runtime_error>(
          "Kernel::compute_host::launch: stream.wait() before host loop failed: {}", ex.what());
    }

    const size_t n = grid_.gpu_count();
    if (n == 0U) {
      return;
    }

    // HostLaunchTraits picks the right span vector per family (Dense /
    // DenseDisg : grid.spans() ; Block : grid.host_spans() ) and runs the
    // per-slab OpenMP loop with the family's iteration shape (Dense /
    // DenseDisg : 3-deep @c collapse(3) over @c iter_extent ; Block : outer
    // brick walk + inner @c collapse(3) over @c MemX*MemY*MemZ ). See
    // @c detail/launch_traits.h for the full surface.
    using Traits = HostLaunchTraits<Grid>;
    const auto& spans = Traits::spans_for_host(grid_);
    if (spans.size() != n) {
      log::throw_with_log<std::logic_error>(
          "Kernel::compute_host::launch: spans size {} must equal gpu_count() {}", spans.size(), n);
    }

    // Loop slab-by-slab on the host: each slab's CPU mirror is independent of the
    // others (multi-GPU partitioning) so the outer loop is sequential and the inner
    // OpenMP parallel-for parallelises within a single slab. This matches Neon's
    // OpenMP path (one launch per slab; no cross-slab work-sharing needed because
    // halos are exchanged separately).
    for (size_t g = 0; g < n; ++g) {
      KernelLoader ctx{g, KernelTarget::Host, KernelLoaderPhase::Launch};
      auto host_fn = loading_fn_(ctx);
      Traits::iterate_slab(spans[g], view, host_fn);
    }
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

private:
  /**
   * @brief One-shot token discovery — twin of @ref DeviceKernelImpl::discover_tokens_ .
   *        Uses @c KernelTarget::Device for the discovery loader (the resulting
   *        tokens are target-agnostic — they carry only @c field_uid + @c AccessMode )
   *        so a HostDevice field's lazy CPU-mirror allocation is not forced just to
   *        learn which fields the kernel touches.
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
  std::vector<AccessToken> tokens_{};
};

}  // namespace detail
}  // namespace carbon
