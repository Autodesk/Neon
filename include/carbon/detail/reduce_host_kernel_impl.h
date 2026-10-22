#pragma once

/**
 * Host (CPU / OpenMP) reduction concrete impl behind ::carbon::Kernel. Twin of
 * @ref carbon::detail::ReduceKernelImpl that runs an OpenMP parallel-for with thread-
 * local partials over the iter extent of @c DataView::Standard, then combines the
 * partials sequentially through the same @c Op the device path uses. Result lands
 * directly in @c ReductionResult<T>::host_data — no device buffer involved, no async
 * memcpy, no @c sync needed (calling @c sync after @c launch is still safe; it just
 * waits on the empty stream and is then a no-op).
 *
 * @par Why thread-local + sequential combine instead of @c #pragma omp reduction
 * @c omp reduction(op: var) requires @c op to be a name OpenMP knows (sum, prod, min,
 * max, ...). Carbon's @c reduce::Sum<T> is a user-supplied functor and the public API
 * lets callers plug arbitrary commutative-associative operators. We therefore allocate
 * one @c T per thread, run the parallel-for, and combine via a sequential
 * @c std::accumulate-style loop using @c Op. The combine step is O(num_threads) — even
 * on a 256-core machine it's a few hundred ns and fully amortised by the parallel-for
 * over millions of cells.
 *
 * @par Scope (matches the device impl)
 * - Multi-GPU and multi-rank are both supported. Each local slab runs its own
 *   OpenMP parallel-for and writes a per-GPU partial; stage 2 sums those partials
 *   sequentially on the rank-local root (the host mirror plays the role of the
 *   intra-process combine kernel), and stage 3 (cross-rank @c MPI_Allreduce when
 *   @c System::distributed is set) runs through @c detail/reduce_stages.h .
 * - Any @ref DataView is accepted on @c launch_on_host . The slab is iterated using
 *   @c span.iter_extent(view) and @c set_and_validate(view, ...) ; an Internal +
 *   Boundary pair recombines to the Standard total via
 *   @c ReductionResult::host_value_combined .
 *
 * @par Stream semantics
 * Same as @ref DenseHostKernelImpl: @c stream.wait() first (so the host reduction sees
 * a flushed CPU mirror), then run the OpenMP loop synchronously, then write the final
 * scalar to @c dst.host_data(view) so @c dst.host_value(view) returns it immediately.
 */

#include <carbon/data_view.h>
#include <carbon/detail/dense/span.h>
#include <carbon/detail/kernel_impl.h>
#include <carbon/detail/launch_traits.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/reduce_dist.h>
#include <carbon/system.h>

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace carbon {

struct KernelLoader;

namespace detail {

/**
 * Concrete @ref KernelImpl for the OpenMP grid-wide reduction.
 *
 * @tparam Grid       Layout type (e.g. @ref Dense).
 * @tparam LoadingFn  Callable @c (KernelLoader&) -> F where @c F is invocable as
 *                    @c (Span, Idx) -> T returning the per-cell contribution.
 * @tparam T          Reduction value type.
 * @tparam Op         Reduction operator (commutative-associative, callable as
 *                    @c op(a, b) -> T, with @c identity() returning the identity element).
 */
template <typename Grid, typename LoadingFn, typename T, typename Op>
class ReduceHostKernelImpl final : public KernelImpl {
public:
  template <typename L>
  ReduceHostKernelImpl(const Grid& g,
                       L&& loading,
                       ReductionResult<T>& dst,
                       Op op,
                       std::string name)
      : KernelImpl(std::move(name)),
        grid_(g),
        loading_fn_(std::forward<L>(loading)),
        dst_(&dst),
        op_(std::move(op)) {
    const auto& gpu_group = grid_.system().gpu_group();
    if (gpu_group.empty()) {
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce_host: System has an empty gpu_group — host reduce still needs at "
          "least one slab (the slab carries the iteration extent / Span)");
    }
    if (dst_->gpu_count() != gpu_group.size()) {
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce_host: ReductionResult was built on a System with {} GPU(s) but the "
          "Grid has {} — the destination's per-GPU partials must match the grid's slabs",
          dst_->gpu_count(), gpu_group.size());
    }

    // total_cells_ is the Standard-view sum of every slab's iteration "cell" count
    // (HostLaunchTraits absorbs the per-family semantic: Dense / DenseDisg count
    // cells via iter_extent.volume(); Block counts retained bricks via
    // iter_block_count). Used for diagnostic accessors only — the launch path
    // recomputes per-slab counts via the trait per DataView.
    using SizingTraits = ReduceLaunchTraits<Grid>;
    total_cells_ = 0;
    for (size_t g_idx = 0; g_idx < gpu_group.size(); ++g_idx) {
      const int64_t v = SizingTraits::iter_cell_count(grid_, g_idx, DataView::Standard);
      if (v < 0) {
        log::throw_with_log<std::runtime_error>(
            "Kernel::reduce_host: negative slab[{}] iter cell count ({}) — overflow", g_idx, v);
      }
      total_cells_ += v;
    }
    discover_tokens_();
  }

  [[nodiscard]] auto
  kind() const noexcept -> KernelKind override {
    return KernelKind::ReductionHost;
  }

  [[nodiscard]] auto
  tokens() const noexcept -> const std::vector<AccessToken>& override {
    return tokens_;
  }

  /** Total cells iterated by the reduction; matches the device impl's accessor. */
  [[nodiscard]] auto
  total_cells() const noexcept -> int64_t {
    return total_cells_;
  }

  [[nodiscard]] auto
  runs_on_host() const noexcept -> bool override {
    return true;
  }

  /**
   * @brief Polymorphic launch — host is the natural target for a host-only reduce
   *        kernel, so @c launch forwards to @ref launch_on_host. @c launch_on_device
   *        throws via the base class default.
   */
  auto
  launch(const ::carbon::Stream& stream,
         DataView view) -> void override {
    this->launch_on_host(stream, view);
  }

  /**
   * @brief Launch order:
   *   1. Validate the view (Standard only) and stream group size (1).
   *   2. @c stream.wait() so a prior device kernel's writes are visible on the host.
   *   3. Resolve the per-cell host callable through @ref KernelLoader.
   *   4. Allocate one @c T accumulator per OpenMP thread (init = @c op_.identity()),
   *      run the parallel-for, combine the per-thread partials sequentially.
   *   5. Write the final scalar to @c dst->host_data()[0]. Skip the device scalar
   *      entirely (it's still allocated by @c ReductionResult's ctor but the host path
   *      never touches it; calling @c dst.sync(stream) afterwards is a no-op).
   */
  auto
  launch_on_host(const ::carbon::Stream& stream,
                 DataView view) -> void override {
    if (stream.size() != grid_.gpu_count()) {
      log::throw_with_log<std::invalid_argument>(
          "Kernel::reduce_host::launch: stream-group size ({}) does not match the grid's "
          "GPU count ({})",
          stream.size(), grid_.gpu_count());
    }
    try {
      stream.wait();
    } catch (const std::exception& ex) {
      log::throw_with_log<std::runtime_error>(
          "Kernel::reduce_host::launch: stream.wait() before host loop failed: {}", ex.what());
    }

    ReductionResult<T>& dst = *dst_;
    const size_t gpu_count = grid_.gpu_count();

    // Per-slab final scalars accumulated below, then folded across slabs into a single
    // rank-local result. Sized to gpu_count so a single-slab system goes through the
    // same code path with one entry (no special-case branch). HostLaunchTraits absorbs
    // the per-family iteration shape (Dense / DenseDisg : 3-deep cell walk over
    // @c iter_extent ; Block : 4-deep brick + voxel walk over @c iter_block_count
    // and @c MemX*MemY*MemZ ) and the host-spans-vs-spans choice.
    using Traits = HostLaunchTraits<Grid>;
    std::vector<T> slab_results(gpu_count, op_.identity());
    const auto& host_spans = Traits::spans_for_host(grid_);
    if (host_spans.size() != gpu_count) {
      log::throw_with_log<std::logic_error>(
          "Kernel::reduce_host::launch: host_spans size {} must equal gpu_count() {}",
          host_spans.size(), gpu_count);
    }

    for (size_t g_idx = 0; g_idx < gpu_count; ++g_idx) {
      KernelLoader ctx{g_idx, KernelTarget::Host, KernelLoaderPhase::Launch};
      auto host_fn = loading_fn_(ctx);
      slab_results[g_idx] =
          Traits::template iterate_slab_reduce<std::remove_cvref_t<decltype(host_spans[g_idx])>,
                                               decltype(host_fn), T, Op>(host_spans[g_idx], view,
                                                                         host_fn, op_);
    }

    // Fold across slabs into the rank-local final scalar. Op is required to be
    // associative + commutative, so this matches the device path's gpu_count == 1
    // shortcut for the common case.
    T result = op_.identity();
    for (const T& p : slab_results) {
      result = op_(result, p);
    }
    dst.host_data(view)[0] = result;

    // Cross-rank fold: when the System carries a Distributed handle, hand the rank-local
    // scalar to MPI_Allreduce so every rank ends up with the same value. The
    // if-constexpr branch keeps the host kernel free of MPI symbols when the (Op, T)
    // pair has no NCCL / MPI mapping (see reduce_dist.h policy); the runtime throw
    // mirrors the device path's diagnostic.
    run_stage3_cross_rank_host_(dst, view);
    dst.mark_fresh(view);
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

  /** Reduction operator (held by value). */
  [[nodiscard]] auto
  op() const noexcept -> const Op& {
    return op_;
  }

private:
  /**
   * @brief Cross-rank fold for the host reduce path: in-place @c MPI_Allreduce on
   *        @c dst.host_data(view) when the System has a @c Distributed handle.
   *
   * Sibling of @ref ReduceKernelImpl::run_stage3_cross_rank_ but on the CPU side: the
   * device path uses NCCL on the rank-local GPU scalar; the host path uses MPI on the
   * rank-local CPU scalar. The trait gate keeps non-NCCL/MPI-mappable @c (Op, T) pairs
   * out of the dispatch (a custom @c reduce::CountingMin<int8_t> with no @c op_kind
   * specialisation can still be reduced rank-locally on a single-process @c System ;
   * it just throws when a @c Distributed handle is present).
   */
  auto
  run_stage3_cross_rank_host_(ReductionResult<T>& dst,
                              DataView view) -> void {
#ifdef CARBON_HAVE_MPI_NCCL
    const ::carbon::System& sys = grid_.system();
    if (sys.distributed() == nullptr) {
      return;
    }
    if constexpr (::carbon::reduce::has_op_kind<Op> && ::carbon::reduce::has_data_kind<T>) {
      ::carbon::reduce::dist_host_allreduce_in_place(sys, dst.host_data(view), /*count=*/1U,
                                                     ::carbon::reduce::op_kind<Op>::value,
                                                     ::carbon::reduce::data_kind<T>::value);
    } else {
      log::throw_with_log<std::runtime_error>(
          "Kernel::reduce_host: cross-rank reduction requested but no MPI mapping is "
          "registered for this (Op, T) pair (specialise carbon::reduce::op_kind / "
          "carbon::reduce::data_kind, or run on a single-process System)");
    }
#else
    (void)dst;
    (void)view;
#endif
  }

  /** One-shot token discovery — same shape as @ref DeviceKernelImpl::discover_tokens_ . */
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
  ReductionResult<T>* dst_{};
  Op op_{};
  int64_t total_cells_{};
  std::vector<AccessToken> tokens_{};
};

}  // namespace detail
}  // namespace carbon
