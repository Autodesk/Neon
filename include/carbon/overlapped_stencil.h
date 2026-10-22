#pragma once

/**
 * @file overlapped_stencil.h
 *
 * Two-stream halo/compute overlap orchestrator. Wraps the canonical pattern that
 * the Internal / Boundary @ref DataView split exists for: run the halo exchange on
 * one stream while the @c Internal compute kernel runs concurrently on a second
 * stream, then fence the @c Boundary compute behind the halo so it sees the
 * up-to-date halo cells.
 *
 * @par Why a primitive?
 * The mechanics — two streams, two events, "skip the halo<-compute wait on the
 * first iteration" — are mechanical bookkeeping, not domain logic, and were
 * already duplicated in @c tests/test_data_view_split_jacobi.cu and would land
 * verbatim in every overlapped Jacobi / Stokes / FV / LBM driver. @ref
 * OverlappedStencil hides the bookkeeping behind one @ref step call so user code
 * looks like:
 *
 * @code
 * carbon::OverlappedStencil overlap(sys);
 * for (int it = 0; it < kIters; ++it) {
 *   const auto& halo_in = (it % 2 == 0) ? halo_a : halo_b;  // ping-pong halos
 *   const auto& step_k  = (it % 2 == 0) ? step_ab : step_ba;
 *   overlap.step(halo_in, step_k);
 * }
 * overlap.wait();
 * @endcode
 *
 * @par What @ref step does (per iteration)
 * 1. If not the first call: @c halo_stream waits for the previous iteration's
 *    @c Boundary launch (via @c evt_compute_) — the previous iteration wrote
 *    owned cells of the field this iteration's halo reads from, so the halo
 *    can't start until that's flushed.
 * 2. @c halo_kernel launches on @c halo_stream.
 * 3. @c compute_kernel launches on @c compute_stream with @c DataView::Internal —
 *    independent of the halo (Internal cells don't read halos), runs
 *    concurrently with the halo exchange.
 * 4. @c evt_halo records on @c halo_stream after the halo exchange.
 * 5. @c compute_stream waits for @c evt_halo — Boundary reads halo cells, so it
 *    needs the halo to be done.
 * 6. @c compute_kernel launches on @c compute_stream with @c DataView::Boundary.
 * 7. @c evt_compute records on @c compute_stream after Boundary.
 *
 * After @ref step returns the host has issued every command but no GPU work has
 * necessarily completed yet — both streams continue executing asynchronously.
 * Call @ref wait to block until both streams drain (or schedule the next
 * @ref step which only re-fences against @c evt_compute internally).
 *
 * @par Single-GPU and small layouts
 * The @c Halo kernel is a no-op when there are no adjacent local pairs and no
 * cross-rank pairs (single-GPU non-distributed runs, or @c halo_z == 0
 * stencils). Internal / Boundary still iterate the right cell sets; on those
 * layouts the overlap collapses to "Internal then Boundary on @c compute_stream
 * with a do-nothing halo on @c halo_stream", which is functionally identical to
 * a single Standard launch on one stream. The orchestrator stays correct — it
 * just doesn't recover any time.
 *
 * @par Stream / event ownership
 * The default constructor allocates two streams and two events from the @c
 * System (so they participate in the @c System 's stream / event history). The
 * advanced overload accepts user-provided streams / events for callers that
 * already manage their own; both must come from the same @c System . Streams
 * and events are shallow handles — copying is cheap and the underlying @c
 * StreamGroup / @c EventGroup is shared.
 *
 * @par Thread safety
 * @ref step mutates the "first call" flag and re-records both events; not
 * thread-safe. Carbon's other types (@c Stream , @c Event , @c Kernel ) follow
 * the same convention.
 */

#include <carbon/data_view.h>
#include <carbon/kernel.h>
#include <carbon/system.h>

#include <utility>

namespace carbon {

/**
 * @brief Two-stream halo/compute overlap helper for the @ref DataView Internal/Boundary split.
 *
 * Owns two @ref Stream handles (compute + halo) and two @ref Event handles, and
 * issues one overlapped iteration per call to @ref step . See the file-level
 * documentation for the full per-iteration sequence and the rationale.
 */
class OverlappedStencil {
public:
  /**
   * @brief Allocate dedicated streams and events from @p sys .
   *
   * Calls @c sys.stream() twice and @c sys.event() twice, so the four handles
   * are recorded in the @c System 's stream / event history (debuggable through
   * @c System::stored_streams() / @c stored_events() ).
   */
  explicit OverlappedStencil(System sys)
      : sys_(std::move(sys)),
        compute_stream_(sys_.stream()),
        halo_stream_(sys_.stream()),
        evt_halo_(sys_.event()),
        evt_compute_(sys_.event()) {}

  /**
   * @brief Adopt user-provided streams + events.
   *
   * Use when the caller already owns streams it wants the overlap to share
   * with (e.g. an outer pipeline that runs unrelated work on @p compute_stream
   * before / after the overlapped loop). All four handles must come from the
   * same @p sys ; carbon does not check this at construction (mismatched
   * handles will fail later in the underlying @c StreamGroup / @c EventGroup
   * primitives with a clean diagnostic).
   */
  OverlappedStencil(System sys,
                    Stream compute_stream,
                    Stream halo_stream,
                    Event evt_halo,
                    Event evt_compute) noexcept
      : sys_(std::move(sys)),
        compute_stream_(std::move(compute_stream)),
        halo_stream_(std::move(halo_stream)),
        evt_halo_(std::move(evt_halo)),
        evt_compute_(std::move(evt_compute)) {}

  /**
   * @brief Issue one overlapped iteration on the orchestrator's two streams.
   *
   * @p halo_kernel is launched on @c halo_stream() (the @ref DataView argument
   * is ignored for halo kernels — see @c HaloKernelImpl::launch ).
   * @p compute_kernel is launched twice on @c compute_stream() , once with
   * @c DataView::Internal and once with @c DataView::Boundary , fenced behind
   * @p halo_kernel via @c evt_halo_ .
   *
   * Both arguments are taken by const reference; @ref Kernel is a value
   * handle, so the user can keep them in any container and pass them in
   * round-robin for ping-pong inputs.
   *
   * @throws std::runtime_error from the underlying stream / event / kernel
   *         primitives on submission failure (forwarded unmodified).
   */
  auto
  step(const Kernel& halo_kernel,
       const Kernel& compute_kernel) -> void {
    // Full N×N fan-in: every halo stream waits for compute on every GPU,
    // not just the per-GPU compute event. The halo kernel issues
    // cudaMemcpyPeerAsync from source GPU memory to destination GPU
    // memory on the destination GPU's halo stream — the destination-side
    // queue gives no implicit ordering against pending writes on the
    // source GPU's compute stream. Without this cross-GPU fence the halo
    // can read transient (or stale) bytes from the source while iter
    // k-1's Boundary kernel is still flushing to source memory, producing
    // a small but non-zero drift over multi-iteration runs. (Same gotcha
    // is documented on `carbon::Halo`'s "Synchronisation contract".) See
    // `StreamGroup::wait_for_all` for the full discussion.
    //
    // This fence runs every iteration, including iter 0 and the first
    // iteration after @ref reset . On a "really" first iteration
    // (overlap.compute_stream() has never had work issued on it) the
    // event has nothing recorded against it yet; cudaStreamWaitEvent on
    // a never-recorded event is a clean no-op per CUDA docs ("if the
    // event has not been recorded, the wait operation will not wait —
    // it returns immediately as a no-op"). After @ref reset
    // evt_compute_ still carries pass-k's last record, which is
    // already signalled (the user must call @c wait() before @c reset
    // per the documented contract), so the wait is again a no-op.
    // first_ is kept on the type for diagnostic / observation purposes
    // and to document the iter-0 contract; it's no longer used to gate
    // the fence.
    halo_stream_.wait_for_all(evt_compute_);
    halo_kernel.launch(halo_stream_, DataView::Standard);
    compute_kernel.launch(compute_stream_, DataView::Internal);
    evt_halo_.record(halo_stream_);
    compute_stream_.wait_for(evt_halo_);
    compute_kernel.launch(compute_stream_, DataView::Boundary);
    evt_compute_.record(compute_stream_);
    first_ = false;
  }

  /**
   * @brief Block until both streams drain.
   *
   * Equivalent to @c compute_stream().wait() followed by
   * @c halo_stream().wait() . Forwards the first failure (if any) and
   * still drains the second stream before throwing, matching
   * @c StreamGroup::wait 's "best-effort drain" contract.
   */
  auto
  wait() -> void {
    // Match StreamGroup::wait's drain order: try compute first (the longer-running
    // queue in a real overlap pattern), then halo. If compute throws, halo is
    // still waited on before we rethrow.
    std::exception_ptr first_err{};
    try {
      compute_stream_.wait();
    } catch (...) {
      first_err = std::current_exception();
    }
    try {
      halo_stream_.wait();
    } catch (...) {
      if (!first_err) {
        first_err = std::current_exception();
      }
    }
    if (first_err) {
      std::rethrow_exception(first_err);
    }
  }

  /**
   * @brief Reset the "first iteration" flag so the next @ref step skips the
   *        @c halo<-compute wait again.
   *
   * Use after @ref wait when starting a logically separate sequence of
   * iterations (e.g. an outer adaptive solver that re-initialises the field).
   * No-op if the orchestrator has not yet been used.
   */
  auto
  reset() noexcept -> void {
    first_ = true;
  }

  /** Stream the compute kernel runs on (Internal + Boundary). */
  [[nodiscard]] auto
  compute_stream() const noexcept -> const Stream& {
    return compute_stream_;
  }

  /** Stream the halo kernel runs on. */
  [[nodiscard]] auto
  halo_stream() const noexcept -> const Stream& {
    return halo_stream_;
  }

  /** Event recorded on @c halo_stream() after every halo launch. */
  [[nodiscard]] auto
  halo_event() const noexcept -> const Event& {
    return evt_halo_;
  }

  /**
   * @brief Event recorded on @c compute_stream() after every Boundary launch.
   *
   * Useful for chaining downstream work after the iteration loop: a download
   * @c Transfer issued on @c compute_stream() can stream after the last
   * @ref step naturally; an unrelated stream that needs to wait for compute
   * can call @c wait_for(overlap.compute_event()) .
   */
  [[nodiscard]] auto
  compute_event() const noexcept -> const Event& {
    return evt_compute_;
  }

  /** @c System the orchestrator is bound to. */
  [[nodiscard]] auto
  system() const noexcept -> const System& {
    return sys_;
  }

private:
  System sys_{};
  Stream compute_stream_{};
  Stream halo_stream_{};
  Event evt_halo_{};
  Event evt_compute_{};
  // Iter-0 marker — true until the first @ref step has run. The current
  // step() implementation always waits on @c evt_compute_ (an in-comment
  // analysis at the wait site explains why the wait is a no-op on iter-0
  // anyway), so this flag is no longer used to gate the fence. Kept on the
  // type for diagnostic / observation purposes and to document the iter-0
  // contract; @ref reset re-arms it so the same orchestrator can drive
  // multiple disjoint loops.
  bool first_{true};
};

}  // namespace carbon
