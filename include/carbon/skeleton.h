#pragma once

/**
 * @file skeleton.h
 * @brief @c carbon::Skeleton — graph-execution layer over a heterogeneous
 *        @c std::vector<Kernel>.
 *
 * Mirrors Neon's @c Skeleton + @c MultiXpuGraph: the user appends kernels in the
 * order they want them logically applied, the skeleton walks each kernel's
 * @c tokens() (the @c (field_uid, AccessMode) records collected by
 * @ref KernelLoader at kernel construction) and builds a dependency DAG, then
 * lowers that DAG to a static, multi-stream schedule the @ref run method replays
 * on every iteration.
 *
 * @par v2 scope
 * - **Token-driven dependency analysis.** Every adjacent pair (in user-add
 *   order) is tested for token conflict via @c carbon::conflicts() ; each
 *   conflicting pair contributes an edge in the @ref edges adjacency list.
 *   Read-Read pairs on the same field do not conflict, so two kernels reading
 *   the same field can be parallelised.
 * - **Transitive reduction.** After the conflict scan, @ref compile derives
 *   @ref min_edges — the unique minimum DAG covering the same partial order
 *   as @ref edges . An edge @c i→j is kept only when no longer path
 *   @c i→…→j exists; redundant edges (e.g. @c 0→3 in the canonical Jacobi
 *   ping-pong, which is implied by @c 0→1→2→3 ) drop out. The reduction is
 *   the right input for downstream scheduling decisions: extra streams and
 *   cross-stream events only need to honour the minimal edge set.
 *   @ref edges remains the full conflict graph
 *   so callers (and tests) keep an unambiguous view of every detected
 *   conflict.
 * - **Greedy multi-stream assignment.** Kernels are mapped to streams by walking
 *   in user-add order: a kernel goes on a stream whose tail is a direct
 *   predecessor (free FIFO ordering, no event needed); else on the lowest-
 *   indexed empty stream up to the cap; else on stream 0 (and pays the
 *   cross-stream event). Default cap is 4 — enough for the typical
 *   stencil/reduce/IO/halo overlap, configurable via the constructor.
 * - **Cross-stream events for in-iteration deps.** For every DAG edge whose
 *   endpoints land on different streams, the predecessor records an event
 *   after its launch and the successor's stream waits for it before launching.
 *   Events are pooled — one event per kernel that has at least one
 *   cross-stream successor (sharing the same record point across multiple
 *   downstream consumers).
 * - **Anchor fork-join around every run.** Multi-stream pipelines lose the
 *   implicit FIFO ordering across @ref run calls — kernel @c A on stream 0
 *   in iteration @c k+1 may need data written by kernel @c B on stream 1 in
 *   iteration @c k . Stream 0 is the System's shared main stream
 *   (@c System::default_stream ) and @ref run brackets its launches with a
 *   "start" event recorded there that every secondary stream waits for, plus
 *   a "join" wait on every secondary stream's tail. So no work can leak
 *   across a run boundary: iteration @c k+1 on any stream begins only after
 *   iteration @c k drained on every stream, and — because the anchor is
 *   shared — so does the first iteration of a @em different skeleton bound to
 *   the same @c System (which is what lets an alternating ping-pong pair run
 *   without a host drain). Intra-iteration overlap is untouched: the bracket
 *   spans the whole iteration, not individual kernels. The single-stream case
 *   (all kernels on stream 0) allocates no events and issues no waits — FIFO
 *   on the anchor already provides both orderings — so the v1 fast path is
 *   preserved bit-for-bit.
 * - **Static schedule.** @c compile is idempotent (@c run calls it lazily on
 *   first invocation; a subsequent @ref add invalidates the schedule and
 *   forces a recompile on the next @c run ). @c reset drops both kernels and
 *   the schedule.
 *
 * @par Out of scope (deferred follow-ups)
 * - Automatic Internal/Boundary auto-overlap (still delegated to
 *   @ref OverlappedStencil today; the skeleton can call its @c step as a
 *   single node).
 * - Cross-rank coordination — @c Halo and @c DistTransfer already handle
 *   that inside the kernel; the skeleton just sequences halo nodes alongside
 *   compute.
 * - Critical-path-aware list scheduling (today the algorithm is greedy, so
 *   a deeper pipeline with branchier dependencies might pick a sub-optimal
 *   stream assignment; the anchor fork-join is also coarser than strictly
 *   necessary — it serialises whole iterations rather than only the stream
 *   pairs that actually conflict across the run boundary).
 *
 * Example (parallel-sibling pipeline):
 * @code
 * carbon::Skeleton sk(sys);
 * sk.add(dense.make_kernel(uv_loader, ""));   // A: u -> v
 * sk.add(dense.make_kernel(xy_loader, ""));   // B: x -> y  (no shared field with A)
 * sk.add(dense.make_kernel(combine_loader, ""));  // C: u, v, x, y -> z
 * sk.compile();
 * // sk.kernel_streams() == { 0, 1, 0 } — A on stream 0, B on stream 1, C on
 * // stream 0 with a cross-stream wait_for(event after B) before launch.
 * for (int it = 0; it < n_iters; ++it) sk.run();
 * sk.wait();
 * @endcode
 */

#include <carbon/access_token.h>
#include <carbon/data_view.h>
#include <carbon/detail/graph_replay.h>
#include <carbon/export.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/system.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace carbon {

namespace detail {
struct StepGraphNode;
}  // namespace detail

/**
 * @brief Overlap-of-Compute-and-Communication strategy applied by
 *        @ref Skeleton::compile around stencil reads.
 *
 * Mirrors @c Neon::skeleton::Occ from the
 * <a href="https://escholarship.org/uc/item/9fz7k633">Neon paper (Sec. V.B,
 * Fig. 4)</a>. Each mode is a graph rewrite around stencil nodes and their
 * mandatory halo updates: rather than serialising halo and stencil, the
 * skeleton splits one or more kernels into Internal + Boundary halves and
 * runs the Internal halves in parallel with the halo exchange. Carbon's
 * @c DataView::Internal / @c DataView::Boundary already partition the cell
 * iteration set; the rewrite drives launches with these views automatically.
 *
 * Rules:
 *
 *   - @c None — fully serialised. Halo update is a barrier before the
 *     stencil; every kernel runs with @c DataView::Standard . Default —
 *     matches a manually-built halo+compute pipeline byte-for-byte.
 *   - @c Standard — split @b only the stencil node into Internal + Boundary.
 *     The Internal launch overlaps with the halo; the Boundary launch fences
 *     after the halo completes. (Equivalent to a manually-wired
 *     @c OverlappedStencil for one (halo, stencil) pair.) Map / Reduce
 *     nodes preceding or following the stencil are unchanged.
 *   - @c Extended — Standard plus the split is propagated @b backward to
 *     every @c Map node whose output transitively feeds the stencil's
 *     input. Each such node runs Boundary first (so its boundary cells are
 *     ready for the halo to send) then Internal (in parallel with the halo).
 *     Reduce nodes stop the backward propagation.
 *   - @c TwoWayExtended — Extended plus a forward propagation: every @c Map
 *     Compute node whose input transitively reads from the stencil's output is
 *     also split (Internal first, then Boundary). Map-pattern @c Reduction
 *     successors are split into Partial Internal + Partial Boundary + Merge so
 *     the Internal reduce can overlap the halo (Neon Two-way Extended OCC).
 *     Stencil-pattern reductions are their own OCC pairs and stop this walk;
 *     other non-Compute kinds likewise halt forward propagation.
 *
 * Universal (dual-target) and Host (OpenMP) compute kernels are not
 * split today — only @c KernelKind::Compute (and Map @c Reduction under
 * TwoWayExtended) participate. A non-Compute non-Reduction kernel
 * encountered during the chain walk halts the propagation at that boundary;
 * the rest of the OCC rewrite still applies.
 */
enum class OverlapMode : unsigned char {
  None = 0,
  Standard = 1,
  Extended = 2,
  TwoWayExtended = 3,
};

/**
 * @brief Dependency-aware sequencer for a heterogeneous pipeline of @c Kernel s.
 *
 * Holds the kernels by value (each is a cheap @c shared_ptr handle), runs them
 * across one or more carbon @ref Stream s allocated on demand by @ref compile ,
 * and validates that the user's add order is a valid topological order of the
 * inferred dependency DAG. The default execution stream is the System's shared
 * anchor stream @c sys.default_stream() ; additional streams are allocated lazily inside @ref compile
 * whenever the dependency graph admits parallelism (no compile-time cap —
 * carbon trusts the graph shape to bound the count).
 *
 * @par Lifetime
 * The @c Skeleton owns its kernel list. Each @c Kernel keeps its underlying impl
 * alive (and through it, the captured grid / fields / halos). Destroying the
 * skeleton releases that ownership; the underlying @c System / @c Stream stay
 * alive through their own shared-handle semantics.
 */
class CARBON_API Skeleton {
public:
  /**
   * @brief Build-time options for the schedule produced by @ref compile .
   *
   * - @ref overlap_mode controls the OCC graph rewrite around stencil nodes
   *   (see @ref OverlapMode ). Default @c None matches the v2 behaviour
   *   bit-for-bit.
   * - @ref strict_halo flips the auto-halo-insertion pass from "silently
   *   insert" (default) to "throw at @c compile time". Useful for users who
   *   want to enforce that every halo is explicit at the call site (Neon's
   *   default before Containers gained automatic halo derivation).
   * - @ref max_streams caps the greedy stream-assignment pass at the given
   *   stream count (0 = no cap, the default). Useful on resource-
   *   constrained boxes where allocating one CUDA stream per parallel-
   *   sibling pair is wasteful; the OCC rewrite will still allocate halo
   *   streams beyond the cap because they're load-bearing (the rewrite
   *   needs a dedicated halo stream to overlap with), and any cap below
   *   the rewrite's needs simply does nothing for those streams. Stream
   *   reuse across @ref reset cycles is independent of this knob.
   * - @ref log_dot_on_compile dumps @ref export_dot through
   *   @c CARBON_LOG_INFO right after every successful @ref compile . Cheap
   *   debug aid for "did the schedule come out as I expected" checks
   *   without having to call @ref export_dot from user code.
   */
  struct Options {
    OverlapMode overlap_mode;
    bool strict_halo;
    size_t max_streams;
    bool log_dot_on_compile;
    // Opt in to CUDA/HIP graph capture + replay of the compiled schedule (see
    // @ref run ). Off by default. Engaged when the schedule is single-*Skeleton*-
    // stream (OverlapMode::None), intra-process (not distributed), free of host
    // kernels, and the backend supports capture. The single GPU behind the stream
    // is captured thread-locally; multiple GPUs are captured together via the
    // global-mode fork/join origin-stream pattern (the halo's cross-device peer
    // copies + event fences fold into one unified cross-device graph). Otherwise
    // the eager launch path is used transparently. Safe to leave on — it never
    // changes results, only how the launches are issued.
    bool use_cuda_graph;
    /** When @c High, streams carrying @c KernelKind::Halo steps use greatest driver priority. */
    StreamPriority halo_stream_priority{StreamPriority::Normal};

    constexpr Options() noexcept
        : overlap_mode(OverlapMode::None),
          strict_halo(false),
          max_streams(0U),
          log_dot_on_compile(false),
          use_cuda_graph(false),
          halo_stream_priority(StreamPriority::Normal) {}
    constexpr Options(OverlapMode mode,
                      bool strict = false,
                      size_t max_streams_cap = 0U,
                      bool dot_on_compile = false,
                      bool cuda_graph = false) noexcept
        : overlap_mode(mode),
          strict_halo(strict),
          max_streams(max_streams_cap),
          log_dot_on_compile(dot_on_compile),
          use_cuda_graph(cuda_graph) {}
  };

  /**
   * @brief Build an empty skeleton bound to @p sys . Stream 0 — the **anchor
   *        stream** — is @c sys.default_stream() ; additional streams are allocated
   *        lazily during @ref compile when the dependency graph admits parallelism
   *        (no compile-time cap).
   *
   * Anchoring on the System's shared main stream (rather than a private
   * @c sys.stream() ) is what orders *separate* skeletons against each other: two
   * skeletons built on the same @c System issue their stream-0 work into the same
   * FIFO, and @ref run brackets each iteration with a start/end fork-join on it, so
   * an alternating ping-pong pair (`sk_ab` / `sk_ba`) is correctly sequenced without
   * a host-side drain between iterations. Pass an explicit stream (next overload) to
   * opt out.
   */
  explicit Skeleton(System sys,
                    Options options = {})
      : sys_(std::move(sys)),
        options_(options) {
    streams_.push_back(sys_.default_stream());
  }

  /**
   * @brief Build an empty skeleton bound to @p sys with a caller-supplied
   *        primary stream. Use when the surrounding pipeline already owns a
   *        stream (e.g. an @ref OverlappedStencil 's compute stream) and the
   *        skeleton should chain onto it instead of starting a fresh one.
   *        Additional streams are still allocated from @p sys when @ref compile
   *        finds parallelism.
   */
  Skeleton(System sys,
           Stream stream,
           Options options = {})
      : sys_(std::move(sys)),
        options_(options) {
    streams_.push_back(std::move(stream));
  }

  /**
   * @brief Append @p k to the pipeline. Empty handles are rejected at add time
   *        rather than at @ref run time so the diagnostic points at the offending
   *        builder line rather than a launch deep in the iteration loop.
   *
   * Invalidates any previously-computed schedule — the next @ref run reanalyses
   * the DAG.
   */
  auto
  add(Kernel k) -> Skeleton& {
    if (k.empty()) {
      log::throw_with_log<std::invalid_argument>(
          "Skeleton::add: kernel is empty (default-constructed). Build it via Dense::make_kernel "
          "/ Field::make_halo_kernel / Kernel::reduce / Kernel::anchor first.");
    }
    kernels_.push_back(std::move(k));
    compiled_ = false;
    return *this;
  }

  /**
   * @brief Analyse the kernel sequence and freeze the schedule.
   *
   * Runs in six phases:
   *
   *   0. **Auto-halo insertion.** Walk the user kernel list maintaining a
   *      per-field "needs halo" flag (set on every write to @c F , cleared by
   *      every halo on @c F or by an auto-inserted halo). When a kernel
   *      reports an @c AccessPattern::Stencil read of @c F and the flag is
   *      set, the snapshotted @c HaloFactoryBase from the @c AccessToken is
   *      invoked to materialise a halo @c Kernel and insert it at this
   *      position. Under @ref Options::strict_halo the same condition throws
   *      @c std::logic_error instead, with a diagnostic naming the offending
   *      kernel and field. The result lives in @ref effective_kernels which
   *      every subsequent phase indexes over.
   *   0b. **PatternScalar readback (deferred reduce).** After phase 0, inserts
   *      @c KernelKind::ScalarReadback immediately after any reduction whose
   *      @c ReduceOptions::host_copy is @c Deferred (via the
   *      @c scalar_readback_factory on the device-scalar token). Readback runs
   *      on a dedicated @b detached copy stream; anchor fork/join events cover
   *      only non-copy streams so the compute anchor can overlap D→H. CUDA
   *      graph capture is disabled when a detached copy stream is present.
   *   1. **Conflict scan.** O(N²) pair-conflict scan over @ref effective_kernels
   *      (typical N is 5–20 so cheap); emits directed edge @c i→j whenever
   *      the kernel pair @c (i, j) shares a @c field_uid with conflicting
   *      modes. Read-Read is the only no-conflict pair.
   *   2. **Stream assignment.** Greedy walk in topological order: kernel
   *      @c k goes on a stream whose tail is a direct predecessor of @c k
   *      (FIFO ordering covers the dep, no event needed); else on the
   *      lowest-indexed empty stream; else allocates a fresh stream (no
   *      cap — carbon trusts the graph shape to bound the count).
   *   3. **Event allocation.** For every cross-stream DAG edge, allocate (or
   *      reuse) one @c Event recorded by the predecessor's stream after its
   *      launch and waited on by the successor's stream before its launch.
   *      Events are pooled per-predecessor (one event per kernel that has at
   *      least one cross-stream successor).
   *   4. **OCC rewrite.** When @c options_.overlap_mode != None , the
   *      schedule is post-processed: for each stencil node @c S with a
   *      preceding halo @c H , the rewrite splits selected kernels into
   *      Internal + Boundary halves so the Internal halves overlap with
   *      @c H 's exchange. The split set varies by mode (see @ref
   *      OverlapMode ); split kernels emit two @c Step entries with
   *      different @c view fields and the surrounding event wiring is
   *      adjusted. Reduce / Universal / Host kernels short-circuit the
   *      chain walk and stay unsplit.
   *   5. **Anchor fork-join events.** When @c streams_.size() > 1 , one
   *      "start" @c Event (recorded on the anchor stream, waited for by
   *      every secondary stream) plus one "join" @c Event per secondary
   *      stream (recorded there, waited for by the anchor) are allocated.
   *      Runs last so it sees the final stream count — the OCC rewrite can
   *      add streams.
   *
   * @c compile is idempotent — calling it on an already-compiled skeleton is a
   * no-op until @ref add (or @ref reset ) invalidates the schedule.
   *
   * Definition lives in @c src/skeleton.cpp .
   *
   * @throws std::logic_error from Phase 0 under @c Options::strict_halo when a
   *         stencil read finds no preceding halo since the last write to the
   *         field; from Phase 0 unconditionally when a stencil-read token
   *         carries a null halo factory (loader-bug guard).
   */
  auto
  compile() -> void;

  /**
   * @brief Replay the compiled schedule once on the bound stream(s).
   *
   * Calls @ref compile lazily on first invocation. For each kernel in user-add
   * order, the assigned stream first waits on every event recorded by a
   * cross-stream predecessor, then launches the kernel; if the kernel itself
   * has cross-stream successors, the post-launch event is recorded on the
   * same stream. After all per-step launches, every active stream records its
   * iteration-tail event so the next @c run can sync across streams.
   *
   * @param view  @ref DataView passed to every kernel's @c launch . Halo /
   *              Anchor kernels ignore @c view ; compute kernels honour it
   *              (Standard / Internal / Boundary). Default Standard matches
   *              the loose pipeline most callers want.
   */
  auto
  run(DataView view = DataView::Standard) -> void;

  /**
   * @brief Whether the most recent @ref run replayed via a captured CUDA/HIP
   *        graph (true) rather than the eager launch path (false).
   *
   * Reflects live state: @c false until the first graph-eligible @ref run
   * captures, and @c false whenever @ref Options::use_cuda_graph is off, the
   * backend cannot capture, the schedule is multi-*Skeleton*-stream (OCC) or
   * distributed, or a capture attempt fell back to eager. Primarily for tests /
   * introspection.
   */
  [[nodiscard]] auto
  uses_graph() const noexcept -> bool {
    return graph_.ready();
  }

  /**
   * @brief Block until every active stream drains. Equivalent to a fan-in
   *        @c wait() on every stream the schedule touches.
   */
  auto
  wait() const -> void {
    for (const Stream& s : streams_) {
      s.wait();
    }
  }

  /**
   * @brief Drop both the kernel list and the cached schedule. The bound
   *        @ref system , the primary @ref stream , and any extra streams already
   *        allocated stay attached so the same skeleton can be re-populated
   *        with a new pipeline without re-binding (the extra streams are
   *        simply not used until the next @ref compile decides whether to
   *        populate them again).
   */
  auto
  reset() -> void {
    kernels_.clear();
    effective_kernels_.clear();
    effective_to_user_.clear();
    edges_.clear();
    min_edges_.clear();
    schedule_.clear();
    record_event_per_kernel_.clear();
    kernel_stream_.clear();
    anchor_start_event_ = kNoEvent_;
    anchor_join_events_.clear();
    copy_stream_idx_ = kNoStream_;
    has_detached_stream_ = false;
    events_.clear();
    field_label_overrides_.clear();
    graph_.reset();
    graph_disabled_ = false;
    captured_view_ = DataView::Standard;
    compiled_ = false;
  }

  /** Number of kernels currently in the pipeline (regardless of compile state). */
  [[nodiscard]] auto
  size() const noexcept -> size_t {
    return kernels_.size();
  }

  /** True iff the pipeline carries no kernels. */
  [[nodiscard]] auto
  empty() const noexcept -> bool {
    return kernels_.empty();
  }

  /**
   * @brief True iff the schedule is in sync with the kernel list. Flips to
   *        @c false on every @ref add ; flips back to @c true after the next
   *        @ref compile (or implicit compile inside @ref run ).
   */
  [[nodiscard]] auto
  compiled() const noexcept -> bool {
    return compiled_;
  }

  /**
   * @brief Adjacency list of the inferred DAG. @c edges()[i] is the sorted set
   *        of indices @c j > i such that kernel @c i must complete before
   *        kernel @c j due to a token conflict. Empty (and @c size() == 0 )
   *        before @ref compile has been called.
   *
   * This is the @em full conflict graph — every detected token overlap shows
   * up as an explicit edge, including edges that are transitively implied by
   * a longer path. For the minimum-equivalent DAG (the right input for
   * scheduling), see @ref min_edges .
   */
  [[nodiscard]] auto
  edges() const noexcept -> const std::vector<std::vector<size_t>>& {
    return edges_;
  }

  /**
   * @brief Transitive reduction of @ref edges — the unique minimum DAG that
   *        induces the same partial order. An edge @c i → j is kept only when
   *        no longer path @c i → … → j exists in @ref edges .
   *
   * Examples:
   * - Linear chain ( @c edges == @c {{1, 2, 3}, {2, 3}, {3}, {}} ) reduces to
   *   @c {{1}, {2}, {3}, {}} — only the consecutive edges survive.
   * - Independent siblings feeding a join ( @c edges == @c {{2}, {2}, {}} )
   *   is already minimal and is returned unchanged.
   *
   * Same length and indexing convention as @ref edges (entry @c i carries the
   * surviving direct successors of kernel @c i , in ascending order). Empty
   * before @ref compile has been called.
   *
   * The full graph is preserved on @ref edges so diagnostics keep an
   * unambiguous view of every detected conflict; the reduction is what
   * downstream phases — stream assignment, cross-stream events, the OCC
   * rewrite — should consume to avoid issuing redundant synchronisation
   * work.
   */
  [[nodiscard]] auto
  min_edges() const noexcept -> const std::vector<std::vector<size_t>>& {
    return min_edges_;
  }

  /**
   * @brief Stream index assigned to each kernel by @ref compile . Same length
   *        and order as @ref kernels ; index @c 0 is the primary stream
   *        (returned by @ref stream ), higher indices are streams allocated
   *        lazily during @ref compile . Empty before @c compile .
   */
  [[nodiscard]] auto
  kernel_streams() const noexcept -> const std::vector<size_t>& {
    return kernel_stream_;
  }

  /**
   * @brief Number of streams the schedule actually populates. @c 1 for purely
   *        sequential pipelines (e.g. the canonical Jacobi ping-pong);
   *        @c >1 when @ref compile found parallelism. No compile-time cap —
   *        the count is bounded only by the parallelism the dependency graph
   *        admits (and by the OCC rewrite, which adds at most one extra
   *        halo stream per stencil node).
   */
  [[nodiscard]] auto
  num_streams() const noexcept -> size_t {
    return streams_.size();
  }

  /**
   * @brief Whether @ref compile allocated anchor fork/join events for this schedule.
   *
   * @c false for single-stream pipelines and for schedules whose only secondary
   * stream is the detached scalar readback copy stream (PatternScalar overlap path).
   */
  [[nodiscard]] auto
  uses_anchor_fork_join() const noexcept -> bool {
    return anchor_start_event_ != kNoEvent_;
  }

  /**
   * @brief Skeleton options bound at construction.
   */
  [[nodiscard]] auto
  options() const noexcept -> const Options& {
    return options_;
  }

  /**
   * @brief Effective kernel list — user kernels plus auto-inserted halos.
   *
   * Populated by @ref compile (Phase 0). Same length and indexing as
   * @ref kernel_streams , @ref edges , @ref min_edges , @ref schedule . An
   * entry whose @ref effective_to_user value is @c kNoKernel is an
   * auto-inserted halo (its @c Kernel was materialised from the
   * @c HaloFactoryBase snapshotted on a stencil-read token). Empty before
   * @ref compile has been called.
   */
  [[nodiscard]] auto
  effective_kernels() const noexcept -> const std::vector<Kernel>& {
    return effective_kernels_;
  }

  /**
   * @brief Maps each effective-kernel index back to the user's kernel index,
   *        or @c kNoKernel for auto-inserted halos.
   */
  [[nodiscard]] auto
  effective_to_user() const noexcept -> const std::vector<size_t>& {
    return effective_to_user_;
  }

  /**
   * @brief Sentinel value used by @ref effective_to_user (and elsewhere) to
   *        denote "no user-kernel index" (i.e. an auto-inserted halo).
   */
  static constexpr size_t kNoKernel = static_cast<size_t>(-1);

  /** The kernel list, in user-add order. Stable across @ref compile / @ref run . */
  [[nodiscard]] auto
  kernels() const noexcept -> const std::vector<Kernel>& {
    return kernels_;
  }

  /** Bound system. */
  [[nodiscard]] auto
  system() const noexcept -> const System& {
    return sys_;
  }

  /**
   * @brief Override the DOT-export label for the field identified by
   *        @p field_uid (typically @c field.gpu_buffers().get_uid() ).
   *
   * @ref AccessToken::field_name is captured by copy at @c KernelLoader
   * record time, so renaming a @ref Field via @c Field::set_name AFTER
   * adding the kernel that references it does not retroactively rewrite
   * the existing tokens. @ref export_dot already prefers the most-recent
   * non-empty token name across all kernels, so the typical "rename
   * then add more kernels" pattern works without this API. This setter
   * is the escape hatch for the rarer "rename AFTER all kernels are
   * added" case: the override wins over every token snapshot for the
   * matching uid in the next @ref export_dot call.
   *
   * Calling with an empty @p label removes any prior override for that
   * uid (so @ref export_dot falls back to the token-snapshot chain).
   * No-op when @p field_uid corresponds to a field no kernel in the
   * skeleton ever touched.
   */
  auto
  set_field_label(std::uint64_t field_uid,
                  std::string label) -> void {
    if (label.empty()) {
      field_label_overrides_.erase(field_uid);
      return;
    }
    field_label_overrides_[field_uid] = std::move(label);
  }

  /**
   * @brief Render the inferred DAG as a GraphViz DOT graph string.
   *
   * The graph is a digraph (one node per @c Kernel in user-add order; one edge
   * per inferred dependency). Each node is labelled with the kernel's
   * @c Kernel::name (or @c k<index> when unnamed), shows its @c kind() , the
   * stream index @ref compile picked, and a tooltip listing the per-token
   * @c (field_name, mode) pairs the kernel reports — exactly the records that
   * drive the dependency analysis. Each edge is labelled with the field
   * name(s) whose token conflict produced the edge ( @c "u_a (W→R)" reads as
   * "predecessor wrote @c u_a , successor reads @c u_a "). Edge tooltips list
   * all conflicting tokens, useful when several fields contribute to the
   * same edge.
   *
   * Streams are encoded as DOT subgraph clusters. Nodes are **schedule steps**
   * (post-OCC), labelled with kernel name, kind, and @c DataView . Solid
   * edges follow execution order; dashed red edges are cross-stream event
   * waits. Token-conflict edges between user kernels are omitted here in
   * favour of the lowered schedule (Neon @c ioToDot parity).
   *
   * @param graph_name Identifier placed after the @c digraph keyword. Free-
   *                   form; defaults to @c "carbon_skeleton" . Useful when
   *                   round-tripping the DOT through a tool that wants a
   *                   stable name.
   *
   * @return DOT graph as a string. Always non-empty (at minimum carries the
   *         @c digraph header and a blank body, even on an empty pipeline);
   *         lazy-compiles before rendering so the user does not have to call
   *         @ref compile first.
   */
  // Definition lives in src/skeleton.cpp (export_dot).
  [[nodiscard]] auto
  export_dot(std::string graph_name = "carbon_skeleton") -> std::string;

  /**
   * @brief Write @ref export_dot to @p path (GraphViz @c .dot file).
   * @return @c false if the file could not be opened.
   */
  [[nodiscard]] auto
  export_dot_file(std::string path,
                  std::string graph_name = "carbon_skeleton") -> bool;

  /**
   * @brief Write a @c .dot file beside @p png_path and render PNG via
   *        GraphViz (@c dot -Tpng ). Requires @c dot on @c PATH .
   * @return @c false if the dot file or render step failed.
   */
  [[nodiscard]] auto
  export_dot_png(std::string png_path,
                 std::string graph_name = "carbon_skeleton") -> bool;

  /**
   * @brief Text listing of every operation @ref run issues for one iteration.
   *
   * Mirrors the eager @ref run path (and the CUDA/HIP graph capture sequence when
   * @ref Options::use_cuda_graph is engaged): anchor fork record/waits, per-step
   * event waits, kernel launches (including auto-inserted halos), event records,
   * and anchor join record/waits. Lazy-compiles before rendering.
   */
  [[nodiscard]] auto
  export_launch_order() -> std::string;

  /**
   * @brief Write @ref export_launch_order to @p path.
   * @return @c false if the file could not be opened.
   */
  [[nodiscard]] auto
  export_launch_order_file(std::string path) -> bool;

  /**
   * @brief GraphViz "launch matrix" of one @ref run iteration: one column per
   *        stream, one row per operation from @ref export_launch_order (anchor
   *        syncs, event waits, launches, event records). Dashed arrows connect
   *        each event record to its waits (red for step events, blue for the
   *        anchor fork/join). Lazy-compiles before rendering.
   */
  [[nodiscard]] auto
  export_launch_matrix_dot(std::string graph_name = "carbon_launch_matrix") -> std::string;

  /**
   * @brief Write @ref export_launch_matrix_dot to @p path.
   * @return @c false if the file could not be opened.
   */
  [[nodiscard]] auto
  export_launch_matrix_dot_file(std::string path,
                                std::string graph_name = "carbon_launch_matrix") -> bool;

  /**
   * @brief GraphViz "kernel list" in the style of @ref export_launch_matrix_dot :
   *        one row per scheduled kernel in user-add order (auto-inserted halos
   *        included, dashed), one column per accessed resource showing the
   *        access mode (R / W / RW, stencil flagged), the reduced data
   *        dependencies, and the launches each kernel lowers to (run() sequence
   *        number, stream, view). Lazy-compiles before rendering.
   */
  [[nodiscard]] auto
  export_kernel_list_dot(std::string graph_name = "carbon_kernel_list") -> std::string;

  /**
   * @brief Write @ref export_kernel_list_dot to @p path.
   * @return @c false if the file could not be opened.
   */
  [[nodiscard]] auto
  export_kernel_list_dot_file(std::string path,
                              std::string graph_name = "carbon_kernel_list") -> bool;

  /**
   * @brief Primary execution stream (stream index 0). Matches the @ref Stream
   *        passed to the constructor (when given) or @c sys.stream() at
   *        construction time.
   */
  [[nodiscard]] auto
  stream() const noexcept -> const Stream& {
    return streams_[0];
  }

  /**
   * @brief Per-stream view. @c streams()[s] is the stream kernel index @c s
   *        was assigned to at compile time. Always at least one entry
   *        (the primary stream).
   */
  [[nodiscard]] auto
  streams() const noexcept -> const std::vector<Stream>& {
    return streams_;
  }

private:
  static constexpr size_t kNoStream_ = static_cast<size_t>(-1);
  static constexpr size_t kNoKernel_ = static_cast<size_t>(-1);
  static constexpr size_t kNoEvent_ = static_cast<size_t>(-1);

  /**
   * @brief One entry of the static schedule produced by @ref compile .
   *
   * @c wait_events lists indices into @c events_ that the assigned stream must
   * wait for before launching the kernel; @c record_event_idx is the event the
   * stream records after the launch (or @c kNoEvent_ when no downstream
   * cross-stream consumer exists). @c view selects which carbon @c DataView
   * the kernel is launched with — @c Standard for unsplit kernels (the
   * default), @c Internal / @c Boundary for the two halves of an OCC-split
   * kernel under a non-@c None @ref OverlapMode .
   *
   * Note: under non-trivial OCC modes a single kernel index may appear in
   * @b two @c Step entries — one with @c Internal , one with @c Boundary —
   * and the rest of the schedule rewires events around those halves.
   */
  struct Step {
    size_t kernel_idx{kNoKernel_};
    size_t stream_idx{0U};
    std::vector<size_t> wait_events{};
    size_t record_event_idx{kNoEvent_};
    DataView view{DataView::Standard};
    reduce::ReduceLaunchMode reduce_launch_mode{reduce::ReduceLaunchMode::Full};
  };

  /**
   * @brief OCC rewrite candidate: a (halo, stencil) pair where @c halo is a
   *        @c KernelKind::Halo direct predecessor of @c stencil (a
   *        @c KernelKind::Compute kernel with at least one
   *        @c AccessPattern::Stencil read of a field @c halo halos).
   *        @c stencil_fields lists the field UIDs of those Stencil reads —
   *        consumed by the Extended back-walk in @ref compile_step_graph_occ_ to
   *        discover Map-pattern predecessors that write any of them.
   */
  struct OccPair {
    size_t halo;
    size_t stencil;
    std::vector<std::uint64_t> stencil_fields;
  };

  enum class LaunchOrderOpKind : unsigned char {
    AnchorStartRecord,
    AnchorStartWait,
    EventWait,
    Launch,
    EventRecord,
    AnchorJoinRecord,
    AnchorJoinWait,
  };

  struct LaunchOrderOp {
    size_t seq{0U};
    LaunchOrderOpKind kind{};
    size_t stream_idx{0U};
    size_t step_idx{kNoKernel_};
    size_t event_idx{kNoEvent_};
  };

  [[nodiscard]] auto
  build_launch_order_ops_() const -> std::vector<LaunchOrderOp>;

  [[nodiscard]] auto
  format_launch_order_line_(const LaunchOrderOp& op) const -> std::string;

  [[nodiscard]] auto
  step_display_label_(size_t step_idx) const -> std::string;

  [[nodiscard]] auto
  event_annotation_(size_t event_idx) const -> std::string;

  /**
   * @brief Test whether any token in @p a conflicts with any token in @p b on
   *        the same @c field_uid . Linear scan over both vectors — fine for
   *        the small per-kernel token counts (1–4) typical of carbon kernels.
   */
  [[nodiscard]] static auto
  pair_conflicts_(const std::vector<AccessToken>& a,
                  const std::vector<AccessToken>& b) noexcept -> bool {
    for (const AccessToken& ta : a) {
      for (const AccessToken& tb : b) {
        if (ta.field_uid == tb.field_uid && conflicts(ta.mode, tb.mode)) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * @brief Allocate the anchor fork-join events (@c compile Phase 5).
   *
   * Appends one "start" @c Event (recorded on the anchor stream, waited for by every
   * secondary stream) plus one "end" @c Event per secondary stream (recorded there,
   * waited for by the anchor) to @c events_ , stashing their indices in
   * @c anchor_start_event_ / @c anchor_join_events_ . Runs after every other phase
   * that appends to @c events_ (and after the OCC rewrite, which can add a stream) so
   * the stashed indices and the stream count are both final. Single-stream schedules
   * allocate nothing — FIFO on the anchor stream already provides the ordering.
   */
  // Definition lives in src/skeleton.cpp (compile_allocate_anchor_events_).
  auto
  compile_allocate_anchor_events_() -> void;

  /**
   * @brief Drop redundant cross-stream waits, unused event records, and covered
   *        anchor joins after the schedule is fully lowered.
   */
  auto
  compile_prune_syncs_() -> void;

  /**
   * @brief Phase pre-0 of @ref compile : reject any Stencil-pattern WRITE
   *        with a single named diagnostic, before the auto-halo bootstrap
   *        builds @ref effective_kernels_ . carbon (mirroring Neon) requires
   *        writes to be local — Stencil reads are how neighbour pulls are
   *        spelled.
   *
   * Split out of @ref compile under P1.α of
   * @c docs/audit_2026_05_third_pass.md so the named contract is testable
   * in isolation and the @ref compile body stays focused on the schedule
   * lowering.
   */
  auto
  compile_validate_no_stencil_writes_() -> void;

  /**
   * @brief Phase 0 of @ref compile : build @ref effective_kernels_ +
   *        @ref effective_to_user_ from the user-added @ref kernels_ ,
   *        inserting an auto-halo ahead of any Stencil-read of a field
   *        whose @b dirty bit is set. Every field starts dirty (a freshly
   *        uploaded multi-GPU field's halo cells are uninitialised on
   *        device); manual halos clean their fields, every other kernel's
   *        Write / ReadWrite token marks the field dirty. @c strict_halo
   *        flips the auto-insert path into a hard error so user code that
   *        forgot a manual halo lights up at compile time instead of
   *        running with stale halo cells.
   *
   * Split out of @ref compile under P1.α of
   * @c docs/audit_2026_05_third_pass.md ; see the original comment block
   * inside the function for the full rationale (esp. why every field
   * starts dirty).
   */
  auto
  compile_auto_insert_halos_() -> void;

  /**
   * @brief Phase 0b: insert scalar readback kernels after deferred reductions.
   */
  auto
  compile_auto_insert_scalar_readbacks_() -> void;

  /**
   * @brief Phase 1 + 1b of @ref compile : token-driven conflict scan over
   *        @ref effective_kernels_ → @ref edges_ , followed by transitive
   *        reduction → @ref min_edges_ . The conflict scan is intentionally
   *        exhaustive (every (i, j) pair with overlapping fields is
   *        reported, the right semantics for diagnostics); the transitive
   *        reduction collapses chains so downstream phases pay no
   *        redundant cross-stream events.
   *
   * Split out of @ref compile under P1.α of
   * @c docs/audit_2026_05_third_pass.md .
   */
  auto
  compile_build_conflict_dag_() -> void;

  /**
   * @brief Neon-style step graph build, OCC edge rewrite, and generic lowering
   *        to @ref schedule_ (every @ref OverlapMode ).
   */
  auto
  compile_step_graph_and_lower_(const std::vector<std::vector<size_t>>& direct_preds_of) -> void;

  /**
   * @brief OCC rewrite of the step graph for every non-@c None @ref OverlapMode .
   *
   * For each (halo H, stencil S) candidate:
   *   - @b Standard : S becomes S-Internal + S-Boundary. S-Internal does not wait on H and is
   *     issued before it; S-Boundary waits on H. A reduction S launches both halves as
   *     @c ReduceLaunchMode::Partial and gains a @c Merge step that its consumers wait on.
   *   - @b Extended : every Compute predecessor P reachable back from S through Compute kernels
   *     that writes a field S stencil-reads becomes P-Boundary then P-Internal; H waits on
   *     P-Boundary only, so the halo overlaps P-Internal.
   *   - @b TwoWayExtended : every Map Compute successor C reachable forward from S that reads a
   *     field S (or an earlier such C) wrote becomes C-Internal + C-Boundary; Map-pattern
   *     Reduction successors become Partial Internal + Partial Boundary + Merge. When C only
   *     reads what a split producer wrote, each half waits on the matching half of the producer
   *     only.
   *
   * A kernel is split at most once; a later pair that reaches an already split kernel only
   * rewires its own halo edge.
   */
  auto
  compile_step_graph_occ_(const std::vector<std::vector<size_t>>& direct_preds_of,
                          std::vector<detail::StepGraphNode>& nodes) -> void;

  /**
   * @brief OCC rewrite phase 1: collect every (halo, stencil) candidate
   *        pair where @p halo is a direct predecessor of @p stencil in
   *        @ref min_edges_ , @p halo is @c KernelKind::Halo , @p stencil
   *        is @c KernelKind::Compute , and @p stencil has a Stencil-read
   *        of a field @p halo halos.
   *
   * Returns by value because the candidate set is typically tiny (a
   * handful of pairs in a real pipeline) and the simple shape outweighs
   * an out-param.
   */
  [[nodiscard]] auto
  occ_collect_candidates_(const std::vector<std::vector<size_t>>& direct_preds_of) const
      -> std::vector<OccPair>;

  /**
   * @brief Compute the transitive reduction of a forward-only adjacency list.
   *
   * Pre-condition: every entry @c full[i] holds indices @c j > i (the user-
   * add ordering @ref compile produces — kernels are appended in topological
   * order by construction, so every dependency edge points forward in the
   * vector). Under that invariant the input is acyclic and a unique
   * transitive reduction exists.
   *
   * Algorithm: walk @c i in reverse order ( @c n-1 down to @c 0 ); for each
   * @c i compute @c reachable_via[i] = union over @c j in @c full[i] of
   * @c reachable[j] , i.e. the set of nodes reachable from @c i through some
   * intermediate (path length @c ≥ 2 ). An edge @c i → j survives the
   * reduction iff @c j ∉ reachable_via[i] . Reachability is stored as
   * @c std::vector<bool> rows (one row per node) so the per-step union and
   * lookup are linear in @c n . Total cost @c O(n²) memory and at worst
   * @c O(n³) time, which is irrelevant for the @c n ≤ 20 typical of carbon
   * pipelines (and still cheap up into the low hundreds).
   */
  [[nodiscard]] static auto
  compute_min_edges_(const std::vector<std::vector<size_t>>& full)
      -> std::vector<std::vector<size_t>> {
    const size_t n = full.size();
    if (n == 0U) {
      return {};
    }

    // reachable[i][k] == true iff there is a directed path from i to k of
    // length >= 1. Built bottom-up in reverse user-add order so reachable[j]
    // is already populated when we visit i and inspect each j in full[i].
    std::vector<std::vector<bool>> reachable(n, std::vector<bool>(n, false));
    for (size_t step = 0; step < n; ++step) {
      const size_t i = n - 1U - step;
      auto& row_i = reachable[i];
      for (size_t j : full[i]) {
        row_i[j] = true;
        const auto& row_j = reachable[j];
        for (size_t k = 0; k < n; ++k) {
          if (row_j[k]) {
            row_i[k] = true;
          }
        }
      }
    }

    std::vector<std::vector<size_t>> reduced(n);
    std::vector<bool> indirect(n);
    for (size_t i = 0; i < n; ++i) {
      // indirect[k] = is k reachable from i via a path of length >= 2 ?
      // i.e. is there any j in full[i] (j != k) such that reachable[j][k] ?
      std::fill(indirect.begin(), indirect.end(), false);
      for (size_t j : full[i]) {
        const auto& row_j = reachable[j];
        for (size_t k = 0; k < n; ++k) {
          if (row_j[k]) {
            indirect[k] = true;
          }
        }
      }
      for (size_t j : full[i]) {
        if (!indirect[j]) {
          reduced[i].push_back(j);
        }
      }
    }
    return reduced;
  }

  System sys_{};
  Options options_{};
  std::vector<Stream> streams_{};
  std::vector<Event> events_{};
  std::vector<Kernel> kernels_{};
  // The kernels actually scheduled = kernels_ plus auto-inserted halos. Built
  // from kernels_ at the top of compile() and used by every subsequent phase.
  // edges_ / min_edges_ / kernel_stream_ / etc. are all indexed by
  // effective_kernels_'s order.
  std::vector<Kernel> effective_kernels_{};
  // Maps each effective_kernels_ index back to the user's kernels_ index, or
  // kNoKernel_ when the entry is an auto-inserted halo. Exposed via
  // @ref effective_kernels and @ref effective_to_user .
  std::vector<size_t> effective_to_user_{};
  std::vector<std::vector<size_t>> edges_{};
  std::vector<std::vector<size_t>> min_edges_{};
  std::vector<Step> schedule_{};
  std::vector<size_t> record_event_per_kernel_{};
  std::vector<size_t> kernel_stream_{};
  // Anchor fork-join (see @ref run_eager_ ). anchor_start_event_ is recorded on the
  // anchor stream (stream 0) at the top of every run and waited for by every secondary
  // stream (the fork); anchor_join_events_[s] is recorded on secondary stream s at the
  // end of the run and waited for by the anchor stream (the join). Both are kNoEvent_
  // for single-stream schedules, where FIFO on stream 0 already orders everything.
  size_t anchor_start_event_{kNoEvent_};
  std::vector<size_t> anchor_join_events_{};
  /** Index in @ref streams_ of the detached copy stream (scalar readback), or @c kNoStream_. */
  size_t copy_stream_idx_{kNoStream_};
  bool has_detached_stream_{false};
  // Optional per-uid label overrides for @ref export_dot . Populated by
  // @ref set_field_label ; consulted by export_dot's uid_to_label map
  // BEFORE the token-snapshot fallback so users can rename fields after
  // adding kernels and still get readable graphs without re-building
  // the pipeline.
  std::unordered_map<std::uint64_t, std::string> field_label_overrides_{};
  bool compiled_{false};

  // CUDA/HIP graph capture state (Options::use_cuda_graph). graph_ holds the
  // instantiated executable graph once captured; captured_view_ records the
  // DataView it was captured for so a run() with a different view re-captures;
  // graph_disabled_ latches after a capture attempt fails so we stop retrying
  // and stay on the eager path. Invalidated (reset) whenever compile() rebuilds
  // the schedule.
  detail::GraphReplay graph_{};
  DataView captured_view_{DataView::Standard};
  bool graph_disabled_{false};

  /**
   * @brief Issue the compiled schedule's per-step launches on their streams.
   *
   * The inner loop shared by the eager path and the graph-capture pass: for each
   * @ref Step it waits on the step's cross-stream events, launches the kernel
   * with the effective @ref DataView, and records the step's post-event. Does
   * @b not do the anchor fork-join (multi-stream-only, handled by
   * @ref run_eager_ ; the graph path is single-stream by construction so it is
   * a no-op there).
   */
  auto
  issue_schedule_(DataView view) -> void;

  /** Eager (non-graph) run: anchor fork + @ref issue_schedule_ + anchor join. */
  auto
  run_eager_(DataView view) -> void;

  /**
   * Graph run: capture-on-first-eligible-run, then replay. Single-*Skeleton*-
   * stream, intra-process (1..N GPUs via fork/join capture).
   */
  auto
  run_graph_(DataView view) -> void;

  /** Whether the compiled schedule is eligible for graph capture right now. */
  [[nodiscard]] auto
  graph_eligible_() const noexcept -> bool;
};

}  // namespace carbon

// Out-of-class member definitions for compile() / run() / compile_step_graph_occ_() /
// compile_allocate_anchor_events_() / export_dot() / export_launch_order() live in
// src/skeleton.cpp .
// User code only needs <carbon/skeleton.h>; the .cpp is statically linked
// into libcarbon. The Skeleton class above is non-template, so moving the
// large bodies out of the public header trims ~700 lines of parse work
// from every consumer translation unit at zero ABI cost.
