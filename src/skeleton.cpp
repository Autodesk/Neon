/**
 * @file src/skeleton.cpp
 *
 * Out-of-line definitions for @c carbon::Skeleton 's heavy member functions.
 * @c skeleton.h carries the public class declaration plus the small inline
 * accessors / constructors / @c add / @c reset / @c wait ; the bodies here
 * cover the schedule lowering pipeline:
 *
 *   - @c compile     — token-driven DAG inference, transitive reduction,
 *                      stream assignment, cross-stream event allocation,
 *                      OCC dispatch, anchor fork-join event allocation.
 *   - @c run         — replay the static schedule for one iteration.
 *   - @c apply_occ_  — Standard / Extended / TwoWayExtended graph rewrite
 *                      around (halo, stencil) pairs.
 *   - @c compile_allocate_anchor_events_ — start / join events for the
 *                                          per-run bracket on the anchor.
 *   - @c export_dot  — GraphViz DOT renderer for the inferred DAG.
 *
 * Splitting these out trims @c <carbon/skeleton.h> by ~700 lines of body
 * and removes ~thousand lines of parse work from every consumer TU.
 */

#include <carbon/access_token.h>
#include <carbon/detail/scalar_readback_factory.h>
#include <carbon/data_view.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/skeleton.h>
#include <carbon/system.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace carbon {

auto
Skeleton::compile() -> void {
  if (compiled_) {
    return;
  }

  // Phase pre-0: contract validation (Stencil-pattern WRITEs are illegal).
  compile_validate_no_stencil_writes_();

  // Phase 0: auto-halo insertion. Builds effective_kernels_ +
  // effective_to_user_ from kernels_ , inserting an auto-halo ahead of
  // every Stencil-read of a dirty field.
  compile_auto_insert_halos_();
  compile_auto_insert_scalar_readbacks_();

  copy_stream_idx_ = kNoStream_;
  has_detached_stream_ = false;

  const size_t n = effective_kernels_.size();
  edges_.assign(n, {});
  min_edges_.assign(n, {});
  schedule_.assign(n, Step{});
  record_event_per_kernel_.assign(n, kNoEvent_);
  kernel_stream_.assign(n, 0U);

  // Recompile path (add() after a prior compile()) rebuilds the event pool from
  // scratch: every phase below only ever push_back()es into events_, so without
  // this clear a second compile would strand the previous compile's CUDA events
  // (a leak) while the freshly-assigned indices point only at the new tail.
  // streams_ is intentionally *not* cleared — extra streams are reused in place.
  events_.clear();

  // Phase 1 + 1b: token-driven conflict scan -> edges_ , followed by the
  // transitive reduction -> min_edges_ .
  compile_build_conflict_dag_();

  // direct_preds_of[j] is the inverse adjacency of min_edges_ — i.e. the
  // *irreducible* predecessors of j. Computed once here and reused by the
  // stream-assignment pass and (when non-None) the OCC rewrite, so a
  // transitive edge i -> j whose ordering is already covered by some
  // intermediate chain does not contribute to stream-assignment
  // preferences and does not allocate a redundant cross-stream event.
  std::vector<std::vector<size_t>> direct_preds_of(n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j : min_edges_[i]) {
      direct_preds_of[j].push_back(i);
    }
  }

  // Phase 2 + 3: greedy stream assignment + cross-stream event allocation.
  compile_assign_streams_and_events_(direct_preds_of);

  // Phase 4: OCC graph rewrite (only for non-None modes). The rewrite
  // mutates schedule_ + kernel_stream_ + streams_ + events_ in place;
  // None is the identity (the schedule above already runs every kernel
  // with view=Standard). All four modes route through apply_occ_
  // (Standard / Extended / TwoWayExtended differ only in how far they
  // propagate the Internal/Boundary split through the DAG).
  if (options_.overlap_mode != OverlapMode::None) {
    apply_occ_(direct_preds_of);
  }

  // Phase 5: anchor fork-join events. Allocated AFTER the OCC rewrite so the
  // stream count is final (the rewrite may have moved halos to fresh halo
  // streams), and after every other phase that appends to events_ so the
  // indices stashed here stay valid.
  compile_allocate_anchor_events_();

  compiled_ = true;

  // A recompile rebuilds the schedule (and its streams / events / captured
  // kernel arguments), so any previously captured graph is stale — drop it and
  // re-arm the capture attempt for the new schedule.
  graph_.reset();
  graph_disabled_ = false;
  captured_view_ = DataView::Standard;

  if (options_.log_dot_on_compile) {
    // export_dot() lazy-compiles, but compiled_ is already true above
    // so it falls through to the rendering path.
    CARBON_LOG_INFO("Skeleton::compile: DOT export\n{}", export_dot());
  }
}

auto
Skeleton::compile_validate_no_stencil_writes_() -> void {
  // carbon, mirroring Neon's stencil semantics, disallows Stencil-pattern
  // WRITES (a kernel's writes must be local — Stencil reads are how
  // neighbour pulls are spelled). Catching this here gives a single,
  // explicit diagnostic instead of a confusing downstream auto-halo or
  // scheduling failure if a future KernelLoader API ever stamps a
  // Stencil pattern on a write token.
  for (size_t k_idx = 0; k_idx < kernels_.size(); ++k_idx) {
    const Kernel& k = kernels_[k_idx];
    for (const AccessToken& t : k.impl()->tokens()) {
      const bool is_write = (t.mode == AccessMode::Write) || (t.mode == AccessMode::ReadWrite);
      if (is_write && t.pattern == AccessPattern::Stencil) {
        log::throw_with_log<std::logic_error>(
            "Skeleton::compile: kernel '{}' (index={}) declares a Stencil-pattern WRITE on "
            "field '{}' (uid={}). carbon (matching Neon) requires writes to be local "
            "(AccessPattern::Map); use loader.write(field) for the write side and "
            "loader.read_stencil(field) only for neighbour reads.",
            std::string{k.name()}, k_idx,
            t.field_name.empty() ? std::string{"<anonymous>"} : t.field_name, t.field_uid);
      }
    }
  }
}

auto
Skeleton::compile_auto_insert_halos_() -> void {
  effective_kernels_.clear();
  effective_to_user_.clear();

  // field_uid -> "Dirty" if a write happened since the last halo
  // (manual or auto). Stencil-read on a dirty field triggers an insert.
  //
  // Every field appearing in any token starts dirty: a freshly uploaded
  // multi-GPU field has UNINITIALIZED halo regions on device (the host
  // mirror's halo cells are never written by Field::fill_owned_host , and
  // a single Transfer<T>(cpu, gpu) blasts the whole buffer including
  // those uninitialized halo bytes onto the device — the GPU halo is
  // garbage until the first halo exchange). Without this, the first
  // stencil read inside the skeleton finds the field "clean" (no
  // in-skeleton write yet) and skips the auto-halo, so the kernel reads
  // garbage across slab boundaries — see the LBM cavity benchmark for a
  // load-bearing example. Pre-existing manual halos in the pipeline still
  // dedup the redundant initial halo: when compile() walks the manual
  // halo first, it cleans the field before the subsequent stencil read
  // sees it (see test_skeleton subtest B' for the dedup contract). For
  // single-GPU layouts the inserted Halo is a no-op (pair_count==0), so
  // the conservative pre-dirty has zero runtime cost there.
  std::vector<std::uint64_t> dirty_fields;
  auto is_dirty = [&dirty_fields](std::uint64_t uid) {
    return std::find(dirty_fields.begin(), dirty_fields.end(), uid) != dirty_fields.end();
  };
  auto mark_clean = [&dirty_fields](std::uint64_t uid) {
    auto it = std::find(dirty_fields.begin(), dirty_fields.end(), uid);
    if (it != dirty_fields.end()) {
      dirty_fields.erase(it);
    }
  };
  auto mark_dirty = [&dirty_fields](std::uint64_t uid) {
    if (std::find(dirty_fields.begin(), dirty_fields.end(), uid) == dirty_fields.end()) {
      dirty_fields.push_back(uid);
    }
  };
  for (const Kernel& k : kernels_) {
    for (const AccessToken& t : k.impl()->tokens()) {
      mark_dirty(t.field_uid);
    }
  }

  for (size_t k_idx = 0; k_idx < kernels_.size(); ++k_idx) {
    const Kernel& k = kernels_[k_idx];
    const auto& toks = k.impl()->tokens();

    // First insert any required halos for this kernel. Track which
    // field_uids we've already inserted a halo for in this kernel so a
    // single kernel reading the same field with multiple Stencil tokens
    // (rare but possible) doesn't generate duplicate halos.
    std::vector<std::uint64_t> halo_inserted_for_this_kernel;
    for (const AccessToken& t : toks) {
      if (t.pattern != AccessPattern::Stencil) {
        continue;
      }
      if (!is_dirty(t.field_uid)) {
        continue;
      }
      if (std::find(halo_inserted_for_this_kernel.begin(), halo_inserted_for_this_kernel.end(),
                    t.field_uid) != halo_inserted_for_this_kernel.end()) {
        continue;
      }
      if (options_.strict_halo) {
        log::throw_with_log<std::logic_error>(
            "Skeleton::compile: strict_halo=true and stencil-read of field '{}' (uid={}) by "
            "kernel '{}' (index={}) finds no preceding halo since the last write. Insert "
            "Field::make_halo_kernel(...) before this kernel or set strict_halo=false to "
            "auto-insert.",
            t.field_name.empty() ? std::string{"<anonymous>"} : t.field_name, t.field_uid,
            std::string{k.name()}, k_idx);
      }
      if (!t.halo_factory) {
        // KernelLoader::read_stencil should always populate the factory.
        // A null factory here means the field's Field type doesn't satisfy
        // the trait — caught at a much better location than later.
        log::throw_with_log<std::logic_error>(
            "Skeleton::compile: stencil-read token for field '{}' (uid={}) carries no halo "
            "factory. The field type must implement make_halo_kernel() returning a "
            "carbon::Kernel — Dense::Field<T, Components> does; user grid families need to "
            "as well.",
            t.field_name.empty() ? std::string{"<anonymous>"} : t.field_name, t.field_uid);
      }
      Kernel auto_halo = t.halo_factory->make();
      effective_kernels_.push_back(std::move(auto_halo));
      effective_to_user_.push_back(kNoKernel_);  // marker: synthetic
      halo_inserted_for_this_kernel.push_back(t.field_uid);
      mark_clean(t.field_uid);
    }

    // Append the user kernel.
    effective_kernels_.push_back(k);
    effective_to_user_.push_back(k_idx);

    // Update the dirty-set based on this kernel's tokens. A Halo kernel
    // (manual or auto) cleans every field it touches; every other
    // kernel's Write / ReadWrite token marks the field dirty. Pure Read
    // tokens leave the state alone.
    const bool is_halo = (k.kind() == KernelKind::Halo);
    for (const AccessToken& t : toks) {
      if (is_halo) {
        mark_clean(t.field_uid);
      } else if (t.mode == AccessMode::Write || t.mode == AccessMode::ReadWrite) {
        mark_dirty(t.field_uid);
      }
    }
  }
}

auto
Skeleton::compile_auto_insert_scalar_readbacks_() -> void {
  std::vector<Kernel> next_kernels;
  std::vector<size_t> next_map;
  next_kernels.reserve(effective_kernels_.size() + 4U);
  next_map.reserve(effective_kernels_.size() + 4U);

  for (size_t k_idx = 0; k_idx < effective_kernels_.size(); ++k_idx) {
    next_kernels.push_back(effective_kernels_[k_idx]);
    next_map.push_back(effective_to_user_[k_idx]);

    std::shared_ptr<detail::ScalarReadbackFactoryBase> factory;
    for (const AccessToken& t : effective_kernels_[k_idx].impl()->tokens()) {
      if (t.scalar_readback_factory) {
        factory = t.scalar_readback_factory;
        break;
      }
    }
    if (!factory) {
      continue;
    }
    if (k_idx + 1U < effective_kernels_.size()) {
      if (effective_kernels_[k_idx + 1U].kind() == KernelKind::ScalarReadback) {
        continue;
      }
    }
    Kernel readback = factory->make();
    next_kernels.push_back(std::move(readback));
    next_map.push_back(kNoKernel_);
  }

  effective_kernels_ = std::move(next_kernels);
  effective_to_user_ = std::move(next_map);
}

auto
Skeleton::compile_build_conflict_dag_() -> void {
  const size_t n = effective_kernels_.size();
  // Phase 1: token-driven conflict scan -> edges_ (directed, i -> j with i < j).
  for (size_t i = 0; i < n; ++i) {
    const auto& tok_i = effective_kernels_[i].impl()->tokens();
    for (size_t j = i + 1; j < n; ++j) {
      const auto& tok_j = effective_kernels_[j].impl()->tokens();
      if (pair_conflicts_(tok_i, tok_j)) {
        edges_[i].push_back(j);
      }
    }
  }

  // Phase 1b: transitive reduction. The conflict scan is intentionally
  // exhaustive (every (i, j) pair with overlapping fields is reported),
  // which is the right semantics for diagnostics but over-eager for
  // scheduling — a path i -> ... -> j already enforces the i -> j
  // ordering at run time. min_edges_ is the unique minimum DAG covering
  // the same partial order as edges_; downstream phases (and the
  // deferred tighter inter-iter barrier) consume it instead of the full
  // graph.
  min_edges_ = compute_min_edges_(edges_);
}

auto
Skeleton::compile_assign_streams_and_events_(
    const std::vector<std::vector<size_t>>& direct_preds_of) -> void {
  const size_t n = effective_kernels_.size();

  // stream_tail_[s] = index of the most-recently assigned kernel on stream s,
  // or kNoKernel_ when the stream is empty. Stream 0 always exists (allocated
  // at construction); higher streams are allocated lazily below.
  std::vector<size_t> stream_tail(streams_.size(), kNoKernel_);

  for (size_t k_idx = 0; k_idx < n; ++k_idx) {
    const std::vector<size_t>& preds = direct_preds_of[k_idx];
    const KernelKind k_kind = effective_kernels_[k_idx].kind();

    // Pick a stream:
    //   - first preference: extend a stream whose tail is a direct predecessor
    //     of k (FIFO covers the dep, no event needed for that endpoint);
    //   - second preference: claim the lowest-indexed empty stream;
    //   - else: allocate a fresh stream (no compile-time cap).
    size_t chosen = kNoStream_;
    for (size_t s = 0; s < stream_tail.size(); ++s) {
      const size_t tail = stream_tail[s];
      if (tail != kNoKernel_) {
        for (size_t p : preds) {
          if (p == tail) {
            chosen = s;
            break;
          }
        }
        if (chosen != kNoStream_) {
          break;
        }
      }
    }
    if (chosen == kNoStream_) {
      for (size_t s = 0; s < stream_tail.size(); ++s) {
        if (stream_tail[s] == kNoKernel_) {
          chosen = s;
          break;
        }
      }
    }
    if (k_kind == KernelKind::ScalarReadback) {
      if (copy_stream_idx_ == kNoStream_) {
        streams_.push_back(sys_.stream());
        copy_stream_idx_ = streams_.size() - 1U;
        has_detached_stream_ = true;
        stream_tail.push_back(kNoKernel_);
      }
      chosen = copy_stream_idx_;
    } else if (chosen == kNoStream_) {
      // Allocate a fresh stream unless the user asked us to cap.
      // When capped, fall back to stream 0 (and pay the cross-stream
      // event below for any non-FIFO predecessor) — same behaviour as
      // the pre-multi-stream skeleton, so a cap of 1 reproduces v1
      // single-stream serial scheduling exactly.
      if (options_.max_streams != 0U && streams_.size() >= options_.max_streams) {
        chosen = 0U;
      } else {
        streams_.push_back(sys_.stream());
        stream_tail.push_back(kNoKernel_);
        chosen = streams_.size() - 1U;
      }
    }

    kernel_stream_[k_idx] = chosen;
    stream_tail[chosen] = k_idx;

    // Cross-stream waits: for every direct predecessor on a different stream,
    // ensure the predecessor records an event after its launch and that this
    // step waits for that event before launching.
    Step& step = schedule_[k_idx];
    step.kernel_idx = k_idx;
    step.stream_idx = chosen;
    for (size_t p : preds) {
      if (kernel_stream_[p] != chosen) {
        if (record_event_per_kernel_[p] == kNoEvent_) {
          events_.push_back(sys_.event());
          record_event_per_kernel_[p] = events_.size() - 1U;
          schedule_[p].record_event_idx = record_event_per_kernel_[p];
        }
        step.wait_events.push_back(record_event_per_kernel_[p]);
      }
    }
  }
}

auto
Skeleton::run(DataView view) -> void {
  compile();

  // CUDA/HIP graph fast path: capture the schedule once, replay it thereafter.
  // Only engaged for single-stream / single-GPU schedules on a capture-capable
  // backend (see graph_eligible_); everything else drops through to the eager
  // path below with identical semantics.
  if (graph_eligible_()) {
    run_graph_(view);
    return;
  }

  run_eager_(view);
}

auto
Skeleton::issue_schedule_(DataView view) -> void {
  for (const Step& step : schedule_) {
    Stream& s = streams_[step.stream_idx];
    for (size_t evt_idx : step.wait_events) {
      s.wait_for(events_[evt_idx]);
    }
    // Step::view selects the per-launch DataView. For pipelines with
    // OverlapMode::None (the default) this is always DataView::Standard
    // and the per-step view collapses to the user-supplied @p view (so
    // run(DataView::Internal) still works for callers driving the whole
    // pipeline on one side). Under non-None OverlapMode, Step::view is
    // forced to Internal / Boundary by the OCC rewrite and overrides the
    // user's @p view.
    const DataView effective_view = (step.view == DataView::Standard) ? view : step.view;
    effective_kernels_[step.kernel_idx].launch(s, effective_view);
    if (step.record_event_idx != kNoEvent_) {
      events_[step.record_event_idx].record(s);
    }
  }
}

auto
Skeleton::run_eager_(DataView view) -> void {
  // Anchor fork ("start"): every secondary stream waits until the anchor stream
  // (stream 0) reaches this run's start point. Two consequences, both required for
  // correctness once the anchor is the System's shared default stream:
  //   * Cross-skeleton — the start event is recorded into the anchor's FIFO, which
  //     already holds the previous skeleton's join, so this skeleton's secondary
  //     streams cannot begin until the previous skeleton has fully finished. That is
  //     what lets an alternating ping-pong pair run without a host drain between
  //     iterations (the halo of iteration N+1 can no longer peer-copy a population
  //     that iteration N is still writing).
  //   * Cross-iteration — same argument applied to this skeleton's own previous run,
  //     which makes this the whole inter-iteration barrier: combined with the join
  //     below, no stream can start iteration N+1 before every stream finished N.
  if (streams_.size() > 1U && anchor_start_event_ != kNoEvent_) {
    events_[anchor_start_event_].record(streams_[0]);
    for (size_t s = 1U; s < streams_.size(); ++s) {
      if (s == copy_stream_idx_) {
        continue;
      }
      streams_[s].wait_for(events_[anchor_start_event_]);
    }
  }

  issue_schedule_(view);

  // Anchor join ("end"): the anchor stream waits for every secondary stream's tail,
  // so after run() the anchor alone represents whole-skeleton completion. That is what
  // makes the *next* start event (recorded into the same FIFO) a valid whole-pipeline
  // barrier, and it means wait()ing the anchor is sufficient to drain the skeleton.
  if (streams_.size() > 1U) {
    for (size_t s = 1U; s < streams_.size(); ++s) {
      if (s == copy_stream_idx_) {
        continue;
      }
      if (anchor_join_events_[s] != kNoEvent_) {
        events_[anchor_join_events_[s]].record(streams_[s]);
        streams_[0].wait_for(events_[anchor_join_events_[s]]);
      }
    }
  }
}

auto
Skeleton::graph_eligible_() const noexcept -> bool {
  if (!options_.use_cuda_graph || graph_disabled_ || !detail::GraphReplay::supported()) {
    return false;
  }
  if (has_detached_stream_) {
    return false;
  }
  // A single Skeleton stream (OverlapMode::None) means no cross-*Skeleton*-stream
  // event choreography to capture. The GPUs behind it may be one (ThreadLocal
  // capture) or many (global-mode fork/join capture over the per-GPU streams —
  // the halo's cross-device peer copies + event fences fold into the graph as
  // intra-graph dependencies). Multi-*Skeleton*-stream (OCC) capture is a tracked
  // follow-up.
  if (streams_.size() != 1U || streams_[0].size() == 0U || effective_kernels_.empty()) {
    return false;
  }
  // The distributed (cross-rank NCCL) halo path is not capture-safe here; only
  // the intra-process multi-GPU case is captured. Single-process runs have no
  // Distributed handle.
  if (sys_.distributed() != nullptr) {
    return false;
  }
  // Host (OpenMP) kernels run on the CPU and cannot live in a device graph.
  for (const Kernel& k : effective_kernels_) {
    const KernelKind kind = k.kind();
    if (kind == KernelKind::ComputeHost || kind == KernelKind::ReductionHost) {
      return false;
    }
  }
  return true;
}

auto
Skeleton::run_graph_(DataView view) -> void {
#if CARBON_BACKEND_IS_CUDALIKE
  // Origin stream = per-GPU stream 0 of the single Skeleton stream; the graph is
  // captured and replayed from the origin's device. For a multi-GPU layout the
  // remaining per-GPU streams are folded into the same graph via fork/join.
  const size_t n_gpus = streams_[0].size();
  const auto& devices = sys_.gpu_group().devices();
  std::vector<detail::gpu_stream_t> capture_streams(n_gpus);
  std::vector<int> device_ids(n_gpus);
  for (size_t i = 0; i < n_gpus; ++i) {
    capture_streams[i] = streams_[0].native_at(i);
    device_ids[i] = devices[i].device_id();
  }
  const detail::gpu_stream_t stream = capture_streams[0];

  if (graph_.ready() && captured_view_ == view) {
    if (graph_.launch(stream)) {
      return;
    }
    // A replay that fails at runtime (rare) drops us back to eager for good.
    graph_.reset();
    graph_disabled_ = true;
    run_eager_(view);
    return;
  }

  // (Re)capture for this view. The enqueue lambda records — does not execute —
  // the schedule, so on success we launch the graph to actually run this
  // iteration exactly once. A schedule that issues a capture-illegal op (e.g. a
  // non-P2P peer copy that must stage on the host) surfaces as a thrown
  // exception from the enqueue path; GraphReplay::capture drains the in-flight
  // capture and rethrows, and we treat that identically to a soft capture
  // failure — nothing executed yet, so we fall back to an eager run.
  bool captured = false;
  try {
    captured = graph_.capture(capture_streams, device_ids, [&] {
      issue_schedule_(view);
    });
  } catch (const std::exception& e) {
    graph_.reset();
    graph_disabled_ = true;
    captured_view_ = view;
    CARBON_LOG_WARN(
        "Skeleton: CUDA graph capture raised '{}' for this pipeline; falling back to eager launch "
        "(results unaffected).",
        e.what());
    run_eager_(view);
    return;
  }
  captured_view_ = view;
  if (captured && graph_.launch(stream)) {
    return;
  }
  // Capture or first launch failed: nothing executed, so run the iteration
  // eagerly and stop attempting capture for this compiled schedule.
  graph_.reset();
  graph_disabled_ = true;
  CARBON_LOG_WARN(
      "Skeleton: CUDA graph capture/replay unavailable for this pipeline; falling back to eager "
      "launch (results unaffected).");
  run_eager_(view);
#else
  run_eager_(view);
#endif
}

auto
Skeleton::compile_allocate_anchor_events_() -> void {
  // compile() clears events_ , so any indices from a previous compile are stale.
  anchor_start_event_ = kNoEvent_;
  anchor_join_events_.assign(streams_.size(), kNoEvent_);
  // A single-stream schedule needs no fork-join: every launch already lands on the
  // anchor stream in FIFO order, which orders both the kernels within an iteration
  // and successive iterations (and successive skeletons sharing the anchor).
  if (streams_.size() <= 1U) {
    return;
  }
  // Copy-only secondaries (deferred scalar readback) are excluded from the anchor
  // bracket so the next skeleton run can overlap D→H on the copy stream.
  bool secondary_needs_anchor = false;
  for (size_t s = 1U; s < streams_.size(); ++s) {
    if (s != copy_stream_idx_) {
      secondary_needs_anchor = true;
      break;
    }
  }
  if (!secondary_needs_anchor) {
    return;
  }
  events_.push_back(sys_.event());
  anchor_start_event_ = events_.size() - 1U;
  for (size_t s = 1U; s < streams_.size(); ++s) {
    if (s == copy_stream_idx_) {
      continue;
    }
    events_.push_back(sys_.event());
    anchor_join_events_[s] = events_.size() - 1U;
  }
}

auto
Skeleton::apply_occ_(const std::vector<std::vector<size_t>>& direct_preds_of) -> void {
  if (effective_kernels_.empty()) {
    return;
  }
  const size_t n = effective_kernels_.size();

  // Phase 1: candidate (halo, stencil) pair collection.
  const std::vector<OccPair> candidates = occ_collect_candidates_(direct_preds_of);
  if (candidates.empty()) {
    return;
  }

  // Initial step_for_kernel index — schedule_'s entries are 1-to-1 with
  // effective_kernels_ at this point (the per-pair Standard split below
  // is the first thing that grows the schedule, and it bumps the index
  // for every kernel after the pivot via occ_bump_step_for_kernel_after_).
  std::vector<size_t> step_for_kernel(n, kNoKernel_);
  for (size_t i = 0; i < schedule_.size(); ++i) {
    const size_t k = schedule_[i].kernel_idx;
    if (step_for_kernel[k] == kNoKernel_) {
      step_for_kernel[k] = i;
    }
  }
  // Which step *finishes* each kernel. Identical to step_for_kernel until a split moves one
  // half of a kernel into a new slot; from then on the two disagree and cross-stream fences
  // must use this one.
  std::vector<size_t> completion_step_for_kernel = step_for_kernel;

  // Map-back state shared across pairs (a single Compute predecessor that
  // writes to multiple stencil fields is split exactly once).
  std::vector<bool> already_split_map_back(n, false);
  std::vector<size_t> map_back_boundary_event(n, kNoEvent_);
  std::vector<bool> already_split_forward(n, false);

  // Build direct_succs_of once if any TwoWayExtended forward walk needs
  // it; otherwise it stays empty and the helper is never invoked.
  std::vector<std::vector<size_t>> direct_succs_of;
  if (options_.overlap_mode == OverlapMode::TwoWayExtended) {
    direct_succs_of.assign(n, {});
    for (size_t k = 0; k < n; ++k) {
      for (size_t pred : direct_preds_of[k]) {
        direct_succs_of[pred].push_back(k);
      }
    }
  }

  for (const OccPair& pair : candidates) {
    // Phase 2: per-pair Standard split (move halo to fresh stream if
    // needed; split stencil into Internal + Boundary halves). Always
    // runs for non-None modes — Extended / TwoWayExtended widen the
    // rewrite further but build on the Standard split.
    occ_apply_standard_split_(pair, direct_preds_of, step_for_kernel, completion_step_for_kernel);

    if (options_.overlap_mode == OverlapMode::Standard) {
      continue;
    }

    // Phase 3: Extended / TwoWayExtended Map-back walk.
    occ_apply_extended_back_walk_(pair, direct_preds_of, step_for_kernel,
                                  completion_step_for_kernel, already_split_map_back,
                                  map_back_boundary_event);

    if (options_.overlap_mode != OverlapMode::TwoWayExtended) {
      continue;
    }

    // Phase 4: TwoWayExtended forward walk.
    occ_apply_two_way_forward_walk_(pair, direct_succs_of, step_for_kernel,
                                    completion_step_for_kernel, already_split_forward);
  }
}

auto
Skeleton::occ_collect_candidates_(const std::vector<std::vector<size_t>>& direct_preds_of) const
    -> std::vector<OccPair> {
  const size_t n = effective_kernels_.size();
  std::vector<OccPair> candidates;
  for (size_t s = 0; s < n; ++s) {
    const Kernel& ks = effective_kernels_[s];
    if (ks.kind() != KernelKind::Compute) {
      continue;
    }
    std::vector<std::uint64_t> stencil_fields;
    for (const AccessToken& t : ks.impl()->tokens()) {
      if (t.pattern == AccessPattern::Stencil) {
        stencil_fields.push_back(t.field_uid);
      }
    }
    if (stencil_fields.empty()) {
      continue;
    }
    for (size_t h : direct_preds_of[s]) {
      const Kernel& kh = effective_kernels_[h];
      if (kh.kind() != KernelKind::Halo) {
        continue;
      }
      bool match = false;
      for (const AccessToken& th : kh.impl()->tokens()) {
        if (std::find(stencil_fields.begin(), stencil_fields.end(), th.field_uid) !=
            stencil_fields.end()) {
          match = true;
          break;
        }
      }
      if (!match) {
        continue;
      }
      candidates.push_back({h, s, std::move(stencil_fields)});
      break;
    }
  }
  return candidates;
}

auto
Skeleton::occ_allocate_record_event_for_(size_t k_idx,
                                         const std::vector<size_t>& completion_step_for_kernel)
    -> size_t {
  if (record_event_per_kernel_[k_idx] != kNoEvent_) {
    return record_event_per_kernel_[k_idx];
  }
  Step& completion = schedule_[completion_step_for_kernel[k_idx]];
  if (completion.record_event_idx != kNoEvent_) {
    record_event_per_kernel_[k_idx] = completion.record_event_idx;
    return completion.record_event_idx;
  }
  events_.push_back(sys_.event());
  const size_t evt = events_.size() - 1U;
  record_event_per_kernel_[k_idx] = evt;
  completion.record_event_idx = evt;
  return evt;
}

auto
Skeleton::occ_bump_step_for_kernel_after_(size_t pivot,
                                          std::vector<size_t>& step_for_kernel) -> void {
  for (size_t k = 0; k < step_for_kernel.size(); ++k) {
    if (step_for_kernel[k] != kNoKernel_ && step_for_kernel[k] > pivot) {
      step_for_kernel[k] += 1U;
    }
  }
}

auto
Skeleton::occ_apply_standard_split_(const OccPair& pair,
                                    const std::vector<std::vector<size_t>>& direct_preds_of,
                                    std::vector<size_t>& step_for_kernel,
                                    std::vector<size_t>& completion_step_for_kernel) -> void {
  const size_t h = pair.halo;
  const size_t s = pair.stencil;

  const size_t step_h_idx = step_for_kernel[h];
  const size_t step_s_idx = step_for_kernel[s];

  Step& step_h = schedule_[step_h_idx];
  Step& step_s = schedule_[step_s_idx];

  // If halo currently shares a stream with stencil, move it to a fresh
  // (unused) stream so the halo and the Internal-view stencil step can
  // overlap. Reuse an empty stream when one exists; otherwise allocate.
  if (step_h.stream_idx == step_s.stream_idx) {
    size_t halo_stream_idx;
    {
      std::vector<bool> used(streams_.size(), false);
      for (size_t s_used : kernel_stream_) {
        if (s_used < used.size()) {
          used[s_used] = true;
        }
      }
      size_t reuse = kNoStream_;
      for (size_t s_idx = 0; s_idx < streams_.size(); ++s_idx) {
        if (!used[s_idx]) {
          reuse = s_idx;
          break;
        }
      }
      if (reuse == kNoStream_) {
        streams_.push_back(sys_.stream());
        halo_stream_idx = streams_.size() - 1U;
      } else {
        halo_stream_idx = reuse;
      }
    }
    const size_t old_h_stream = step_h.stream_idx;
    step_h.stream_idx = halo_stream_idx;
    kernel_stream_[h] = halo_stream_idx;

    // Predecessors of halo that previously rode FIFO with it on the old
    // stream now need explicit cross-stream events to enforce the same
    // ordering on the new halo stream.
    for (size_t p : direct_preds_of[h]) {
      if (kernel_stream_[p] == old_h_stream) {
        const size_t evt = occ_allocate_record_event_for_(p, completion_step_for_kernel);
        if (std::find(step_h.wait_events.begin(), step_h.wait_events.end(), evt) ==
            step_h.wait_events.end()) {
          step_h.wait_events.push_back(evt);
        }
      }
    }
  }

  const size_t halo_event = occ_allocate_record_event_for_(h, completion_step_for_kernel);

  events_.push_back(sys_.event());
  const size_t s_internal_event = events_.size() - 1U;

  // Boundary half: waits on halo + on Internal half's record event,
  // inherits the original stencil step's downstream consumers via the
  // record_event slot below.
  Step boundary_step = step_s;
  boundary_step.view = DataView::Boundary;
  boundary_step.wait_events.clear();
  boundary_step.wait_events.push_back(halo_event);
  boundary_step.wait_events.push_back(s_internal_event);

  // Internal half: drops the halo wait (it doesn't need haloed cells),
  // records the s_internal_event so Boundary can wait on it.
  step_s.view = DataView::Internal;
  step_s.wait_events.erase(
      std::remove(step_s.wait_events.begin(), step_s.wait_events.end(), halo_event),
      step_s.wait_events.end());
  step_s.record_event_idx = s_internal_event;

  schedule_.insert(schedule_.begin() + static_cast<std::ptrdiff_t>(step_s_idx) + 1, boundary_step);
  occ_bump_step_for_kernel_after_(step_s_idx, step_for_kernel);
  occ_bump_step_for_kernel_after_(step_s_idx, completion_step_for_kernel);
  // The Boundary half runs last, so it is what "kernel s finished" means to anything fencing
  // against s from another stream.
  completion_step_for_kernel[s] = step_s_idx + 1U;
}

auto
Skeleton::occ_apply_extended_back_walk_(const OccPair& pair,
                                        const std::vector<std::vector<size_t>>& direct_preds_of,
                                        std::vector<size_t>& step_for_kernel,
                                        std::vector<size_t>& completion_step_for_kernel,
                                        std::vector<bool>& already_split_map_back,
                                        std::vector<size_t>& map_back_boundary_event) -> void {
  const size_t n = effective_kernels_.size();
  const size_t h = pair.halo;
  const size_t s = pair.stencil;

  // BFS-back from stencil through direct_preds_of collecting Compute
  // predecessors that write any of the stencil's Stencil-read fields.
  // Halo / non-Compute predecessors halt the recursion (the chain-stop
  // rule documented on @ref OverlapMode ).
  std::vector<bool> visited(n, false);
  visited[s] = true;
  visited[h] = true;
  std::vector<size_t> stack{s};
  std::vector<size_t> map_back_set;
  while (!stack.empty()) {
    const size_t k = stack.back();
    stack.pop_back();
    for (size_t p : direct_preds_of[k]) {
      if (visited[p]) {
        continue;
      }
      visited[p] = true;
      const Kernel& kp = effective_kernels_[p];
      if (kp.kind() != KernelKind::Compute) {
        continue;
      }
      bool writes_stencil_field = false;
      for (const AccessToken& tp : kp.impl()->tokens()) {
        if (tp.mode != AccessMode::Write && tp.mode != AccessMode::ReadWrite) {
          continue;
        }
        if (std::find(pair.stencil_fields.begin(), pair.stencil_fields.end(), tp.field_uid) !=
            pair.stencil_fields.end()) {
          writes_stencil_field = true;
          break;
        }
      }
      if (writes_stencil_field) {
        map_back_set.push_back(p);
      }
      stack.push_back(p);
    }
  }

  for (size_t p : map_back_set) {
    size_t p_boundary_event = map_back_boundary_event[p];
    if (!already_split_map_back[p]) {
      const size_t step_p_idx = step_for_kernel[p];
      Step& step_p = schedule_[step_p_idx];

      events_.push_back(sys_.event());
      p_boundary_event = events_.size() - 1U;
      map_back_boundary_event[p] = p_boundary_event;

      // Internal half: inherits the original record event so downstream
      // consumers transparently wait on the full completion.
      Step internal_step = step_p;
      internal_step.view = DataView::Internal;
      internal_step.wait_events.clear();
      internal_step.wait_events.push_back(p_boundary_event);
      internal_step.record_event_idx = step_p.record_event_idx;

      // Boundary half (existing slot): records the new event used by halo.
      step_p.view = DataView::Boundary;
      step_p.record_event_idx = p_boundary_event;

      schedule_.insert(schedule_.begin() + static_cast<std::ptrdiff_t>(step_p_idx) + 1,
                       internal_step);
      occ_bump_step_for_kernel_after_(step_p_idx, step_for_kernel);
      occ_bump_step_for_kernel_after_(step_p_idx, completion_step_for_kernel);
      // This split is the mirror of the Standard one: the existing slot becomes the Boundary
      // half and the Internal half is what runs last.
      completion_step_for_kernel[p] = step_p_idx + 1U;

      already_split_map_back[p] = true;
    }

    Step& step_h_post = schedule_[step_for_kernel[h]];

    const size_t old_p_event = record_event_per_kernel_[p];
    if (old_p_event != kNoEvent_) {
      step_h_post.wait_events.erase(
          std::remove(step_h_post.wait_events.begin(), step_h_post.wait_events.end(), old_p_event),
          step_h_post.wait_events.end());
    }
    if (std::find(step_h_post.wait_events.begin(), step_h_post.wait_events.end(),
                  p_boundary_event) == step_h_post.wait_events.end()) {
      step_h_post.wait_events.push_back(p_boundary_event);
    }
  }
}

auto
Skeleton::occ_apply_two_way_forward_walk_(const OccPair& pair,
                                          const std::vector<std::vector<size_t>>& direct_succs_of,
                                          std::vector<size_t>& step_for_kernel,
                                          std::vector<size_t>& completion_step_for_kernel,
                                          std::vector<bool>& already_split_forward) -> void {
  const size_t n = effective_kernels_.size();
  const size_t s = pair.stencil;

  std::vector<std::uint64_t> tainted_fields;
  for (const AccessToken& ts : effective_kernels_[s].impl()->tokens()) {
    if (ts.mode == AccessMode::Write || ts.mode == AccessMode::ReadWrite) {
      tainted_fields.push_back(ts.field_uid);
    }
  }

  // BFS-forward from stencil through direct_succs_of collecting Compute
  // successors that read tainted fields. Two chain-stop rules apply:
  //   * any successor with a Stencil-pattern read halts that branch (its
  //     own halo-bracketed pair will be handled separately);
  //   * non-Compute successors (Halo / Reduction / Anchor / Universal*)
  //     halt the recursion.
  std::vector<bool> fwd_visited(n, false);
  fwd_visited[s] = true;
  std::vector<size_t> fwd_queue{s};
  std::vector<size_t> map_forward_set;
  while (!fwd_queue.empty()) {
    const size_t k = fwd_queue.front();
    fwd_queue.erase(fwd_queue.begin());
    for (size_t c : direct_succs_of[k]) {
      if (fwd_visited[c]) {
        continue;
      }
      fwd_visited[c] = true;
      const Kernel& kc = effective_kernels_[c];
      if (kc.kind() != KernelKind::Compute) {
        continue;
      }
      bool has_stencil_read = false;
      bool reads_tainted = false;
      for (const AccessToken& tc : kc.impl()->tokens()) {
        if (tc.pattern == AccessPattern::Stencil) {
          has_stencil_read = true;
          break;
        }
      }
      if (has_stencil_read) {
        continue;
      }
      for (const AccessToken& tc : kc.impl()->tokens()) {
        if (tc.mode == AccessMode::Read || tc.mode == AccessMode::ReadWrite) {
          if (std::find(tainted_fields.begin(), tainted_fields.end(), tc.field_uid) !=
              tainted_fields.end()) {
            reads_tainted = true;
            break;
          }
        }
      }
      if (!reads_tainted) {
        continue;
      }
      map_forward_set.push_back(c);
      // Propagate the taint: c's writes are now downstream of stencil's
      // writes, so any successor reading c's outputs is also tainted.
      for (const AccessToken& tc : kc.impl()->tokens()) {
        if ((tc.mode == AccessMode::Write || tc.mode == AccessMode::ReadWrite) &&
            std::find(tainted_fields.begin(), tainted_fields.end(), tc.field_uid) ==
                tainted_fields.end()) {
          tainted_fields.push_back(tc.field_uid);
        }
      }
      fwd_queue.push_back(c);
    }
  }

  for (size_t c : map_forward_set) {
    if (already_split_forward[c]) {
      continue;
    }
    const size_t step_c_idx = step_for_kernel[c];
    Step& step_c = schedule_[step_c_idx];

    events_.push_back(sys_.event());
    const size_t c_internal_event = events_.size() - 1U;

    Step boundary_c = step_c;
    boundary_c.view = DataView::Boundary;
    boundary_c.wait_events.push_back(c_internal_event);

    step_c.view = DataView::Internal;
    step_c.record_event_idx = c_internal_event;
    step_c.wait_events.clear();

    schedule_.insert(schedule_.begin() + static_cast<std::ptrdiff_t>(step_c_idx) + 1, boundary_c);
    occ_bump_step_for_kernel_after_(step_c_idx, step_for_kernel);
    occ_bump_step_for_kernel_after_(step_c_idx, completion_step_for_kernel);
    completion_step_for_kernel[c] = step_c_idx + 1U;

    already_split_forward[c] = true;
  }
}

auto
Skeleton::export_dot(std::string graph_name) -> std::string {
  compile();

  auto fmt_mode = [](AccessMode m) -> const char* {
    switch (m) {
      case AccessMode::Read:
        return "R";
      case AccessMode::Write:
        return "W";
      case AccessMode::ReadWrite:
        return "RW";
    }
    return "?";
  };
  auto fmt_kind = [](KernelKind k) -> const char* {
    switch (k) {
      case KernelKind::Compute:
        return "compute";
      case KernelKind::Reduction:
        return "reduction";
      case KernelKind::Halo:
        return "halo";
      case KernelKind::Anchor:
        return "anchor";
      case KernelKind::ComputeHost:
        return "compute_host";
      case KernelKind::ReductionHost:
        return "reduction_host";
      case KernelKind::Universal:
        return "universal";
      case KernelKind::UniversalReduction:
        return "universal_reduction";
      case KernelKind::ScalarReadback:
        return "scalar_readback";
    }
    return "?";
  };
  std::unordered_map<std::uint64_t, std::string> uid_to_label;
  for (const Kernel& k : effective_kernels_) {
    for (const AccessToken& t : k.impl()->tokens()) {
      if (!t.field_name.empty()) {
        uid_to_label[t.field_uid] = t.field_name;
      }
    }
  }
  for (const auto& [uid, label] : field_label_overrides_) {
    uid_to_label[uid] = label;
  }
  auto field_label = [&uid_to_label](const AccessToken& t) -> std::string {
    auto it = uid_to_label.find(t.field_uid);
    if (it != uid_to_label.end() && !it->second.empty()) {
      return it->second;
    }
    return std::string{"<uid:"} + std::to_string(t.field_uid) + ">";
  };
  auto escape = [](std::string_view s) -> std::string {
    std::string out;
    out.reserve(s.size() + 4U);
    for (char c : s) {
      if (c == '"' || c == '\\') {
        out.push_back('\\');
      }
      out.push_back(c);
    }
    return out;
  };

  std::ostringstream os;
  os << "digraph " << graph_name << " {\n";
  os << "  rankdir=TB;\n";
  os << "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\"];\n";
  os << "  edge [fontname=\"Helvetica\", fontsize=10];\n";

  if (schedule_.empty()) {
    os << "  /* empty schedule */\n";
    os << "}\n";
    return os.str();
  }

  static constexpr std::array<std::string_view, 8> kStreamColors = {
      "#cfe2f3", "#fce5cd", "#d9ead3", "#f4cccc", "#ead1dc", "#fff2cc", "#d0e0e3", "#e6e6e6"};

  std::vector<std::vector<size_t>> steps_by_stream(streams_.size());
  for (size_t si = 0; si < schedule_.size(); ++si) {
    const size_t stream_idx = schedule_[si].stream_idx;
    if (stream_idx >= steps_by_stream.size()) {
      continue;
    }
    steps_by_stream[stream_idx].push_back(si);
  }

  for (size_t s = 0; s < streams_.size(); ++s) {
    const auto& bucket = steps_by_stream[s];
    if (bucket.empty()) {
      continue;
    }
    const std::string_view color = kStreamColors[s % kStreamColors.size()];
    os << "  subgraph cluster_stream_" << s << " {\n";
    if (s == copy_stream_idx_) {
      os << "    label=\"copy stream (detached, scalar readback)\";\n";
    } else {
      os << "    label=\"stream " << s << "\";\n";
    }
    os << "    style=\"rounded,dashed\";\n";
    os << "    color=\"#888888\";\n";
    os << "    bgcolor=\"" << color << "\";\n";
    for (size_t si : bucket) {
      const Step& step = schedule_[si];
      const Kernel& kernel = effective_kernels_[step.kernel_idx];
      const std::string_view raw_name = kernel.name();
      const std::string label = raw_name.empty()
                                    ? (std::string{"k"} + std::to_string(step.kernel_idx))
                                    : std::string{raw_name};
      os << "    s" << si << " [label=\"" << escape(label) << "\\n[" << fmt_kind(kernel.kind())
         << "/" << to_string(step.view) << "]\"";
      const auto& tokens = kernel.impl()->tokens();
      if (!tokens.empty()) {
        os << ", tooltip=\"";
        os << escape(label) << " tokens: ";
        for (size_t i = 0; i < tokens.size(); ++i) {
          if (i > 0) {
            os << ", ";
          }
          os << escape(field_label(tokens[i])) << ":" << fmt_mode(tokens[i].mode);
        }
        os << "\"";
      }
      os << "];\n";
    }
    os << "  }\n";
  }

  for (size_t si = 1; si < schedule_.size(); ++si) {
    os << "  s" << (si - 1U) << " -> s" << si << " [label=\"order\", color=\"#666666\"];\n";
  }

  std::vector<size_t> event_recorder(events_.size(), kNoKernel_);
  for (size_t si = 0; si < schedule_.size(); ++si) {
    const size_t evt = schedule_[si].record_event_idx;
    if (evt != kNoEvent_ && evt < event_recorder.size()) {
      event_recorder[evt] = si;
    }
  }
  auto dep_label = [&](size_t pred_step, size_t succ_step) -> std::string {
    const Kernel& pred_k = effective_kernels_[schedule_[pred_step].kernel_idx];
    const Kernel& succ_k = effective_kernels_[schedule_[succ_step].kernel_idx];
    for (const AccessToken& ta : pred_k.impl()->tokens()) {
      for (const AccessToken& tb : succ_k.impl()->tokens()) {
        if (ta.field_uid == tb.field_uid && conflicts(ta.mode, tb.mode)) {
          return std::string{field_label(ta)};
        }
      }
    }
    return "wait";
  };
  for (size_t sj = 0; sj < schedule_.size(); ++sj) {
    for (size_t evt : schedule_[sj].wait_events) {
      if (evt < event_recorder.size() && event_recorder[evt] != kNoKernel_) {
        const size_t si = event_recorder[evt];
        os << "  s" << si << " -> s" << sj << " [label=\"" << escape(dep_label(si, sj))
           << "\", style=dashed, color=\"#aa3333\"];\n";
      }
    }
  }

  os << "}\n";
  return os.str();
}

auto
Skeleton::export_dot_file(std::string path,
                          std::string graph_name) -> bool {
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) {
    return false;
  }
  out << export_dot(std::move(graph_name));
  return out.good();
}

auto
Skeleton::export_dot_png(std::string png_path,
                         std::string graph_name) -> bool {
  std::string dot_path = png_path;
  const size_t slash = dot_path.find_last_of("/\\");
  const size_t dot = dot_path.find_last_of('.');
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
    dot_path.replace(dot, dot_path.size() - dot, ".dot");
  } else {
    dot_path += ".dot";
  }
  if (!export_dot_file(dot_path, graph_name)) {
    return false;
  }
  const std::string cmd = "dot -Tpng \"" + dot_path + "\" -o \"" + png_path + "\" >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}

}  // namespace carbon
