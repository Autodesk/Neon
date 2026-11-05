#pragma once

#include <carbon/boundary_mode_distributed.h>
#include <carbon/dense_distributed.h>
#include <carbon/detail/allocation.h>
#include <carbon/detail/backend/select.h>
#include <carbon/detail/buffer_group.h>
#include <carbon/detail/dense/field.h>
#include <carbon/detail/halo_build.h>
#include <carbon/detail/halo_distributed_build.h>
#include <carbon/detail/halo_pack.h>
#include <carbon/logging.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/dist_transfer.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace carbon {

/**
 * Halo-exchange descriptor for a single @ref DenseDistributed::Field.
 *
 * @par What it does
 * Every @ref DenseDistributed chunk carries a per-axis halo (driven by the grid's @ref Stencil and
 * @ref ProcessGrid). Before any kernel that reads neighbours can run, those halo cells must be
 * filled from the adjacent chunk's owned cells. For a Slab this is one Z pass; Pencil adds Y;
 * Brick adds X. Multi-axis layouts run staged @c X -> @c Y -> @c Z passes so edge and corner
 * halos propagate. @c HaloDistributed precomputes the full per-pass pair set at construction and turns it
 * into:
 * - an intra-process @ref Transfer<T> covering every adjacent @b local chunk pair (carbon's
 *   single-process multi-GPU case, plus self-wrap pairs on a periodic axis);
 * - and, when @c CARBON_HAVE_MPI_NCCL is set and @c system.distributed() is non-null, a
 *   cross-rank @ref DistTransfer<T> covering every pair whose neighbour lives on a different
 *   rank (carbon's distributed one-rank-per-GPU case, plus wrap-around peers on a periodic
 *   axis).
 *
 * @ref operator() issues both on the user's stream — async, single-shot, same call ergonomics
 * as @c Transfer / @c DistTransfer.
 *
 * @par Host (CPU) variant
 * @ref launch_on_host runs the same exchange against the field's @c cpu_buffers via
 * synchronous @c std::memcpy and (when distributed) @c MPI_Sendrecv on host pointers
 * — the natural pair to @c DenseDistributed::make_host_kernel for fully-host pipelines. The host
 * variant lazily forces the field's host mirror to be allocated for
 * @c MemorySide::HostDevice fields (the device path keeps using @c gpu_buffers
 * unchanged). The @c HaloDistributedKernelImpl that backs @c Field::make_halo_kernel exposes
 * @c launch_on_host as well, so a heterogeneous pipeline that mixes host and device
 * kernels around a halo can call @c kernel.launch_on_host(stream) on the halo node
 * and have it Just Work without rebuilding the descriptor.
 *
 * @par Pair model
 * The legacy Slab path is built around an @b adjacency list of global z-slab index pairs
 * @c (g_lo, g_up). For every @c g in @c [0, global_slab_count - 1) it adds @c (g, g+1), plus
 * the wrap pair under @c PeriodicZ. The multi-axis path uses @ref ProcessGrid to find the
 * neighbour rank/coordinate on each axis and emits per-axis send/receive runs.
 *
 * Each pair is then dispatched based on which rank owns each end:
 * - @b Both local: emit two intra-process @c Transfer<T> entries (one upward, one downward),
 *   each entry contributing one copy per component plane. Self-pairs (one chunk wrapping to
 *   itself on a periodic axis) copy between non-overlapping offsets in the same buffer.
 * - @b One local, one remote: emit NCCL group entries per component run ( @c Send / @c Recv )
 *   to the peer rank.
 * - @b Both remote: skipped on this rank.
 *
 * Per-component planes are non-contiguous in carbon's SoA layout (separated by
 * @c span.linear_storage_elements()), so each @c Components plane contributes its own entry
 * on both the intra and dist sides.
 *
 * @par NCCL pairing order
 * Within each direction @c Send precedes @c Recv, and cross-rank entries are sorted
 * deterministically by @c (peer, op, (axis, source_rank, source_side, dest_rank, dest_side,
 * component, run_index)) so both ends of a same-peer connection, including the two directions
 * of a periodic @c dims == 2 axis, enqueue operations in a matching order. The run index is
 * required because one X/Y halo face fans out into multiple runs that share the same axis,
 * ranks, and component; without it @c std::sort could permute those runs differently on the
 * two ranks.
 *
 * @par BoundaryModeDistributed
 * - @ref BoundaryModeDistributed::Open: out-of-domain reads return @c valid = false; no wrap pair is
 *   added on that axis.
 * - @ref BoundaryModeDistributed::PeriodicX / @c PeriodicY / @c PeriodicZ: the corresponding axis wraps.
 *   On an axis with exactly one process-grid coordinate the wrap pair is a self-pair handled
 *   intra. Periodicity is independent per axis.
 *
 * @par Synchronisation contract
 * @c HaloDistributed issues async copies (intra) and grouped @c ncclSend / @c ncclRecv (dist) on the
 * user's stream. The caller must:
 * - ensure any kernel writing the @b source cells (own owned cells, both directions) has
 *   completed before @c operator() is called (e.g. by sharing the same stream or by event
 *   sync);
 * - and wait for the stream before any kernel that reads the halo cells (the natural
 *   ordering on the same stream is enough).
 *
 * Multi-axis layouts fence each staged pass before starting the next one (and synchronise the
 * distributed communicator between passes under MPI/NCCL), so the X result is visible to the Y
 * pass and Y to Z. Closing the last pass therefore covers the whole exchange.
 *
 * @note On multi-GPU layouts the "same stream is enough" rule does @b not by itself order a
 * peer copy against the @b source GPU's compute: a cross-device copy is enqueued on the
 * @b destination GPU's stream and carries no implicit ordering against pending writes on the
 * source GPU's stream (see @c StreamGroup::wait_for_all ). @c operator() therefore opens with
 * an internal cross-GPU producer fence — it records every per-GPU stream's tail and fan-in-
 * waits — so the copies are sequenced behind every GPU's prior compute, and the same fence on
 * the next call covers the copy-vs-next-overwrite WAR. Callers driving @c HaloDistributed on a single
 * stream get correct multi-GPU results without any extra event juggling; the fence collapses
 * to a no-op on single-GPU / no-halo layouts.
 *
 * @tparam T          Element type of the field (matches @c DenseDistributed::Field<T, Components>).
 * @tparam Components Per-cell component count (matches @c DenseDistributed::Field<T, Components>).
 */
template <typename T, int Components>
class HaloDistributed {
public:
  /** Empty descriptor; @ref operator() is a no-op (zero entries). */
  HaloDistributed() = default;

  /**
   * Build a @c StencilSemantic::Standard halo-exchange descriptor for @p field — every
   * component plane of every adjacency pair is moved as its own
   * @c cudaMemcpyAsync entry per direction (the aggregated layout has no cross-component
   * fusion, since cardinality planes sit @c linear_storage_elements apart).
   *
   * @throws std::invalid_argument when the field has no GPU buffers and the layout requires
   *         inter-slab exchange, or when the field's @ref BoundaryModeDistributed is unrecognised.
   */
  explicit HaloDistributed(DenseDistributed::Field<T,
                                                   Components> field)
      : HaloDistributed(std::move(field),
                        Stencil{},
                        StencilSemantic::Standard) {}

  /**
   * Build a halo-exchange descriptor with explicit @p semantic .
   *
   * @c StencilSemantic::Standard ignores @p stencil and is equivalent to the no-stencil
   * constructor. @c StencilSemantic::Lattice interprets @p stencil 's points as the discrete
   * velocities @c c_q of the field's @c Components populations and only sends populations
   * whose @c c_q component on the current axis matches the transfer direction. The aggregated
   * layout has no cross-cardinality fusion, so each axis pass moves the populations selected
   * by that axis (see @ref Stencil::lattice_profile ).
   *
   * @par Lattice pre-conditions (validated at construction)
   *
   * - @c stencil.size() must equal @c Components .
   * - @c stencil.radius() must be @c <= the corresponding per-axis halo thickness.
   * - @p semantic must be one of @c Standard / @c Lattice / @c LatticeDistributed .
   *
   * @throws std::invalid_argument on any of the above failing.
   */
  HaloDistributed(DenseDistributed::Field<T,
                                          Components> field,
                  Stencil stencil,
                  StencilSemantic semantic = StencilSemantic::Lattice);

  /** Field this descriptor is bound to. */
  [[nodiscard]] auto
  field() const noexcept -> const DenseDistributed::Field<T,
                                                          Components>& {
    return field_;
  }

  /** Boundary mode driving the exchange (read off the field's @c DenseDistributed at construction). */
  [[nodiscard]] auto
  boundary_mode() const noexcept -> BoundaryModeDistributed {
    return field_.dense().boundary_mode();
  }

  [[nodiscard]] auto
  stencil_semantic() const noexcept -> StencilSemantic {
    return semantic_;
  }

  [[nodiscard]] auto
  stencil() const noexcept -> const Stencil& {
    return stencil_;
  }

  /**
   * Number of intra-process transfer entries the exchange will issue across all staged axis
   * passes. Zero for a single GPU, an empty layout, no halo, or a distributed
   * one-rank-per-GPU layout where every adjacent pair is cross-rank.
   */
  [[nodiscard]] auto
  pair_count() const noexcept -> size_t {
    if (!axis_passes_.empty()) {
      size_t n = 0;
      for (const AxisPass& pass : axis_passes_) {
        n += pass.intra.src_counts().size();
      }
      return n;
    }
    return intra_.src_counts().size();
  }

  /**
   * Number of cross-rank @c Send / @c Recv NCCL group entries the exchange will issue. Equals
   * @c 2 * Components * peer_count (with @c peer_count in @c {0, 1, 2}) per active axis when
   * MPI is enabled and the field's @c System carries a @c Distributed handle; @c 0 in non-MPI
   * builds, in single-process layouts, or when no axis needs halo exchange.
   */
  [[nodiscard]] auto
  dist_entry_count() const noexcept -> size_t {
#ifdef CARBON_HAVE_MPI_NCCL
    if (!axis_passes_.empty()) {
      size_t n = 0;
      for (const AxisPass& pass : axis_passes_) {
        n += pass.dist.entry_count();
#if CARBON_BACKEND_IS_CUDALIKE
        n += pass.dist_staged.entry_count();
#endif
      }
      return n;
    }
    return dist_.entry_count();
#else
    return 0U;
#endif
  }

  /**
   * Issue the halo updates on @p stream. Dispatches the precomputed intra-process
   * @c Transfer<T> followed by the cross-rank @c DistTransfer<T> (the latter only in MPI
   * builds). Both are no-ops when their respective entry counts are zero, so this is safe to
   * call regardless of the layout.
   */
  auto
  operator()(const Stream& stream) const -> void {
    // Cross-GPU producer fence (see @c StreamGroup::wait_for_all). The intra peer
    // copies read each pair's SOURCE-GPU owned cells — written by prior compute on
    // the source GPU's stream — but @c cudaMemcpyPeerAsync enqueued on the
    // DESTINATION GPU's stream carries no implicit ordering against that source-side
    // compute. Record every per-GPU stream's current tail and fan-in-wait so each
    // copy is sequenced behind every GPU's prior writes; the same fence on the next
    // halo call transitively covers the copy-vs-next-overwrite WAR. Without this the
    // halo silently reads transient source bytes under cross-GPU timing pressure (a
    // genuine but contention-masked numerical drift, not just harness contention).
    // @c fence_event_ is empty for single-GPU / no-halo layouts, where the whole
    // block is skipped.
    if (!fence_event_.empty()) {
      fence_event_.record(stream);
      stream.wait_for_all(fence_event_);
    }
    if (!axis_passes_.empty()) {
      for (size_t i = 0; i < axis_passes_.size(); ++i) {
        const AxisPass& pass = axis_passes_[i];

        // Packed X/Y faces: gather -> [intra] -> one NCCL message per side -> scatter, all on
        // `stream`, so ordinary stream order (not a host barrier) sequences gather < comm < scatter
        // and this pass < the next pass. Empty for Z passes and for run-based fallbacks.
#if CARBON_BACKEND_IS_CUDALIKE
        run_pack_steps_(pass.gathers, stream, /*gather=*/true);
#endif
        pass.intra(stream);
#ifdef CARBON_HAVE_MPI_NCCL
        pass.dist(stream);
#if CARBON_BACKEND_IS_CUDALIKE
        pass.dist_staged(stream);
#endif
#endif
#if CARBON_BACKEND_IS_CUDALIKE
        run_pack_steps_(pass.scatters, stream, /*gather=*/false);
#endif

        if (i + 1U < axis_passes_.size()) {
          if (!pass.fence_event.empty()) {
            pass.fence_event.record(stream);
            stream.wait_for_all(pass.fence_event);
          }

#ifdef CARBON_HAVE_MPI_NCCL
          // Host-blocking barrier between axis passes. OFF by default; set
          // CARBON_HALO_PASS_BARRIER=1 to enable it. It should not be needed for
          // correctness (every pass is ordered on `stream`; intra copies are fenced above) and it
          // serialises the host against the GPU, which defeats overlap with interior compute.
          if (pass_barrier_) {
            if (const Distributed* distributed = field_.dense().system().distributed();
                distributed != nullptr) {
              stream.wait();
              distributed->barrier();
            }
          }
#endif
        }
      }
      return;
    }

    intra_(stream);
#ifdef CARBON_HAVE_MPI_NCCL
    dist_(stream);
#endif
  }

  /**
   * @brief Issue the halo updates on the field's @b host (CPU) mirror.
   *
   * Twin of @ref operator() for fully-host pipelines:
   * - Intra-process: @c std::memcpy between adjacent slabs' @c cpu_buffers using the
   *   same offset arithmetic as the device path (storage layout is identical
   *   between the two sides — the host mirror is always sized from the same
   *   @c span.linear_storage_elements() ).
   * - Cross-rank: @b currently @b unsupported. Calling @ref launch_on_host on a
   *   distributed @c HaloDistributed whose adjacency includes a non-local pair throws
   *   @c std::runtime_error . The MPI host @c Sendrecv path is a follow-up; the
   *   AGENTS.md memory bullet calls this out and tracks it.
   *
   * Synchronous: returns when every memcpy has completed (no async work is queued).
   * @p stream is waited on first so any prior device kernel writing to the field
   * has flushed to the CPU mirror via a preceding @c Transfer ; mirroring the
   * @ref carbon::detail::DenseHostKernelImpl stream contract.
   *
   * Triggers the lazy host-mirror allocation on @c MemorySide::HostDevice fields
   * via @c Field::cpu_buffers() — so a halo built ahead of any host-side touch
   * still finds the buffers populated when the user finally calls
   * @ref launch_on_host .
   *
   * @throws std::invalid_argument when the field has no host mirror to write into
   *         (built with @c MemorySide::Device with no lazy fault-in path because
   *         the side is not @c HostDevice ).
   * @throws std::runtime_error when @c host_dist_pending_ is non-zero — i.e. there
   *         is at least one cross-rank halo entry and the host MPI_Sendrecv
   *         fallback has not been wired in yet.
   */
  auto
  launch_on_host(const Stream& stream) const -> void {
    stream.wait();
    if (host_dist_pending_ > 0U) {
      log::throw_with_log<std::runtime_error>(
          "HaloDistributed::launch_on_host: cross-rank host halo is not yet supported (this halo "
          "has {} "
          "cross-rank slab entries). The host MPI_Sendrecv fallback is a follow-up tracked in "
          "AGENTS.md (\"HaloDistributed on host\" memory bullet); for now use launch (device) on a "
          "halo "
          "whose pairs cross ranks, or restrict the layout to one rank.",
          host_dist_pending_);
    }
    const bool multi_axis = !axis_passes_.empty();
    size_t host_entries = host_intra_.size();
    if (multi_axis) {
      host_entries = 0;
      for (const AxisPass& pass : axis_passes_) {
        host_entries += pass.host_intra.size();
      }
    }
    if (host_entries == 0U) {
      return;
    }

    auto& cpu = field_.cpu_buffers();
    if (cpu.size() == 0U) {
      log::throw_with_log<std::invalid_argument>(
          "HaloDistributed::launch_on_host: field has no host mirror (cpu_buffers().size() == 0); "
          "construct the field with MemorySide::Host or MemorySide::HostDevice");
    }

    const auto copy = [&](const HostIntraEntry& e) {
      T* const dst_ptr = cpu[e.dst_slab].data() + e.dst_offset;
      const T* const src_ptr = cpu[e.src_slab].data() + e.src_offset;
      std::memcpy(static_cast<void*>(dst_ptr), static_cast<const void*>(src_ptr),
                  e.count * sizeof(T));
    };

    if (multi_axis) {
      for (const AxisPass& pass : axis_passes_) {
        for (const HostIntraEntry& e : pass.host_intra) {
          copy(e);
        }
      }
    } else {
      for (const HostIntraEntry& e : host_intra_) {
        copy(e);
      }
    }
  }

  /**
   * @brief Number of intra-process @c std::memcpy entries the host launch will issue.
   *
   * Same shape as @ref pair_count but measured against the host pipeline. Equals
   * @ref pair_count today (the pair set is independent of which buffer side
   * receives the writes); kept as its own accessor so future asymmetric host /
   * device pair sets (e.g. when the host path lifts to @c MPI_Sendrecv ) have a
   * dedicated query.
   */
  [[nodiscard]] auto
  host_intra_count() const noexcept -> size_t {
    if (!axis_passes_.empty()) {
      size_t n = 0;
      for (const AxisPass& pass : axis_passes_) {
        n += pass.host_intra.size();
      }
      return n;
    }
    return host_intra_.size();
  }

private:
  /**
   * Per-direction intra-process host halo entry: a single @c std::memcpy of @c count
   * elements of @p T from @c src_slab 's buffer at @c src_offset (in elements) into
   * @c dst_slab 's buffer at @c dst_offset .
   *
   * Per-component planes contribute their own entry, same as the device intra
   * path. Storage layout matches between gpu and cpu buffers (both sized from
   * @c span.linear_storage_elements() ), so @c src_offset / @c dst_offset / @c count
   * are computed once at @c HaloDistributed construction and reused at every host launch.
   */
  struct HostIntraEntry {
    size_t src_slab{};
    size_t dst_slab{};
    size_t src_offset{};
    size_t dst_offset{};
    size_t count{};
  };

  struct AxisPass {
    Axis axis{Axis::X};
    Transfer<T> intra{};
#ifdef CARBON_HAVE_MPI_NCCL
    DistTransfer<T> dist{};
#endif
    std::vector<HostIntraEntry> host_intra{};
    mutable Event fence_event{};
#if CARBON_BACKEND_IS_CUDALIKE
    // Packed X/Y exchange (see detail/halo_pack_plan.h). Empty unless this pass was packed.
    std::vector<detail::halo_pack::PackStep> gathers{};
    std::vector<detail::halo_pack::PackStep> scatters{};
    std::vector<detail::halo_pack::PackMsg> staged_msgs{};
#ifdef CARBON_HAVE_MPI_NCCL
    DistTransfer<T> dist_staged{};  ///< bound to HaloDistributed::stage_, not to the field buffer
#endif
#endif
  };

  auto
  build_multi_axis_() -> void;

#if CARBON_BACKEND_IS_CUDALIKE
  /** Launch every gather (or scatter) kernel of one pass on @p stream. */
  auto
  run_pack_steps_(const std::vector<detail::halo_pack::PackStep>& steps,
                  const Stream& stream,
                  bool gather) const -> void {
    if (steps.empty()) {
      return;
    }
    const detail::Gpu& gpu = field_.dense().system().gpu_group().devices()[0];
    const detail::gpu_stream_t native = stream.native_at(0);
    T* field_base = field_.gpu_buffers()[0].data();
    T* stage_base = stage_[0].data();
    for (const detail::halo_pack::PackStep& step : steps) {
      const detail::gpu_error_t err =
          gather
              ? detail::halo_pack::launch_gather(gpu, native, sizeof(T), field_base,
                                                 stage_base + step.stage_offset, step.args)
              : detail::halo_pack::launch_scatter(
                    gpu, native, sizeof(T), stage_base + step.stage_offset, field_base, step.args);
      if (err != detail::gpu_success) {
        log::throw_with_log<std::runtime_error>(
            "HaloDistributed packed {} kernel launch failed: {}", gather ? "gather" : "scatter",
            detail::gpu_error_string(err));
      }
    }
  }
#endif

  DenseDistributed::Field<T, Components> field_{};
  Transfer<T> intra_{};
#ifdef CARBON_HAVE_MPI_NCCL
  DistTransfer<T> dist_{};
#endif
  // Per-GPU event group backing the cross-GPU producer fence in @ref operator().
  // Allocated only when the layout has intra-process peer pairs (multi-local-GPU);
  // stays empty (so the fence is skipped) for single-GPU / no-halo layouts. Mutable
  // because the fence records into it from the const @ref operator() — the group is a
  // shallow shared handle, so recording mutates the underlying events in place.
  mutable Event fence_event_{};
  std::vector<HostIntraEntry> host_intra_{};
  std::vector<AxisPass> axis_passes_{};
#if CARBON_BACKEND_IS_CUDALIKE
  // One contiguous staging buffer shared by every packed pass (single local GPU only). Allocated
  // in the constructor so no allocation can land inside graph capture or a timed region.
  Buffer<T> stage_{};
#endif
  // See operator(): host barrier between axis passes. CARBON_HALO_PASS_BARRIER=1 enables it.
  bool pass_barrier_{false};
  Stencil stencil_{};
  StencilSemantic semantic_{StencilSemantic::Standard};
  // Number of cross-rank slab pairs that the host pipeline would need to handle.
  // Non-zero in distributed runs; consulted by @ref launch_on_host so we throw a
  // clean diagnostic instead of silently dropping the cross-rank halo on the
  // floor. Once the @c MPI_Sendrecv fallback lands this becomes the count of
  // host-MPI entries instead of a "todo" sentinel.
  size_t host_dist_pending_{};
};

// ---------------------------------------------------------------------------
// HaloDistributed<T, Components>::HaloDistributed — descriptor builder.
//
// Behaviour-preserving split of the original ~262 LOC ctor body into a thin
// orchestrator that delegates to phase helpers in `detail/halo_build.h` and
// keeps only the DenseDistributed-specific per-pair offset arithmetic inline. See the
// fourth-pass audit's P1.α resolution narrative for the design rationale and
// the symmetric `HaloDisg` ctor refactor.
// ---------------------------------------------------------------------------
template <typename T,
          int Components>
HaloDistributed<T,
                Components>::HaloDistributed(DenseDistributed::Field<T,
                                                                     Components> field,
                                             Stencil stencil,
                                             StencilSemantic semantic)
    : field_(std::move(field)),
      stencil_(std::move(stencil)),
      semantic_(semantic) {
  // ---------- Phase 1: validate semantic + boundary + lattice radius ----------
  detail::halo_build::validate_stencil_semantic(semantic_, "HaloDistributed");

  const DenseDistributed& dense = field_.dense();
  const auto& spans = dense.spans();
  const auto& gpu_buffers = field_.gpu_buffers();
  const size_t n_gpus = spans.size();

  // ---------- Phase 2: short-circuit on empty / no-halo layouts ----------
  if (n_gpus == 0U) {
    // Empty layout: no slabs to exchange. intra_/dist_ stay default-constructed.
    return;
  }

  const bool legacy_slab = dense.process_grid().is_slab() &&
                           !is_periodic(dense.boundary_mode(), Axis::X) &&
                           !is_periodic(dense.boundary_mode(), Axis::Y);

  // The legacy slab builder emits one cross-rank message per component per direction. For a
  // distributed Z-slab with several components per face, hand the layout to the axis builder
  // instead: it packs each face into one message per side (a single Z pass). Single-process
  // multi-GPU slabs keep the legacy peer-copy path.
#if CARBON_BACKEND_IS_CUDALIKE
  const bool slab_via_axis_builder = legacy_slab && Components > 1 && n_gpus == 1U &&
                                     dense.system().distributed() != nullptr &&
                                     spans[0].halo().z > 0 && gpu_buffers.size() == n_gpus &&
                                     detail::halo_pack::env_flag("CARBON_HALO_PACK", true) &&
                                     detail::halo_pack::env_flag("CARBON_HALO_PACK_Z", true) &&
                                     detail::halo_pack::element_size_supported(sizeof(T)) &&
                                     dense.system().gpu_group().size() == 1U;
#else
  constexpr bool slab_via_axis_builder = false;
#endif

  if (!legacy_slab || slab_via_axis_builder) {
    detail::halo_distributed_build::validate_boundary_mode_bits(dense.boundary_mode(),
                                                                "HaloDistributed");

    const int64_3d halo = spans[0].halo();

    if (is_lattice_semantic(semantic_)) {
      if (stencil_.size() != static_cast<size_t>(Components)) {
        log::throw_with_log<std::invalid_argument>(
            "HaloDistributed(lattice): stencil.size() = {} but Components = {}; the lattice halo "
            "pairs "
            "population q with stencil.points()[q] and requires equal counts",
            stencil_.size(), Components);
      }

      const int32_3d r = stencil_.radius3();
      if ((halo.x > 0 && static_cast<int64_t>(r.x) > halo.x) ||
          (halo.y > 0 && static_cast<int64_t>(r.y) > halo.y) ||
          (halo.z > 0 && static_cast<int64_t>(r.z) > halo.z)) {
        log::throw_with_log<std::invalid_argument>(
            "HaloDistributed(lattice): stencil radius3 ({}, {}, {}) exceeds halo ({}, {}, {})", r.x,
            r.y, r.z, halo.x, halo.y, halo.z);
      }
    }

    if (halo.x > 0 || halo.y > 0 || halo.z > 0) {
      build_multi_axis_();
    }
    return;
  }

  // Reaching the legacy branch means X and Y are open; reject stray mask bits, then read Z.
  detail::halo_distributed_build::validate_boundary_mode_bits(dense.boundary_mode(),
                                                              "HaloDistributed");
  const bool periodic_z = is_periodic(dense.boundary_mode(), Axis::Z);

  const int64_t halo_z = spans[0].halo_z();
  if (halo_z <= 0) {
    // Stencil radius 0: there is no halo region in storage to exchange.
    return;
  }

  if (is_lattice_semantic(semantic_)) {
    if (stencil_.size() != static_cast<size_t>(Components)) {
      log::throw_with_log<std::invalid_argument>(
          "HaloDistributed(lattice): stencil.size() = {} but Components = {}; the lattice halo "
          "pairs "
          "population q with stencil.points()[q] and requires equal counts",
          stencil_.size(), Components);
    }
    if (stencil_.radius() > static_cast<int>(halo_z)) {
      log::throw_with_log<std::invalid_argument>(
          "HaloDistributed(lattice): stencil radius {} > halo_z {} (each lattice halo entry copies "
          "halo_z xy-planes; the field's halo is too thin for this stencil)",
          stencil_.radius(), halo_z);
    }
  }

  // Cardinality activity filter: the only direction-relevant axis on a z-partitioned grid
  // is z (mirroring Neon's lattice halo). Standard semantic keeps every cardinality.
  // LatticeDistributed sends both nonzero z signs in both directions so a wall
  // bounce-back can read opp(q) across a z-split rank boundary.
  const bool bidirectional_z = semantic_ == StencilSemantic::LatticeDistributed;
  auto active_up = [&](int c) noexcept -> bool {
    if (semantic_ == StencilSemantic::Standard) {
      return true;
    }
    const int64_t z = stencil_.points()[static_cast<size_t>(c)].z;
    return bidirectional_z ? z != 0 : z > 0;
  };
  auto active_down = [&](int c) noexcept -> bool {
    if (semantic_ == StencilSemantic::Standard) {
      return true;
    }
    const int64_t z = stencil_.points()[static_cast<size_t>(c)].z;
    return bidirectional_z ? z != 0 : z < 0;
  };

  // Host-only fields (MemorySide::Host) carry no GPU buffers; the host_intra_
  // metadata is still populated and @ref launch_on_host works. The device path
  // (operator() / launch_on_device) checks for a populated intra_/dist_ at call
  // time, so the only failure mode is "user constructed a host-only field but
  // tried to launch the halo on a device kernel" — which trips the empty-Transfer
  // path naturally. We only reject the cross-rank case below where gpu_buffers
  // are required for NCCL.
  const bool has_gpu_buffers = (gpu_buffers.size() == n_gpus);

  // ---------- Phase 3: build adjacency list ----------
  const size_t global_n = dense.global_slab_count();
  const size_t local_offset = dense.local_slab_offset();
  const auto adjacency = detail::halo_build::adjacency_pairs(global_n, periodic_z);

  // Legacy Slab path: every local span shares the full x/y extent and pitch. Cache common
  // values off the local span (used for both intra and dist offset arithmetic).
  const int64_t pitch = spans[0].xy_slab_elements();
  const size_t halo_count = static_cast<size_t>(halo_z * pitch);

  auto is_local = [&](size_t g) noexcept -> bool {
    return g >= local_offset && g < local_offset + n_gpus;
  };

  // ---------- Phase 4: walk adjacency, classify pairs, accumulate entries ----------
  //
  // Intra-process entries cover every adjacency pair where both endpoints are local
  // (single-process multi-GPU + PeriodicZ self-wrap on a single-slab layout).
  // Cross-rank entries cover pairs with exactly one local endpoint; one Send + one
  // Recv per component plane per pair. Send/Recv FIFO ordering is preserved by
  // issuing Send before Recv within each direction and iterating components in
  // ascending order on every rank, so NCCL's per-peer FIFO matches automatically.
  detail::halo_build::IntraTransferBuilder<T> intra;
#ifdef CARBON_HAVE_MPI_NCCL
  detail::halo_build::DistTransferBuilder<T> dist;
#endif

  for (const auto& [g_lo, g_up] : adjacency) {
    const bool lo_local = is_local(g_lo);
    const bool up_local = is_local(g_up);
    if (!lo_local && !up_local) {
      continue;  // Foreign pair; another rank handles it (or no one in non-MPI builds).
    }

    if (lo_local && up_local) {
      // Both endpoints local: emit two intra entries per component (upward + downward).
      const size_t i_lo = g_lo - local_offset;
      const size_t i_up = g_up - local_offset;
      const auto& span_lo = spans[i_lo];
      const auto& span_up = spans[i_up];
      const int64_t lim_z_lo = span_lo.logic_extent().z;
      const int64_t per_comp_lo = span_lo.linear_storage_elements();
      const int64_t per_comp_up = span_up.linear_storage_elements();

      for (int c = 0; c < Components; ++c) {
        const size_t cc = static_cast<size_t>(c);
        // Upward: slab g_lo's last halo_z owned -> slab g_up's lower halo. Skipped under
        // lattice semantic when stencil[c].z <= 0.
        if (active_up(c)) {
          const size_t up_src_off =
              cc * static_cast<size_t>(per_comp_lo) + static_cast<size_t>(lim_z_lo * pitch);
          const size_t up_dst_off = cc * static_cast<size_t>(per_comp_up);
          if (has_gpu_buffers) {
            intra.push(gpu_buffers[i_lo], gpu_buffers[i_up], up_src_off, up_dst_off, halo_count);
          }
          host_intra_.push_back({i_lo, i_up, up_src_off, up_dst_off, halo_count});
        }

        // Downward: slab g_up's first halo_z owned -> slab g_lo's upper halo. Skipped under
        // lattice semantic when stencil[c].z >= 0.
        if (active_down(c)) {
          const size_t dn_src_off =
              cc * static_cast<size_t>(per_comp_up) + static_cast<size_t>(halo_z * pitch);
          const size_t dn_dst_off = cc * static_cast<size_t>(per_comp_lo) +
                                    static_cast<size_t>((halo_z + lim_z_lo) * pitch);
          if (has_gpu_buffers) {
            intra.push(gpu_buffers[i_up], gpu_buffers[i_lo], dn_src_off, dn_dst_off, halo_count);
          }
          host_intra_.push_back({i_up, i_lo, dn_src_off, dn_dst_off, halo_count});
        }
      }
      continue;
    }

    // Adjacency pair has exactly one local endpoint — handled by the dist branch
    // below (under MPI). Track for the host launch so we can throw a pointed
    // "host MPI halo not implemented" diagnostic on launch_on_host instead of
    // silently dropping the cross-rank update on the floor.
    ++host_dist_pending_;

#ifdef CARBON_HAVE_MPI_NCCL
    // The cross-rank branch routes through NCCL on the device buffers; host-only
    // fields can't take this path today (the host MPI_Sendrecv fallback is the
    // tracked follow-up).
    if (!has_gpu_buffers) {
      log::throw_with_log<std::invalid_argument>(
          "HaloDistributed: cross-rank halo on a host-only field (MemorySide::Host) is not yet "
          "supported; "
          "use MemorySide::HostDevice (or single-rank layout)");
    }
    // Exactly one endpoint is local: the other is on a peer rank. Under carbon's
    // one-rank-per-GPU model peer rank == peer global slab index.
    const size_t local_g = lo_local ? g_lo : g_up;
    const size_t peer_g = lo_local ? g_up : g_lo;
    const int peer = static_cast<int>(peer_g);
    const size_t i_local = local_g - local_offset;
    const auto& span = spans[i_local];
    const int64_t lim_z = span.logic_extent().z;
    const int64_t per_comp = span.linear_storage_elements();

    // lo_local: we are the LOWER of the pair, peer is above us. Send our last halo_z owned,
    //           Recv into our upper halo.
    // up_local: we are the UPPER of the pair, peer is below us. Send our first halo_z owned,
    //           Recv into our lower halo.
    const size_t send_offset_base =
        lo_local ? static_cast<size_t>(lim_z * pitch) : static_cast<size_t>(halo_z * pitch);
    const size_t recv_offset_base =
        lo_local ? static_cast<size_t>((halo_z + lim_z) * pitch) : static_cast<size_t>(0);

    // Lattice activity. lo_local sends "upward" populations (c.z > 0) and receives
    // "downward" populations (c.z < 0); up_local is the mirror.
    for (int c = 0; c < Components; ++c) {
      const bool send_active = lo_local ? active_up(c) : active_down(c);
      const bool recv_active = lo_local ? active_down(c) : active_up(c);
      if (!send_active && !recv_active) {
        continue;
      }
      const size_t plane = static_cast<size_t>(c) * static_cast<size_t>(per_comp);
      if (send_active) {
        dist.push(peer, DistOp::Send, plane + send_offset_base, halo_count);
      }
      if (recv_active) {
        dist.push(peer, DistOp::Recv, plane + recv_offset_base, halo_count);
      }
    }
#else
    // Adjacency pair has exactly one local endpoint, but Carbon was built without MPI -- a
    // distributed DenseDistributed should have been rejected earlier; reaching here means a misconfigured
    // layout.
    log::throw_with_log<std::invalid_argument>(
        "HaloDistributed: adjacency pair (g_lo={}, g_up={}) has a foreign endpoint, but Carbon was "
        "built "
        "without CARBON_ENABLE_MPI=ON; build with MPI to enable cross-rank halo exchange",
        g_lo, g_up);
#endif
  }

  // ---------- Phase 5: finalise the accumulated entries into Transfer / DistTransfer ----------
  intra_ = intra.finalise(dense.system());
#ifdef CARBON_HAVE_MPI_NCCL
  dist_ = dist.finalise(dense.system(), gpu_buffers);
#endif

  // ---------- Phase 6: allocate the cross-GPU producer fence ----------
  // Only multi-local-GPU layouts have intra peer copies that can race the source
  // GPU's prior compute; single-GPU / no-halo layouts leave fence_event_ empty so
  // operator() skips the fence entirely. System::event() is non-const (it appends to
  // the System's event history), and dense.system() is a const ref, so take a shallow
  // System copy to allocate from.
  if (intra_.src_counts().size() > 0U) {
    System sys = dense.system();
    fence_event_ = sys.event();
  }
}

template <typename T,
          int Components>
auto
HaloDistributed<T,
                Components>::build_multi_axis_() -> void {
  using detail::halo_distributed_build::axis_halo_runs;
  using detail::halo_distributed_build::axis_neighbour;
  using detail::halo_distributed_build::axis_owned_face_runs;
  using detail::halo_distributed_build::axis_value;
  using detail::halo_distributed_build::AxisGeometry;
  using detail::halo_distributed_build::AxisSide;

  const DenseDistributed& dense = field_.dense();
  const ProcessGrid& pg = dense.process_grid();
  const auto& spans = dense.spans();
  const auto& gpu_buffers = field_.gpu_buffers();
  const bool has_gpu_buffers = gpu_buffers.size() == spans.size();

#ifdef CARBON_HAVE_MPI_NCCL
  const Distributed* distributed_handle = dense.system().distributed();
  const bool distributed = distributed_handle != nullptr;
  const int world_rank = distributed ? distributed_handle->world_rank() : 0;
#else
  const bool distributed = false;
  const int world_rank = 0;
#endif

  const auto geom_of = [](const DenseDistributed::Span& s) -> AxisGeometry {
    return AxisGeometry{
        s.logic_extent(),
        s.halo(),
        s.haloed_extent(),
        s.xy_slab_elements(),
    };
  };

  const auto velocity = [&](int c, Axis axis) -> int64_t {
    return axis_value(stencil_.points()[static_cast<size_t>(c)], axis);
  };

  const bool bidirectional = semantic_ == StencilSemantic::LatticeDistributed;
  const auto active_send = [&](int c, Axis axis, AxisSide side) -> bool {
    if (semantic_ == StencilSemantic::Standard) {
      return true;
    }
    const int64_t v = velocity(c, axis);
    if (bidirectional) {
      return v != 0;
    }
    return side == AxisSide::Upper ? v > 0 : v < 0;
  };

  const auto active_recv = [&](int c, Axis axis, AxisSide side) -> bool {
    if (semantic_ == StencilSemantic::Standard) {
      return true;
    }
    const int64_t v = velocity(c, axis);
    if (bidirectional) {
      return v != 0;
    }
    return side == AxisSide::Upper ? v < 0 : v > 0;
  };

  pass_barrier_ = detail::halo_pack::env_flag("CARBON_HALO_PASS_BARRIER", false);

#if CARBON_BACKEND_IS_CUDALIKE
  // Packed X/Y faces: one message per side instead of one per contiguous row. Needs exactly one
  // local GPU (every distributed rank, or a single-GPU process) and a plain 1/2/4/8/16-byte T.
  // CARBON_HALO_PACK=0 forces the run-based path (A/B testing, debugging).
  const bool pack_ok = detail::halo_pack::env_flag("CARBON_HALO_PACK", true) &&
                       detail::halo_pack::element_size_supported(sizeof(T)) && has_gpu_buffers &&
                       spans.size() == 1U && dense.system().gpu_group().size() == 1U;
  // A Z face is already contiguous PER COMPONENT, so packing only pays when a face carries
  // several components: otherwise it costs one NCCL message per component per direction (76 per
  // rank per exchange for D3Q19 Standard), and the ones crossing a node boundary go through the
  // network proxy back to back. CARBON_HALO_PACK_Z=0 restores the per-component Z path.
  const bool pack_z =
      pack_ok && Components > 1 && detail::halo_pack::env_flag("CARBON_HALO_PACK_Z", true);
  size_t stage_total = 0;
#endif

#ifdef CARBON_HAVE_MPI_NCCL
  struct DistEntry {
    int peer;
    DistOp op;
    size_t first;
    size_t count;
    std::tuple<int, int, int, int, int, int, size_t> key;
  };
#endif

  for (const Axis axis : {Axis::X, Axis::Y, Axis::Z}) {
    if (axis_value(spans[0].halo(), axis) <= 0) {
      continue;
    }

    AxisPass pass;
    pass.axis = axis;

    // X/Y faces are strided, so they are always packed. A Z face is one contiguous run per
    // component: it is packed only when it carries several components (see pack_z above).
    bool pack_axis = false;
#if CARBON_BACKEND_IS_CUDALIKE
    detail::halo_pack::AxisPackPlan pack_plan;
    if (pack_ok && (axis == Axis::X || axis == Axis::Y || (axis == Axis::Z && pack_z))) {
      pack_axis = true;
      const int32_3d pcoords = distributed ? pg.coords_of(world_rank) : pg.coords_of(0);
      const int32_t prank = distributed ? world_rank : 0;
      pack_plan = detail::halo_pack::build_axis_pack_plan(
          pg, dense.boundary_mode(), pcoords, prank, geom_of(spans[0]), axis, Components,
          static_cast<int64_t>(spans[0].linear_storage_elements()), active_send, active_recv,
          stage_total);
      stage_total += pack_plan.stage_elems;
    }
#endif

    detail::halo_build::IntraTransferBuilder<T> intra;
#ifdef CARBON_HAVE_MPI_NCCL
    detail::halo_build::DistTransferBuilder<T> dist;
    std::vector<DistEntry> dist_entries;
#endif

    for (size_t i = 0; i < spans.size(); ++i) {
      const DenseDistributed::Span& dst_span = spans[i];
      const int32_3d coords =
          distributed ? pg.coords_of(world_rank) : pg.coords_of(static_cast<int32_t>(i));
      const int32_t local_rank = distributed ? world_rank : static_cast<int32_t>(i);

      const AxisGeometry dst_geom = geom_of(dst_span);
      const int64_t dst_per_card = dst_span.linear_storage_elements();

      for (const AxisSide side : {AxisSide::Lower, AxisSide::Upper}) {
        const auto nb = axis_neighbour(pg, coords, axis, dense.boundary_mode(), side);
        if (!nb.present) {
          continue;
        }

        const auto dst_runs = axis_halo_runs(dst_geom, axis, side);
        const AxisSide src_side = side == AxisSide::Lower ? AxisSide::Upper : AxisSide::Lower;

        if (!distributed || nb.rank == local_rank) {
          const size_t src_slab = distributed ? i : static_cast<size_t>(nb.rank);
          const AxisGeometry src_geom = geom_of(spans[src_slab]);
          const auto src_runs = axis_owned_face_runs(src_geom, axis, src_side);

          if (src_runs.size() != dst_runs.size()) {
            log::throw_with_log<std::invalid_argument>(
                "HaloDistributed: axis run cardinality mismatch ({} vs {})", src_runs.size(),
                dst_runs.size());
          }

          const int64_t src_per_card = spans[src_slab].linear_storage_elements();

          for (size_t r = 0; r < dst_runs.size(); ++r) {
            if (src_runs[r].count != dst_runs[r].count) {
              log::throw_with_log<std::invalid_argument>(
                  "HaloDistributed: axis run count mismatch ({} vs {})", src_runs[r].count,
                  dst_runs[r].count);
            }

            for (int c = 0; c < Components; ++c) {
              if (!active_recv(c, axis, side)) {
                continue;
              }

              const size_t src_off =
                  static_cast<size_t>(c) * static_cast<size_t>(src_per_card) + src_runs[r].offset;
              const size_t dst_off =
                  static_cast<size_t>(c) * static_cast<size_t>(dst_per_card) + dst_runs[r].offset;

              if (has_gpu_buffers && !pack_axis) {
                intra.push(gpu_buffers[src_slab], gpu_buffers[i], src_off, dst_off,
                           dst_runs[r].count);
              }

              pass.host_intra.push_back({
                  src_slab,
                  i,
                  src_off,
                  dst_off,
                  dst_runs[r].count,
              });
            }
          }
          continue;
        }

#ifdef CARBON_HAVE_MPI_NCCL
        ++host_dist_pending_;

        if (!has_gpu_buffers) {
          log::throw_with_log<std::invalid_argument>(
              "HaloDistributed: cross-rank halo on a host-only field is not supported");
        }

        if (pack_axis) {
          continue;  // this side is carried by the packed messages in pack_plan
        }

        const auto src_runs = axis_owned_face_runs(dst_geom, axis, side);
        if (src_runs.size() != dst_runs.size()) {
          log::throw_with_log<std::invalid_argument>(
              "HaloDistributed: axis run cardinality mismatch ({} vs {})", src_runs.size(),
              dst_runs.size());
        }

        for (size_t r = 0; r < dst_runs.size(); ++r) {
          if (src_runs[r].count != dst_runs[r].count) {
            log::throw_with_log<std::invalid_argument>(
                "HaloDistributed: axis run count mismatch ({} vs {})", src_runs[r].count,
                dst_runs[r].count);
          }

          const int axis_id = static_cast<int>(axis);
          const int local_side = static_cast<int>(side);
          const int peer_side = 1 - local_side;

          for (int c = 0; c < Components; ++c) {
            const size_t plane = static_cast<size_t>(c) * static_cast<size_t>(dst_per_card);

            if (active_send(c, axis, side)) {
              dist_entries.push_back({
                  static_cast<int>(nb.rank),
                  DistOp::Send,
                  plane + src_runs[r].offset,
                  src_runs[r].count,
                  {
                      axis_id,
                      static_cast<int>(local_rank),
                      local_side,
                      static_cast<int>(nb.rank),
                      peer_side,
                      c,
                      r,
                  },
              });
            }
            if (active_recv(c, axis, side)) {
              dist_entries.push_back({
                  static_cast<int>(nb.rank),
                  DistOp::Recv,
                  plane + dst_runs[r].offset,
                  dst_runs[r].count,
                  {
                      axis_id,
                      static_cast<int>(nb.rank),
                      peer_side,
                      static_cast<int>(local_rank),
                      local_side,
                      c,
                      r,
                  },
              });
            }
          }
        }
#else
        log::throw_with_log<std::invalid_argument>(
            "HaloDistributed: multi-axis cross-rank exchange requires CARBON_ENABLE_MPI=ON");
#endif
      }
    }

    pass.intra = intra.finalise(dense.system());
#ifdef CARBON_HAVE_MPI_NCCL
    // NCCL P2P has no tags: keep both sides' per-peer operation order identical.
    std::sort(dist_entries.begin(), dist_entries.end(), [](const DistEntry& a, const DistEntry& b) {
      return std::tie(a.peer, a.op, a.key) < std::tie(b.peer, b.op, b.key);
    });
    for (const DistEntry& entry : dist_entries) {
      dist.push(entry.peer, entry.op, entry.first, entry.count);
    }
    pass.dist = dist.finalise(dense.system(), gpu_buffers);
#endif

    if (!pass.intra.src_counts().empty()) {
      System sys = dense.system();
      pass.fence_event = sys.event();
    }

#if CARBON_BACKEND_IS_CUDALIKE
    if (pack_axis) {
      pass.gathers = std::move(pack_plan.gathers);
      pass.scatters = std::move(pack_plan.scatters);
      pass.staged_msgs = std::move(pack_plan.msgs);
    }
#endif

    axis_passes_.push_back(std::move(pass));
  }

  for (const AxisPass& pass : axis_passes_) {
    if (!pass.intra.src_counts().empty()) {
      System sys = dense.system();
      fence_event_ = sys.event();
      break;
    }
  }

#if CARBON_BACKEND_IS_CUDALIKE
  if (stage_total > 0U) {
    stage_ = dense.system().template buffer<T>(std::vector<size_t>{stage_total});
#ifdef CARBON_HAVE_MPI_NCCL
    for (AxisPass& pass : axis_passes_) {
      if (pass.staged_msgs.empty()) {
        continue;
      }
      // Same rule as the run-based path: NCCL P2P has no tags, so both ends of every peer pair must
      // issue operations in the same order (all sends before recvs, ordered by a symmetric key).
      detail::halo_pack::sort_msgs(pass.staged_msgs);
      detail::halo_build::DistTransferBuilder<T> staged;
      for (const detail::halo_pack::PackMsg& m : pass.staged_msgs) {
        staged.push(m.peer, m.op == detail::halo_pack::MsgOp::Send ? DistOp::Send : DistOp::Recv,
                    m.first, m.count);
      }
      pass.dist_staged = staged.finalise(dense.system(), stage_);
    }
#endif
  }
#endif
}

/**
 * CTAD guide: @c HaloDistributed(field) deduces @c HaloDistributed<T, Components> from a @c DenseDistributed::Field<T, Components>.
 */
template <typename T,
          int Components>
HaloDistributed(DenseDistributed::Field<T,
                                        Components>) -> HaloDistributed<T,
                                                                        Components>;

template <typename T,
          int Components>
HaloDistributed(DenseDistributed::Field<T,
                                        Components>,
                Stencil,
                StencilSemantic) -> HaloDistributed<T,
                                                    Components>;

}  // namespace carbon
