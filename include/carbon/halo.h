#pragma once

#include <carbon/boundary_mode.h>
#include <carbon/dense.h>
#include <carbon/detail/allocation.h>
#include <carbon/detail/buffer_group.h>
#include <carbon/detail/dense/field.h>
#include <carbon/detail/halo_build.h>
#include <carbon/logging.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/dist_transfer.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace carbon {

/**
 * Halo-exchange descriptor for a single @ref Dense::Field.
 *
 * @par What it does
 * Carbon's @ref Dense slabs each carry an @c halo_z thickness on each z side (driven by the
 * grid's @ref Stencil). Before any kernel that reads neighbours can run, those halo cells
 * must be filled with the adjacent slab's owned cells. @c Halo precomputes the full pair set
 * at construction and turns it into:
 * - an intra-process @ref Transfer<T> covering every adjacent @b local slab pair (carbon's
 *   single-process multi-GPU case, plus self-wrap pairs for @c PeriodicZ);
 * - and, when @c CARBON_HAVE_MPI_NCCL is set and @c system.distributed() is non-null, a
 *   cross-rank @ref DistTransfer<T> covering the pairs whose neighbour lives on a different
 *   rank (carbon's distributed one-rank-per-GPU case, plus the wrap-around peer pair when
 *   @c PeriodicZ).
 *
 * @ref operator() issues both on the user's stream — async, single-shot, same call ergonomics
 * as @c Transfer / @c DistTransfer.
 *
 * @par Host (CPU) variant
 * @ref launch_on_host runs the same exchange against the field's @c cpu_buffers via
 * synchronous @c std::memcpy and (when distributed) @c MPI_Sendrecv on host pointers
 * — the natural pair to @c Dense::make_host_kernel for fully-host pipelines. The host
 * variant lazily forces the field's host mirror to be allocated for
 * @c MemorySide::HostDevice fields (the device path keeps using @c gpu_buffers
 * unchanged). The @c HaloKernelImpl that backs @c Field::make_halo_kernel exposes
 * @c launch_on_host as well, so a heterogeneous pipeline that mixes host and device
 * kernels around a halo can call @c kernel.launch_on_host(stream) on the halo node
 * and have it Just Work without rebuilding the descriptor.
 *
 * @par Pair model
 * The exchange is built around an @b adjacency list of global slab index pairs @c (g_lo, g_up)
 * where @c g_up is @c g_lo's z-upward neighbour:
 * - For every @c g in @c [0, global_slab_count - 1) we add @c (g, g+1).
 * - For @c BoundaryMode::PeriodicZ we additionally add @c (global_slab_count - 1, 0) — the
 *   wrap pair. In @c BoundaryMode::Open this pair is omitted, and the top/bottom global
 *   slabs' exterior halos are left untouched (kernels see @c Partition::neighbour return
 *   @c valid = false there).
 *
 * Each pair is then dispatched based on which rank owns each end:
 * - @b Both local: emit two intra-process @c Transfer<T> entries (one upward, one downward),
 *   each entry contributing one @c cudaMemcpyAsync per component plane. Self-pairs (one slab
 *   wrapping to itself in @c PeriodicZ on a single-slab layout) work the same way and copy
 *   between non-overlapping offsets in the same buffer.
 * - @b One local, one remote: emit two NCCL group entries per component plane (one @c Send,
 *   one @c Recv) to the peer rank.
 * - @b Both remote: skipped on this rank.
 *
 * Per-component planes are non-contiguous in carbon's SoA layout (separated by
 * @c span.linear_storage_elements()), so each @c Components plane contributes its own entry
 * on both the intra and dist sides.
 *
 * @par NCCL pairing order
 * Within each direction we issue @c Send before @c Recv on every rank, and components are
 * iterated in ascending order on every rank, so NCCL's per-@c (src,dst) FIFO matches the
 * flows automatically without explicit tags or barriers.
 *
 * @par BoundaryMode
 * - @ref BoundaryMode::Open: out-of-domain reads return @c valid = false; the wrap pair is
 *   omitted.
 * - @ref BoundaryMode::PeriodicZ: the z-axis wraps. The wrap pair is added; on a single
 *   global slab (@c global_slab_count == 1) it becomes a self-pair handled intra. X and Y
 *   stay Open (carbon's storage halo is z-only).
 *
 * @par Synchronisation contract
 * @c Halo issues async @c cudaMemcpyAsync (intra) and grouped @c ncclSend / @c ncclRecv
 * (dist) on the user's stream. The caller must:
 * - ensure any kernel writing the @b source cells (own owned cells, both directions) has
 *   completed before @c operator() is called (e.g. by sharing the same stream or by event
 *   sync);
 * - and wait for the stream before any kernel that reads the halo cells (the natural
 *   ordering on the same stream is enough).
 *
 * @note On multi-GPU layouts the "same stream is enough" rule does @b not by itself order a
 * peer copy against the @b source GPU's compute: @c cudaMemcpyPeerAsync is enqueued on the
 * @b destination GPU's stream and carries no implicit ordering against pending writes on the
 * source GPU's stream (see @c StreamGroup::wait_for_all ). @c operator() therefore opens with
 * an internal cross-GPU producer fence — it records every per-GPU stream's tail and fan-in-
 * waits — so the copies are sequenced behind every GPU's prior compute, and the same fence on
 * the next call covers the copy-vs-next-overwrite WAR. Callers driving @c Halo on a single
 * stream get correct multi-GPU results without any extra event juggling; the fence collapses
 * to a no-op on single-GPU / no-halo layouts.
 *
 * @tparam T          Element type of the field (matches @c Dense::Field<T, Components>).
 * @tparam Components Per-cell component count (matches @c Dense::Field<T, Components>).
 */
template <typename T, int Components>
class Halo {
public:
  /** Empty descriptor; @ref operator() is a no-op (zero entries). */
  Halo() = default;

  /**
   * Build a @c StencilSemantic::Standard halo-exchange descriptor for @p field — every
   * component plane of every adjacency pair is moved as its own
   * @c cudaMemcpyAsync entry per direction (the aggregated layout has no cross-component
   * fusion, since cardinality planes sit @c linear_storage_elements apart).
   *
   * @throws std::invalid_argument when the field has no GPU buffers and the layout requires
   *         inter-slab exchange, or when the field's @ref BoundaryMode is unrecognised.
   */
  explicit Halo(Dense::Field<T,
                             Components> field)
      : Halo(std::move(field),
             Stencil{},
             StencilSemantic::Standard) {}

  /**
   * Build a halo-exchange descriptor with explicit @p semantic .
   *
   * @c StencilSemantic::Standard ignores @p stencil and is equivalent to the no-stencil
   * constructor. @c StencilSemantic::Lattice interprets @p stencil 's points as the discrete
   * velocities @c c_q of the field's @c Components populations and only sends populations
   * whose @c c_q.z signs match the transfer direction. The aggregated layout has no
   * cross-cardinality fusion, so the lattice halo here is exactly @c positive_z populations
   * upward + @c negative_z populations downward per pair (see
   * @ref Stencil::lattice_profile ).
   *
   * @par Lattice pre-conditions (validated at construction)
   *
   * - @c stencil.size() must equal @c Components .
   * - @c stencil.radius() must be @c <= halo_z (the available halo thickness).
   * - @p semantic must be one of @c Standard / @c Lattice .
   *
   * @throws std::invalid_argument on any of the above failing.
   */
  Halo(Dense::Field<T,
                    Components> field,
       Stencil stencil,
       StencilSemantic semantic = StencilSemantic::Lattice);

  /** Field this descriptor is bound to. */
  [[nodiscard]] auto
  field() const noexcept -> const Dense::Field<T,
                                               Components>& {
    return field_;
  }

  /** Boundary mode driving the exchange (read off the field's @c Dense at construction). */
  [[nodiscard]] auto
  boundary_mode() const noexcept -> BoundaryMode {
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
   * Number of intra-process pair entries the exchange will issue. Equals
   * @c 2 * (gpu_count - 1) * Components for an Open-mode multi-local-GPU layout with
   * @c halo_z > 0, and @c 0 otherwise (single GPU, empty layout, no halo, or distributed
   * one-rank-per-GPU layout where every adjacent pair is cross-rank).
   */
  [[nodiscard]] auto
  pair_count() const noexcept -> size_t {
    return intra_.src_counts().size();
  }

  /**
   * Number of cross-rank @c Send / @c Recv NCCL group entries the exchange will issue. Equals
   * @c 2 * Components * peer_count (with @c peer_count in @c {0, 1, 2}) when MPI is enabled
   * and the field's @c System carries a @c Distributed handle; @c 0 in non-MPI builds, in
   * single-process layouts, or when @c halo_z == 0.
   */
  [[nodiscard]] auto
  dist_entry_count() const noexcept -> size_t {
#ifdef CARBON_HAVE_MPI_NCCL
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
   *   distributed @c Halo whose adjacency includes a non-local pair throws
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
          "Halo::launch_on_host: cross-rank host halo is not yet supported (this halo has {} "
          "cross-rank slab entries). The host MPI_Sendrecv fallback is a follow-up tracked in "
          "AGENTS.md (\"Halo on host\" memory bullet); for now use launch (device) on a halo "
          "whose pairs cross ranks, or restrict the layout to one rank.",
          host_dist_pending_);
    }
    if (host_intra_.empty()) {
      return;
    }
    auto& cpu = field_.cpu_buffers();
    if (cpu.size() == 0U) {
      log::throw_with_log<std::invalid_argument>(
          "Halo::launch_on_host: field has no host mirror (cpu_buffers().size() == 0); "
          "construct the field with MemorySide::Host or MemorySide::HostDevice");
    }
    for (const HostIntraEntry& e : host_intra_) {
      T* const dst_ptr = cpu[e.dst_slab].data() + e.dst_offset;
      const T* const src_ptr = cpu[e.src_slab].data() + e.src_offset;
      std::memcpy(static_cast<void*>(dst_ptr), static_cast<const void*>(src_ptr),
                  e.count * sizeof(T));
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
   * are computed once at @c Halo construction and reused at every host launch.
   */
  struct HostIntraEntry {
    size_t src_slab{};
    size_t dst_slab{};
    size_t src_offset{};
    size_t dst_offset{};
    size_t count{};
  };

  Dense::Field<T, Components> field_{};
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
// Halo<T, Components>::Halo — descriptor builder.
//
// Behaviour-preserving split of the original ~262 LOC ctor body into a thin
// orchestrator that delegates to phase helpers in `detail/halo_build.h` and
// keeps only the Dense-specific per-pair offset arithmetic inline. See the
// fourth-pass audit's P1.α resolution narrative for the design rationale and
// the symmetric `HaloDisg` ctor refactor.
// ---------------------------------------------------------------------------
template <typename T,
          int Components>
Halo<T,
     Components>::Halo(Dense::Field<T,
                                    Components> field,
                       Stencil stencil,
                       StencilSemantic semantic)
    : field_(std::move(field)),
      stencil_(std::move(stencil)),
      semantic_(semantic) {
  // ---------- Phase 1: validate semantic + boundary + lattice radius ----------
  detail::halo_build::validate_stencil_semantic(semantic_, "Halo");

  const Dense& dense = field_.dense();
  const auto& spans = dense.spans();
  const auto& gpu_buffers = field_.gpu_buffers();
  const size_t n_gpus = spans.size();

  // ---------- Phase 2: short-circuit on empty / no-halo layouts ----------
  if (n_gpus == 0U) {
    // Empty layout: no slabs to exchange. intra_/dist_ stay default-constructed.
    return;
  }

  const bool periodic_z =
      detail::halo_build::validate_boundary_mode_is_periodic_z(dense.boundary_mode(), "Halo");

  const int64_t halo_z = spans[0].halo_z();
  if (halo_z <= 0) {
    // Stencil radius 0: there is no halo region in storage to exchange.
    return;
  }

  if (semantic_ == StencilSemantic::Lattice) {
    if (stencil_.size() != static_cast<size_t>(Components)) {
      log::throw_with_log<std::invalid_argument>(
          "Halo(lattice): stencil.size() = {} but Components = {}; the lattice halo pairs "
          "population q with stencil.points()[q] and requires equal counts",
          stencil_.size(), Components);
    }
    if (stencil_.radius() > static_cast<int>(halo_z)) {
      log::throw_with_log<std::invalid_argument>(
          "Halo(lattice): stencil radius {} > halo_z {} (each lattice halo entry copies "
          "halo_z xy-planes; the field's halo is too thin for this stencil)",
          stencil_.radius(), halo_z);
    }
  }

  // Cardinality activity filter: the only direction-relevant axis on a z-partitioned grid
  // is z (mirroring Neon's lattice halo). Standard semantic keeps every cardinality.
  auto active_up = [&](int c) noexcept -> bool {
    if (semantic_ == StencilSemantic::Standard) {
      return true;
    }
    return stencil_.points()[static_cast<size_t>(c)].z > 0;
  };
  auto active_down = [&](int c) noexcept -> bool {
    if (semantic_ == StencilSemantic::Standard) {
      return true;
    }
    return stencil_.points()[static_cast<size_t>(c)].z < 0;
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

  // Carbon's halos are z-only, so all spans share x/y extents and pitch. Cache common values
  // off the local span (used for both intra and dist offset arithmetic).
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
          "Halo: cross-rank halo on a host-only field (MemorySide::Host) is not yet supported; "
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
    // distributed Dense should have been rejected earlier; reaching here means a misconfigured
    // layout.
    log::throw_with_log<std::invalid_argument>(
        "Halo: adjacency pair (g_lo={}, g_up={}) has a foreign endpoint, but Carbon was built "
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

/**
 * CTAD guide: @c Halo(field) deduces @c Halo<T, Components> from a @c Dense::Field<T, Components>.
 */
template <typename T,
          int Components>
Halo(Dense::Field<T,
                  Components>) -> Halo<T,
                                       Components>;

template <typename T,
          int Components>
Halo(Dense::Field<T,
                  Components>,
     Stencil,
     StencilSemantic) -> Halo<T,
                              Components>;

}  // namespace carbon
