#pragma once

#include <carbon/boundary_mode.h>
#include <carbon/dense_disg.h>
#include <carbon/detail/allocation.h>
#include <carbon/detail/buffer_group.h>
#include <carbon/detail/dense_disg/field.h>
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
 * Halo-exchange descriptor for a single @c DenseDisg::Field .
 *
 * @par What it does (vs. @c carbon::Halo )
 *
 * The aggregated dense layout stores cell @c (x, y, z, c) at address
 * @c c · linear_storage_elements + z · xy_pitch + y · X + x , so each cardinality plane
 * contributes its own @c cudaMemcpyAsync entry on the halo path. The disaggregated layout
 * groups all cardinalities of a region into a single contiguous block — the lower ghost
 * is @c [0, C · xy_pitch) , the down boundary is @c [C · xy_pitch, 2C · xy_pitch) , the
 * up boundary is @c [Lz · C · xy_pitch, (Lz + 1) · C · xy_pitch) , and the upper ghost is
 * @c [(Lz + 1) · C · xy_pitch, (Lz + 2) · C · xy_pitch) . That collapses the per-pair
 * halo update to two contiguous transfers per pair (regardless of @c Components ):
 *
 * - @b Upward: slab @c i 's up boundary block ( @c Lz_i · C · xy_pitch , size
 *   @c C · xy_pitch ) → slab @c i+1 's lower ghost block (offset @c 0 ).
 * - @b Downward: slab @c i+1 's down boundary block (offset @c C · xy_pitch ) →
 *   slab @c i 's upper ghost block (offset @c (Lz_i + 1) · C · xy_pitch ).
 *
 * Buffer-side layout matches between the device and host mirrors, so @ref launch_on_host
 * runs the same exchange against @c cpu_buffers via @c std::memcpy .
 *
 * @par Pair model and BoundaryMode
 *
 * Same global adjacency model as @c carbon::Halo : pairs @c (g_lo, g_up) for every
 * adjacent slab in @c [0, global_slab_count() - 1) , plus a wrap pair
 * @c (global_slab_count() - 1, 0) under @c BoundaryMode::PeriodicZ . On a single global
 * slab in PeriodicZ the wrap pair becomes a self-pair and the up boundary copies into
 * the same buffer's lower ghost, the down boundary into the upper ghost — non-overlapping
 * offsets, one buffer.
 *
 * @par Distributed scope
 *
 * Cross-rank pairs (one local endpoint, one peer-rank endpoint) are dispatched through an
 * internal @c DistTransfer<T> built on the field's @c gpu_buffers — same pattern as
 * @c carbon::Halo 's @c CARBON_HAVE_MPI_NCCL branch, adapted to the disg layout's region
 * offsets. Two component-run-fused transfers per pair, regardless of @c Components :
 *
 * - @b Lo-local: Send the up boundary at @c Lz · C · xy_pitch upward; Recv into the upper
 *   ghost at @c (Lz + 1) · C · xy_pitch . Lattice semantic narrows the Send to
 *   @c c_q.z > 0 runs and the Recv to @c c_q.z < 0 runs.
 * - @b Up-local: Send the down boundary at @c C · xy_pitch downward; Recv into the lower
 *   ghost at @c 0 . Lattice semantic narrows the Send to @c c_q.z < 0 runs and the Recv
 *   to @c c_q.z > 0 runs.
 *
 * Iteration order is @c runs_up first, then @c runs_down on every rank, so the per-peer
 * NCCL FIFO pairs Send ↔ Recv automatically. The host launch path still rejects pairs
 * with one foreign endpoint via @c host_dist_pending_ (the @c MPI_Sendrecv fallback is
 * the same follow-up as on @c carbon::Halo ).
 *
 * @tparam T          Element type of the field.
 * @tparam Components Per-cell component count (matches @c DenseDisg::Field<T, Components> ).
 */
template <typename T, int Components>
class HaloDisg {
public:
  HaloDisg() = default;

  /**
   * Build a @c StencilSemantic::Standard halo-exchange descriptor for @p field — every
   * cardinality plane of every adjacency pair's boundary block is moved as a single
   * contiguous transfer per direction (the disg layout's whole point).
   *
   * @throws std::invalid_argument when @p field has no GPU buffers and the layout
   *         requires intra-process transfers, or when the field's @c BoundaryMode is
   *         unrecognised.
   */
  explicit HaloDisg(DenseDisg::Field<T,
                                     Components> field)
      : HaloDisg(std::move(field),
                 Stencil{},
                 StencilSemantic::Standard) {}

  /**
   * Build a halo-exchange descriptor with explicit @p semantic .
   *
   * @c StencilSemantic::Standard ignores @p stencil and is equivalent to the no-stencil
   * constructor.
   *
   * @c StencilSemantic::Lattice interprets @p stencil 's points as the discrete velocities
   * @c c_q of the field's @c Components populations and only sends populations whose
   * @c c_q.z signs match the transfer direction (up / down). Adjacent active populations
   * are run-length fused into a single @c cudaMemcpyAsync — see
   * @ref Stencil::lattice_profile to inspect the resulting transfer count up-front.
   *
   * @par Lattice pre-conditions (validated at construction)
   *
   * - @c stencil.size() must equal @c Components (one stencil point per population).
   * - @c stencil.radius() must be @c <= 1 (the disg layout pins ghost / boundary radius
   *   to 1; lattices like Q39 / Q125 with radius @c >= 2 need extending the disg
   *   region count first).
   * - @p semantic must be one of @c Standard / @c Lattice .
   *
   * @throws std::invalid_argument on any of the above failing.
   */
  HaloDisg(DenseDisg::Field<T,
                            Components> field,
           Stencil stencil,
           StencilSemantic semantic = StencilSemantic::Lattice);

  [[nodiscard]] auto
  field() const noexcept -> const DenseDisg::Field<T,
                                                   Components>& {
    return field_;
  }

  [[nodiscard]] auto
  boundary_mode() const noexcept -> BoundaryMode {
    return field_.dense_disg().boundary_mode();
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
   * Total number of intra-process transfer entries the device launch will issue. For
   * @c StencilSemantic::Standard this is @c 2 * (number of contributing adjacency pairs)
   * (one upward + one downward per pair, regardless of @c Components ). For
   * @c StencilSemantic::Lattice it is @c 2 * pairs * (run_count_positive +
   * run_count_negative) (one entry per maximal run of consecutive same-z-sign
   * populations); identical to the standard count when the stencil ordering is
   * "fully groupable" — see @ref Stencil::lattice_profile .
   */
  [[nodiscard]] auto
  pair_count() const noexcept -> size_t {
    return intra_.src_counts().size();
  }

  /** Number of @c std::memcpy entries the host launch will issue (mirrors @ref pair_count ). */
  [[nodiscard]] auto
  host_intra_count() const noexcept -> size_t {
    return host_intra_.size();
  }

  /**
   * Number of cross-rank @c Send / @c Recv NCCL group entries the exchange will issue.
   * Equals @c 2 * (run_count_up + run_count_down) * peer_count (with @c peer_count in
   * @c {0, 1, 2} ) under @c CARBON_HAVE_MPI_NCCL ; @c 0 in non-MPI builds, in
   * single-process layouts, or when the disg has no halo region (@c halo_z == 0 ).
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
   * Issue the device halo exchange on @p stream . Dispatches the precomputed
   * intra-process @c Transfer<T> followed by the cross-rank @c DistTransfer<T>
   * (the latter only in MPI builds). Both are no-ops when their respective entry
   * counts are zero.
   */
  auto
  operator()(const Stream& stream) const -> void {
    // Cross-GPU producer fence — see the matching note on @c carbon::Halo::operator()
    // and @c StreamGroup::wait_for_all. The intra peer copies read each pair's
    // source-GPU owned cells written by prior compute on the source GPU's stream, but a
    // peer copy on the destination GPU's stream has no implicit ordering against it.
    // Empty fence_event_ (single-GPU / no-halo) skips the block.
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
   * Synchronous host halo exchange against @c field.cpu_buffers() . Waits on @p stream
   * first so any prior device kernel writing to the field has flushed to the CPU mirror
   * via a preceding @c Transfer .
   *
   * @throws std::invalid_argument when @p field has no host mirror.
   * @throws std::runtime_error when there is at least one cross-rank pair (the host MPI
   *         fallback is a follow-up; v1 only handles intra-process pairs on the host).
   */
  auto
  launch_on_host(const Stream& stream) const -> void {
    stream.wait();
    if (host_dist_pending_ > 0U) {
      log::throw_with_log<std::runtime_error>(
          "HaloDisg::launch_on_host: cross-rank host halo is not yet supported (this halo has {} "
          "cross-rank entries). Use the device launch path on a distributed layout.",
          host_dist_pending_);
    }
    if (host_intra_.empty()) {
      return;
    }
    auto& cpu = field_.cpu_buffers();
    if (cpu.size() == 0U) {
      log::throw_with_log<std::invalid_argument>(
          "HaloDisg::launch_on_host: field has no host mirror; build with MemorySide::Host or "
          "MemorySide::HostDevice");
    }
    for (const HostIntraEntry& e : host_intra_) {
      T* const dst_ptr = cpu[e.dst_slab].data() + e.dst_offset;
      const T* const src_ptr = cpu[e.src_slab].data() + e.src_offset;
      std::memcpy(static_cast<void*>(dst_ptr), static_cast<const void*>(src_ptr),
                  e.count * sizeof(T));
    }
  }

private:
  struct HostIntraEntry {
    size_t src_slab{};
    size_t dst_slab{};
    size_t src_offset{};
    size_t dst_offset{};
    size_t count{};
  };

  DenseDisg::Field<T, Components> field_{};
  Transfer<T> intra_{};
#ifdef CARBON_HAVE_MPI_NCCL
  DistTransfer<T> dist_{};
#endif
  std::vector<HostIntraEntry> host_intra_{};
  Stencil stencil_{};
  StencilSemantic semantic_{StencilSemantic::Standard};
  // Number of cross-rank pairs the host pipeline does not yet handle — see launch_on_host.
  size_t host_dist_pending_{};
  // Per-GPU event group backing the cross-GPU producer fence in @ref operator() (see the
  // matching member on @c carbon::Halo). Allocated only for multi-local-GPU layouts.
  mutable Event fence_event_{};
};

namespace detail::halo_disg {

/**
 * Run-length descriptor for the lattice halo: a contiguous range of population
 * indices @c [j_start, j_start + j_len) whose @c c_q.z all share the transfer
 * direction's sign. Returned by @ref runs_for_direction below.
 */
struct CardinalityRun {
  int j_start{};
  int j_len{};
};

enum class TransferDir : int {
  Up = 0,
  Down = 1
};

/**
 * Compute the run-length-fused population windows that contribute to a single
 * direction of a lattice halo. Standard semantic always returns the single run
 * @c [0, Components) ; lattice semantic walks @c stencil.points() and emits one
 * @c CardinalityRun per maximal run of consecutive same-sign-z populations.
 */
template <int Components>
[[nodiscard]] inline auto
runs_for_direction(StencilSemantic semantic,
                   const Stencil& stencil,
                   TransferDir dir) -> std::vector<CardinalityRun> {
  std::vector<CardinalityRun> runs;
  if (semantic == StencilSemantic::Standard) {
    runs.push_back({0, Components});
    return runs;
  }
  // Lattice: walk cardinalities and emit one run per maximal active stretch.
  int j_start = -1;
  int j_len = 0;
  auto flush = [&]() {
    if (j_len > 0) {
      runs.push_back({j_start, j_len});
      j_len = 0;
    }
  };
  for (int j = 0; j < Components; ++j) {
    const auto& s = stencil.points()[static_cast<size_t>(j)];
    const bool active = (dir == TransferDir::Up) ? (s.z > 0) : (s.z < 0);
    if (active) {
      if (j_len == 0) {
        j_start = j;
      }
      ++j_len;
    } else {
      flush();
    }
  }
  flush();
  return runs;
}

}  // namespace detail::halo_disg

// ---------------------------------------------------------------------------
// HaloDisg<T, Components>::HaloDisg — descriptor builder.
//
// Behaviour-preserving split of the original ~230 LOC ctor body into a thin
// orchestrator that delegates to phase helpers in `detail/halo_build.h` and
// keeps only the disg-specific per-pair offset arithmetic + run-fused
// directional iteration inline. See the fourth-pass audit's P1.α resolution
// narrative for the design rationale and the symmetric `Halo` ctor refactor.
// ---------------------------------------------------------------------------
template <typename T,
          int Components>
HaloDisg<T,
         Components>::HaloDisg(DenseDisg::Field<T,
                                                Components> field,
                               Stencil stencil,
                               StencilSemantic semantic)
    : field_(std::move(field)),
      stencil_(std::move(stencil)),
      semantic_(semantic) {
  // ---------- Phase 1: validate semantic + lattice radius (boundary mode comes after geometry) ----------
  detail::halo_build::validate_stencil_semantic(semantic_, "HaloDisg");
  if (semantic_ == StencilSemantic::Lattice) {
    if (stencil_.size() != static_cast<size_t>(Components)) {
      log::throw_with_log<std::invalid_argument>(
          "HaloDisg(lattice): stencil.size() = {} but Components = {}; the lattice halo pairs "
          "population q with stencil.points()[q] and requires equal counts",
          stencil_.size(), Components);
    }
    if (stencil_.radius() > 1) {
      log::throw_with_log<std::invalid_argument>(
          "HaloDisg(lattice): stencil radius {} > 1 (the disg layout pins ghost / boundary "
          "radius to 1; multi-radius lattices like Q39 / Q125 need extending the disg "
          "region count first)",
          stencil_.radius());
    }
  }

  const DenseDisg& dense = field_.dense_disg();
  const auto& spans = dense.spans();
  const auto& gpu_buffers = field_.gpu_buffers();
  const size_t n_gpus = spans.size();

  // ---------- Phase 2: short-circuit on empty / no-halo layouts ----------
  if (n_gpus == 0U) {
    return;
  }

  const bool periodic_z =
      detail::halo_build::validate_boundary_mode_is_periodic_z(dense.boundary_mode(), "HaloDisg");

  const int64_t halo_z = spans[0].halo_z();
  if (halo_z <= 0) {
    return;
  }

  const bool has_gpu_buffers = (gpu_buffers.size() == n_gpus);

  // ---------- Phase 3: build adjacency list ----------
  const size_t global_n = dense.global_slab_count();
  const size_t local_offset = dense.local_slab_offset();
  const auto adjacency = detail::halo_build::adjacency_pairs(global_n, periodic_z);

  // Disg layout has no XY pitch padding (xy_slab_elements == X * Y), and all spans share
  // x/y extents and pitch.
  const int64_t pitch = spans[0].xy_slab_elements();
  const size_t pitch_sz = static_cast<size_t>(pitch);
  const auto runs_up = detail::halo_disg::runs_for_direction<Components>(
      semantic_, stencil_, detail::halo_disg::TransferDir::Up);
  const auto runs_down = detail::halo_disg::runs_for_direction<Components>(
      semantic_, stencil_, detail::halo_disg::TransferDir::Down);

  auto is_local = [&](size_t g) noexcept -> bool {
    return g >= local_offset && g < local_offset + n_gpus;
  };

  // ---------- Phase 4: walk adjacency, classify pairs, accumulate entries ----------
  //
  // The disg layout's whole point is that one contiguous block per direction covers every
  // component plane: each pair contributes one entry per active run (one run total under
  // Standard semantic; one or more under Lattice depending on the stencil's z-sign profile).
  // Cross-rank pairs iterate runs_up first then runs_down on every rank — the per-peer
  // NCCL FIFO matches Sends ↔ Recvs by call order alone.
  detail::halo_build::IntraTransferBuilder<T> intra;
#ifdef CARBON_HAVE_MPI_NCCL
  detail::halo_build::DistTransferBuilder<T> dist;
#endif

  for (const auto& [g_lo, g_up] : adjacency) {
    const bool lo_local = is_local(g_lo);
    const bool up_local = is_local(g_up);
    if (!lo_local && !up_local) {
      continue;  // Foreign pair; another rank handles it.
    }

    if (!lo_local || !up_local) {
      // Cross-rank pair: dispatch through DistTransfer below (under MPI) or throw if the
      // build has no MPI. host_dist_pending_ tracks pair count so the *host* launch path
      // can still throw a clean diagnostic — the host MPI_Sendrecv fallback is the same
      // tracked follow-up as on carbon::Halo.
      ++host_dist_pending_;
#ifdef CARBON_HAVE_MPI_NCCL
      if (!has_gpu_buffers) {
        log::throw_with_log<std::invalid_argument>(
            "HaloDisg: cross-rank halo on a host-only field (MemorySide::Host) is not yet "
            "supported; use MemorySide::HostDevice (or single-rank layout)");
      }
      const size_t local_g = lo_local ? g_lo : g_up;
      const size_t peer_g = lo_local ? g_up : g_lo;
      const int peer = static_cast<int>(peer_g);
      const size_t i_local = local_g - local_offset;
      const auto& span_local = spans[i_local];
      const int64_t lz_local = span_local.logic_extent().z;

      // Lo-local: Send up-boundary (Lz·C·pitch) upward, Recv into upper-ghost ((Lz+1)·C·pitch).
      // Up-local: Send down-boundary (C·pitch) downward, Recv into lower-ghost (0).
      const size_t send_region_start =
          lo_local ? static_cast<size_t>(lz_local) * static_cast<size_t>(Components) * pitch_sz
                   : static_cast<size_t>(Components) * pitch_sz;
      const size_t recv_region_start = lo_local ? (static_cast<size_t>(lz_local) + 1U) *
                                                      static_cast<size_t>(Components) * pitch_sz
                                                : 0U;

      // First block: runs_up — lo_local Sends, up_local Recvs.
      {
        const size_t region_start = lo_local ? send_region_start : recv_region_start;
        const DistOp op = lo_local ? DistOp::Send : DistOp::Recv;
        for (const auto& run : runs_up) {
          const size_t off = static_cast<size_t>(run.j_start) * pitch_sz;
          const size_t cnt = static_cast<size_t>(run.j_len) * pitch_sz;
          dist.push(peer, op, region_start + off, cnt);
        }
      }
      // Second block: runs_down — lo_local Recvs, up_local Sends.
      {
        const size_t region_start = lo_local ? recv_region_start : send_region_start;
        const DistOp op = lo_local ? DistOp::Recv : DistOp::Send;
        for (const auto& run : runs_down) {
          const size_t off = static_cast<size_t>(run.j_start) * pitch_sz;
          const size_t cnt = static_cast<size_t>(run.j_len) * pitch_sz;
          dist.push(peer, op, region_start + off, cnt);
        }
      }
      continue;
#else
      log::throw_with_log<std::invalid_argument>(
          "HaloDisg: adjacency pair (g_lo={}, g_up={}) has a foreign endpoint, but Carbon was "
          "built without CARBON_ENABLE_MPI=ON; build with MPI to enable cross-rank halo "
          "exchange",
          g_lo, g_up);
#endif
    }

    // Both endpoints local: emit one intra entry per run per direction.
    const size_t i_lo = g_lo - local_offset;
    const size_t i_up = g_up - local_offset;
    const auto& span_lo = spans[i_lo];
    const int64_t lz_lo = span_lo.logic_extent().z;

    // Upward: slab i_lo's up boundary block (region start offset Lz_lo · C · xy_pitch) →
    // slab i_up's lower ghost block (region start offset 0). One transfer per run.
    const size_t up_src_region_start =
        static_cast<size_t>(lz_lo) * static_cast<size_t>(Components) * pitch_sz;
    const size_t up_dst_region_start = 0U;
    for (const auto& run : runs_up) {
      const size_t off = static_cast<size_t>(run.j_start) * pitch_sz;
      const size_t cnt = static_cast<size_t>(run.j_len) * pitch_sz;
      const size_t src_off = up_src_region_start + off;
      const size_t dst_off = up_dst_region_start + off;
      if (has_gpu_buffers) {
        intra.push(gpu_buffers[i_lo], gpu_buffers[i_up], src_off, dst_off, cnt);
      }
      host_intra_.push_back({i_lo, i_up, src_off, dst_off, cnt});
    }

    // Downward: slab i_up's down boundary block (offset C · xy_pitch) →
    // slab i_lo's upper ghost block (offset (Lz_lo + 1) · C · xy_pitch). One transfer per run.
    const size_t dn_src_region_start = static_cast<size_t>(Components) * pitch_sz;
    const size_t dn_dst_region_start =
        (static_cast<size_t>(lz_lo) + 1U) * static_cast<size_t>(Components) * pitch_sz;
    for (const auto& run : runs_down) {
      const size_t off = static_cast<size_t>(run.j_start) * pitch_sz;
      const size_t cnt = static_cast<size_t>(run.j_len) * pitch_sz;
      const size_t src_off = dn_src_region_start + off;
      const size_t dst_off = dn_dst_region_start + off;
      if (has_gpu_buffers) {
        intra.push(gpu_buffers[i_up], gpu_buffers[i_lo], src_off, dst_off, cnt);
      }
      host_intra_.push_back({i_up, i_lo, src_off, dst_off, cnt});
    }
  }

  // ---------- Phase 5: finalise the accumulated entries into Transfer / DistTransfer ----------
  intra_ = intra.finalise(dense.system());
#ifdef CARBON_HAVE_MPI_NCCL
  dist_ = dist.finalise(dense.system(), gpu_buffers);
#endif

  // ---------- Phase 6: allocate the cross-GPU producer fence (multi-local-GPU only) ----------
  if (intra_.src_counts().size() > 0U) {
    System sys = dense.system();
    fence_event_ = sys.event();
  }
}

template <typename T,
          int Components>
HaloDisg(DenseDisg::Field<T,
                          Components>) -> HaloDisg<T,
                                                   Components>;

template <typename T,
          int Components>
HaloDisg(DenseDisg::Field<T,
                          Components>,
         Stencil,
         StencilSemantic) -> HaloDisg<T,
                                      Components>;

}  // namespace carbon
