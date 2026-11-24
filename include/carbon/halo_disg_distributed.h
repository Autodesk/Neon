#pragma once

#include <carbon/boundary_mode.h>
#include <carbon/boundary_mode_distributed.h>
#include <carbon/dense_disg_distributed.h>
#include <carbon/detail/allocation.h>
#include <carbon/detail/buffer_group.h>
#include <carbon/detail/dense_disg_distributed/field.h>
#include <carbon/detail/dense_disg_distributed/halo_plan.h>
#include <carbon/detail/dense_disg_distributed/layout.h>
#include <carbon/detail/dense_disg_distributed/row_copy.h>
#include <carbon/detail/halo_build.h>
#include <carbon/halo_disg.h>
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
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace carbon {

/**
 * Halo-exchange descriptor for a single @c DenseDisgDistributed::Field . Twin of
 * @ref HaloDisg over the @ref ProcessGrid -partitioned slab, pencil and brick layouts.
 *
 * @par Slab layouts: zero-copy z faces
 *
 * Every z face of a disaggregated slab is one contiguous region block holding all
 * cardinalities ( @c C · xy_pitch elements, see @c dense_disg_distributed::Span::Region ), so
 * each adjacency pair needs no pack / unpack kernel: the exchange is a direct copy between the
 * region blocks of the two slab buffers.
 *
 * - @b Upward: slab @c i 's @c UpBoundary block → slab @c i+1 's @c LowerGhost block.
 * - @b Downward: slab @c i+1 's @c DownBoundary block → slab @c i 's @c UpperGhost block.
 *
 * @par Pair model and boundary mode
 *
 * Pairs @c (g_lo, g_up) cover every adjacent process-grid z coordinate, plus the wrap pair
 * @c (Pz - 1, 0) under @ref BoundaryModeDistributed::PeriodicZ .
 *
 * @par Distributed scope
 *
 * Pairs with both endpoints on this process go through an intra-process @c Transfer<T> (device)
 * and a @c std::memcpy plan (host). Pairs with one foreign endpoint go through a
 * @c DistTransfer<T> under @c CARBON_HAVE_MPI_NCCL ; the peer rank is
 * @c process_grid().rank_of({0, 0, peer_z}) . Entries iterate the up runs first, then the down
 * runs on every rank, so the per-peer NCCL FIFO pairs Send ↔ Recv by call order.
 *
 * @par Lattice fusion
 *
 * Under @c StencilSemantic::Lattice each transfer is narrowed to the populations whose
 * @c c_q.z sign matches the direction, run-length fused exactly as in @ref HaloDisg
 * ( @c detail::halo_disg::runs_for_direction ).
 *
 * @par Pencil layouts: direct face + edge peers
 *
 * A pencil chunk (storage with y ghosts, see @c dense_disg_distributed::Span ) exchanges with up
 * to four face peers ( @c ±y , @c ±z ) and four yz-edge peers in @b one NCCL group — no staged
 * axis passes and no gather / scatter of faces. Every message is one contiguous range of both
 * chunk buffers: z faces are the @c Down / @c UpBoundary blocks (zero-copy, maximal lattice runs
 * as on slabs), y faces the @c YDown / @c YUpBoundary blocks and edges the @c EdgeSend* strips,
 * each widened to one covering component run. Before the exchange one row-copy launch per chunk
 * refreshes the y-face shadow rows and edge strips from the canonical z-boundary rows (at most
 * eight strided row blocks). The direction table and component selection live in
 * @c detail/dense_disg_distributed/halo_plan.h .
 *
 * @par Brick layouts: direct face + edge + corner peers
 *
 * A brick chunk ( @c ghost_x() == 1 ) exchanges with up to six face, twelve edge and eight corner
 * peers in the same single NCCL group (D3Q19 lattice semantics select no corner population, so
 * an LBM q19 run posts at most 18 peers). Every message is @b one contiguous range: the selected
 * components are widened to one covering run per direction (z faces included), and the send block
 * is the z-boundary plane, the y / x face group (shadow rows / columns refreshed first) or an
 * edge / corner strip. One row-copy launch per chunk performs every refresh (at most 40 strided
 * jobs, see @c dense_disg_distributed::append_brick_send_refresh ).
 *
 * Messages to / from the same peer are posted in the canonical direction order (Send to
 * @c coords + d , then Recv from @c coords - d , per direction), so the per-peer NCCL FIFOs pair
 * up on every rank. Peers on this process go through the intra-process @c Transfer<T> (device)
 * and @c std::memcpy plan (host) instead.
 *
 * @tparam T          Element type of the field.
 * @tparam Components Per-cell component count.
 */
template <typename T, int Components>
class HaloDisgDistributed {
public:
  HaloDisgDistributed() = default;

  /** Build a @c StencilSemantic::Standard halo for @p field (one transfer per pair direction). */
  explicit HaloDisgDistributed(DenseDisgDistributed::Field<T,
                                                           Components> field)
      : HaloDisgDistributed(std::move(field),
                            Stencil{},
                            StencilSemantic::Standard) {}

  /**
   * Build a halo-exchange descriptor with explicit @p semantic . Same lattice pre-conditions
   * as @ref HaloDisg : @c stencil.size() == Components and @c stencil.radius() <= 1 .
   *
   * @throws std::invalid_argument on an invalid semantic / lattice stencil, or on a cross-rank
   *         pair in a build without MPI.
   */
  HaloDisgDistributed(DenseDisgDistributed::Field<T,
                                                  Components> field,
                      Stencil stencil,
                      StencilSemantic semantic = StencilSemantic::Lattice);

  [[nodiscard]] auto
  field() const noexcept -> const DenseDisgDistributed::Field<T,
                                                              Components>& {
    return field_;
  }

  [[nodiscard]] auto
  boundary_mode() const noexcept -> BoundaryModeDistributed {
    return field_.dense_disg_distributed().boundary_mode();
  }

  [[nodiscard]] auto
  stencil_semantic() const noexcept -> StencilSemantic {
    return semantic_;
  }

  [[nodiscard]] auto
  stencil() const noexcept -> const Stencil& {
    return stencil_;
  }

  /** Number of intra-process device transfer entries (one per run per pair direction). */
  [[nodiscard]] auto
  pair_count() const noexcept -> size_t {
    return intra_.src_counts().size();
  }

  /** Number of @c std::memcpy entries the host launch will issue (mirrors @ref pair_count ). */
  [[nodiscard]] auto
  host_intra_count() const noexcept -> size_t {
    return host_intra_.size();
  }

  /** Number of shadow / strip row-copy jobs the device and host launches run before the copies. */
  [[nodiscard]] auto
  shadow_job_count() const noexcept -> size_t {
    size_t n = 0;
    for (const auto& a : shadow_) {
      n += static_cast<size_t>(a.job_count);
    }
    return n;
  }

  /** Number of cross-rank @c Send / @c Recv entries; @c 0 without @c CARBON_HAVE_MPI_NCCL . */
  [[nodiscard]] auto
  dist_entry_count() const noexcept -> size_t {
#ifdef CARBON_HAVE_MPI_NCCL
    return dist_.entry_count();
#else
    return 0U;
#endif
  }

  /** Issue the device halo exchange on @p stream (intra-process, then cross-rank). */
  auto
  operator()(const Stream& stream) const -> void {
    refresh_shadows_(stream);
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
   * Synchronous host halo exchange against @c field.cpu_buffers() . Waits on @p stream first.
   *
   * @throws std::invalid_argument when @p field has no host mirror.
   * @throws std::runtime_error when the halo has a cross-rank pair (host MPI fallback is not
   *         implemented, same as @ref HaloDisg ).
   */
  auto
  launch_on_host(const Stream& stream) const -> void {
    stream.wait();
    if (host_dist_pending_ > 0U) {
      log::throw_with_log<std::runtime_error>(
          "HaloDisgDistributed::launch_on_host: cross-rank host halo is not yet supported (this "
          "halo has {} cross-rank entries). Use the device launch path on a distributed layout.",
          host_dist_pending_);
    }
    if (host_intra_.empty()) {
      return;
    }
    auto& cpu = field_.cpu_buffers();
    if (cpu.size() == 0U) {
      log::throw_with_log<std::invalid_argument>(
          "HaloDisgDistributed::launch_on_host: field has no host mirror; build with "
          "MemorySide::Host or MemorySide::HostDevice");
    }
    for (size_t i = 0; i < shadow_.size(); ++i) {
      detail::dense_disg_distributed::row_copy_host(cpu[i].data(), shadow_[i]);
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

  auto
  build_slab_() -> void;

  auto
  build_pencil_() -> void;

  auto
  build_brick_() -> void;

  auto
  finalise_shadows_() -> void;

  auto
  refresh_shadows_(const Stream& stream) const -> void {
    if (shadow_.empty()) {
      return;
    }
#if CARBON_BACKEND_IS_CUDALIKE
    if (detail::dense_disg_distributed::row_copy_element_size_supported(sizeof(T))) {
      const auto& devices = field_.dense_disg_distributed().system().gpu_group().devices();
      for (size_t i = 0; i < shadow_.size(); ++i) {
        if (shadow_[i].empty()) {
          continue;
        }
        const detail::gpu_error_t err = detail::dense_disg_distributed::launch_row_copy(
            devices[i], stream.native_at(i), sizeof(T), field_.gpu_buffers()[i].data(), shadow_[i]);
        if (err != detail::gpu_success) {
          log::throw_with_log<std::runtime_error>(
              "HaloDisgDistributed: shadow row-copy launch failed: {}",
              detail::gpu_error_string(err));
        }
      }
      return;
    }
#endif
    shadow_rows_(stream);
  }

  DenseDisgDistributed::Field<T, Components> field_{};
  Transfer<T> intra_{};
#ifdef CARBON_HAVE_MPI_NCCL
  DistTransfer<T> dist_{};
#endif
  std::vector<detail::dense_disg_distributed::RowCopyArgs> shadow_{};
  Transfer<T> shadow_rows_{};
  std::vector<HostIntraEntry> host_intra_{};
  Stencil stencil_{};
  StencilSemantic semantic_{StencilSemantic::Standard};
  size_t host_dist_pending_{};
  mutable Event fence_event_{};
};

template <typename T,
          int Components>
HaloDisgDistributed<T,
                    Components>::HaloDisgDistributed(DenseDisgDistributed::Field<T,
                                                                                 Components> field,
                                                     Stencil stencil,
                                                     StencilSemantic semantic)
    : field_(std::move(field)),
      stencil_(std::move(stencil)),
      semantic_(semantic) {
  detail::halo_build::validate_stencil_semantic(semantic_, "HaloDisgDistributed");
  if (is_lattice_semantic(semantic_)) {
    if (stencil_.size() != static_cast<size_t>(Components)) {
      log::throw_with_log<std::invalid_argument>(
          "HaloDisgDistributed(lattice): stencil.size() = {} but Components = {}; the lattice "
          "halo pairs population q with stencil.points()[q] and requires equal counts",
          stencil_.size(), Components);
    }
    if (stencil_.radius() > 1) {
      log::throw_with_log<std::invalid_argument>(
          "HaloDisgDistributed(lattice): stencil radius {} > 1 (the disaggregated layout pins "
          "ghost / boundary radius to 1)",
          stencil_.radius());
    }
  }

  const auto& spans = field_.dense_disg_distributed().spans();
  if (spans.empty()) {
    return;
  }
  if (spans[0].ghost_x() != 0) {
    build_brick_();
  } else if (spans[0].ghost_y() != 0) {
    build_pencil_();
  } else {
    build_slab_();
  }
}

template <typename T,
          int Components>
auto
HaloDisgDistributed<T,
                    Components>::build_slab_() -> void {
  using Region = detail::dense_disg_distributed::Region;

  const DenseDisgDistributed& grid = field_.dense_disg_distributed();
  const auto& spans = grid.spans();
  const auto& gpu_buffers = field_.gpu_buffers();
  const size_t n_gpus = spans.size();

  const bool periodic_z = detail::dense_disg_distributed::to_boundary_mode(grid.boundary_mode()) ==
                          BoundaryMode::PeriodicZ;

  if (spans[0].halo_z() <= 0) {
    return;
  }

  const bool has_gpu_buffers = (gpu_buffers.size() == n_gpus);

  const size_t global_n = grid.global_slab_count();
  const size_t local_offset = grid.local_slab_offset();
  const auto adjacency = detail::halo_build::adjacency_pairs(global_n, periodic_z);

  const size_t pitch_sz = static_cast<size_t>(spans[0].xy_slab_elements());
  const auto runs_up = detail::halo_disg::runs_for_direction<Components>(
      semantic_, stencil_, detail::halo_disg::TransferDir::Up);
  const auto runs_down = detail::halo_disg::runs_for_direction<Components>(
      semantic_, stencil_, detail::halo_disg::TransferDir::Down);

  auto is_local = [&](size_t g) noexcept -> bool {
    return g >= local_offset && g < local_offset + n_gpus;
  };
  auto region_start = [&](size_t i, Region r) -> size_t {
    return static_cast<size_t>(spans[i].region_storage_offset(r, Components));
  };

  detail::halo_build::IntraTransferBuilder<T> intra;
#ifdef CARBON_HAVE_MPI_NCCL
  detail::halo_build::DistTransferBuilder<T> dist;
#endif

  for (const auto& [g_lo, g_up] : adjacency) {
    const bool lo_local = is_local(g_lo);
    const bool up_local = is_local(g_up);
    if (!lo_local && !up_local) {
      continue;
    }

    if (!lo_local || !up_local) {
      ++host_dist_pending_;
#ifdef CARBON_HAVE_MPI_NCCL
      if (!has_gpu_buffers) {
        log::throw_with_log<std::invalid_argument>(
            "HaloDisgDistributed: cross-rank halo on a host-only field (MemorySide::Host) is not "
            "yet supported; use MemorySide::HostDevice (or a single-rank layout)");
      }
      const size_t local_g = lo_local ? g_lo : g_up;
      const size_t peer_g = lo_local ? g_up : g_lo;
      const int peer = grid.process_grid().rank_of(int32_3d{0, 0, static_cast<int32_t>(peer_g)});
      const size_t i_local = local_g - local_offset;

      const size_t send_region_start = lo_local ? region_start(i_local, Region::UpBoundary)
                                                : region_start(i_local, Region::DownBoundary);
      const size_t recv_region_start = lo_local ? region_start(i_local, Region::UpperGhost)
                                                : region_start(i_local, Region::LowerGhost);

      {
        const size_t start = lo_local ? send_region_start : recv_region_start;
        const DistOp op = lo_local ? DistOp::Send : DistOp::Recv;
        for (const auto& run : runs_up) {
          const size_t off = static_cast<size_t>(run.j_start) * pitch_sz;
          const size_t cnt = static_cast<size_t>(run.j_len) * pitch_sz;
          dist.push(peer, op, start + off, cnt);
        }
      }
      {
        const size_t start = lo_local ? recv_region_start : send_region_start;
        const DistOp op = lo_local ? DistOp::Recv : DistOp::Send;
        for (const auto& run : runs_down) {
          const size_t off = static_cast<size_t>(run.j_start) * pitch_sz;
          const size_t cnt = static_cast<size_t>(run.j_len) * pitch_sz;
          dist.push(peer, op, start + off, cnt);
        }
      }
      continue;
#else
      log::throw_with_log<std::invalid_argument>(
          "HaloDisgDistributed: adjacency pair (g_lo={}, g_up={}) has a foreign endpoint, but "
          "Carbon was built without CARBON_ENABLE_MPI=ON; build with MPI to enable cross-rank "
          "halo exchange",
          g_lo, g_up);
#endif
    }

    const size_t i_lo = g_lo - local_offset;
    const size_t i_up = g_up - local_offset;

    const size_t up_src_region_start = region_start(i_lo, Region::UpBoundary);
    const size_t up_dst_region_start = region_start(i_up, Region::LowerGhost);
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

    const size_t dn_src_region_start = region_start(i_up, Region::DownBoundary);
    const size_t dn_dst_region_start = region_start(i_lo, Region::UpperGhost);
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

  intra_ = intra.finalise(grid.system());
#ifdef CARBON_HAVE_MPI_NCCL
  dist_ = dist.finalise(grid.system(), gpu_buffers);
#endif

  if (intra_.src_counts().size() > 0U) {
    System sys = grid.system();
    fence_event_ = sys.event();
  }
}

template <typename T,
          int Components>
auto
HaloDisgDistributed<T,
                    Components>::build_pencil_() -> void {
  namespace ddd = detail::dense_disg_distributed;

  const DenseDisgDistributed& grid = field_.dense_disg_distributed();
  const auto& spans = grid.spans();
  const auto& gpu_buffers = field_.gpu_buffers();
  const size_t n_gpus = spans.size();
  const bool has_gpu_buffers = (gpu_buffers.size() == n_gpus);
  const ProcessGrid& pg = grid.process_grid();
  const BoundaryModeDistributed mode = grid.boundary_mode();
  const int32_t first_rank = grid.local_rank_offset();

  if (spans[0].halo_z() <= 0) {
    return;
  }

  auto local_index = [&](int32_t rank) noexcept -> std::optional<size_t> {
    const int64_t i = static_cast<int64_t>(rank) - first_rank;
    if (i >= 0 && i < static_cast<int64_t>(n_gpus)) {
      return static_cast<size_t>(i);
    }
    return std::nullopt;
  };

  detail::halo_build::IntraTransferBuilder<T> intra;
#ifdef CARBON_HAVE_MPI_NCCL
  detail::halo_build::DistTransferBuilder<T> dist;
#endif
  shadow_.assign(n_gpus, ddd::RowCopyArgs{});

  for (size_t i = 0; i < n_gpus; ++i) {
    const ddd::Span& s = spans[i];
    const int32_3d coords = pg.coords_of(first_rank + static_cast<int32_t>(i));
    for (const ddd::PencilDir d : ddd::kPencilDirs) {
      if ((d.dy != 0 && s.halo_y() <= 0) || (d.dz != 0 && s.halo_z() <= 0)) {
        continue;
      }
      const auto runs = ddd::pencil_runs<Components>(semantic_, stencil_, d);
      if (runs.empty()) {
        continue;
      }
      const ddd::Region send = ddd::send_region(d);
      const ddd::Region recv = ddd::recv_region(d);

      if (const auto to = ddd::pencil_peer(pg, coords, d, mode); to.has_value()) {
        const int32_t peer_rank = pg.rank_of(*to);
        for (const auto& run : runs) {
          ddd::append_send_refresh(s, Components, d, run, shadow_[i]);
        }
        if (const auto j = local_index(peer_rank); j.has_value()) {
          for (const auto& run : runs) {
            const ddd::PencilMessage src = ddd::region_run(s, send, Components, run);
            const ddd::PencilMessage dst = ddd::region_run(spans[*j], recv, Components, run);
            if (src.count != dst.count) {
              log::throw_with_log<std::logic_error>(
                  "HaloDisgDistributed: pencil message size mismatch between local chunks {} and "
                  "{} (direction ({}, {}), {} vs {} elements)",
                  i, *j, d.dy, d.dz, src.count, dst.count);
            }
            if (has_gpu_buffers) {
              intra.push(gpu_buffers[i], gpu_buffers[*j], src.first, dst.first, src.count);
            }
            host_intra_.push_back({i, *j, src.first, dst.first, src.count});
          }
        } else {
          ++host_dist_pending_;
#ifdef CARBON_HAVE_MPI_NCCL
          if (!has_gpu_buffers) {
            log::throw_with_log<std::invalid_argument>(
                "HaloDisgDistributed: cross-rank halo on a host-only field (MemorySide::Host) is "
                "not yet supported; use MemorySide::HostDevice (or a single-rank layout)");
          }
          for (const auto& run : runs) {
            const ddd::PencilMessage src = ddd::region_run(s, send, Components, run);
            dist.push(peer_rank, DistOp::Send, src.first, src.count);
          }
#else
          log::throw_with_log<std::invalid_argument>(
              "HaloDisgDistributed: pencil peer rank {} is on another process, but Carbon was "
              "built without CARBON_ENABLE_MPI=ON; build with MPI to enable cross-rank halo "
              "exchange",
              peer_rank);
#endif
        }
      }

      if (const auto from = ddd::pencil_peer(pg, coords, ddd::PencilDir{-d.dy, -d.dz}, mode);
          from.has_value()) {
        const int32_t peer_rank = pg.rank_of(*from);
        if (!local_index(peer_rank).has_value()) {
          ++host_dist_pending_;
#ifdef CARBON_HAVE_MPI_NCCL
          for (const auto& run : runs) {
            const ddd::PencilMessage dst = ddd::region_run(s, recv, Components, run);
            dist.push(peer_rank, DistOp::Recv, dst.first, dst.count);
          }
#endif
        }
      }
    }
  }

  finalise_shadows_();

  intra_ = intra.finalise(grid.system());
#ifdef CARBON_HAVE_MPI_NCCL
  dist_ = dist.finalise(grid.system(), gpu_buffers);
#endif

  if (intra_.src_counts().size() > 0U) {
    System sys = grid.system();
    fence_event_ = sys.event();
  }
}

template <typename T,
          int Components>
auto
HaloDisgDistributed<T,
                    Components>::finalise_shadows_() -> void {
  namespace ddd = detail::dense_disg_distributed;

  const DenseDisgDistributed& grid = field_.dense_disg_distributed();
  const auto& gpu_buffers = field_.gpu_buffers();
  bool any_shadow = false;
  for (const auto& a : shadow_) {
    any_shadow = any_shadow || !a.empty();
  }
  if (!any_shadow) {
    shadow_.clear();
    return;
  }
  if (gpu_buffers.size() != shadow_.size()) {
    return;
  }
  bool kernel_path = false;
#if CARBON_BACKEND_IS_CUDALIKE
  kernel_path = ddd::row_copy_element_size_supported(sizeof(T));
#endif
  if (kernel_path) {
    return;
  }
  detail::halo_build::IntraTransferBuilder<T> rows;
  for (size_t i = 0; i < shadow_.size(); ++i) {
    const ddd::RowCopyArgs& a = shadow_[i];
    for (int32_t k = 0; k < a.job_count; ++k) {
      const ddd::RowCopyJob& job = a.jobs[k];
      const bool contiguous = job.src_step == 1 && job.dst_step == 1;
      for (int64_t r = 0; r < job.rows; ++r) {
        const int64_t src = job.src_first + r * job.src_stride;
        const int64_t dst = job.dst_first + r * job.dst_stride;
        if (contiguous) {
          rows.push(gpu_buffers[i], gpu_buffers[i], static_cast<size_t>(src),
                    static_cast<size_t>(dst), static_cast<size_t>(job.len));
          continue;
        }
        for (int64_t x = 0; x < job.len; ++x) {
          rows.push(gpu_buffers[i], gpu_buffers[i], static_cast<size_t>(src + x * job.src_step),
                    static_cast<size_t>(dst + x * job.dst_step), 1U);
        }
      }
    }
  }
  shadow_rows_ = rows.finalise(grid.system());
}

template <typename T,
          int Components>
auto
HaloDisgDistributed<T,
                    Components>::build_brick_() -> void {
  namespace ddd = detail::dense_disg_distributed;

  const DenseDisgDistributed& grid = field_.dense_disg_distributed();
  const auto& spans = grid.spans();
  const auto& gpu_buffers = field_.gpu_buffers();
  const size_t n_gpus = spans.size();
  const bool has_gpu_buffers = (gpu_buffers.size() == n_gpus);
  const ProcessGrid& pg = grid.process_grid();
  const BoundaryModeDistributed mode = grid.boundary_mode();
  const int32_t first_rank = grid.local_rank_offset();

  if (spans[0].halo_z() <= 0) {
    return;
  }

  auto local_index = [&](int32_t rank) noexcept -> std::optional<size_t> {
    const int64_t i = static_cast<int64_t>(rank) - first_rank;
    if (i >= 0 && i < static_cast<int64_t>(n_gpus)) {
      return static_cast<size_t>(i);
    }
    return std::nullopt;
  };

  detail::halo_build::IntraTransferBuilder<T> intra;
#ifdef CARBON_HAVE_MPI_NCCL
  detail::halo_build::DistTransferBuilder<T> dist;
#endif
  shadow_.assign(n_gpus, ddd::RowCopyArgs{});

  for (size_t i = 0; i < n_gpus; ++i) {
    const ddd::Span& s = spans[i];
    const int32_3d coords = pg.coords_of(first_rank + static_cast<int32_t>(i));
    for (const ddd::BrickDir d : ddd::kBrickDirs) {
      if ((d.dx != 0 && s.halo_x() <= 0) || (d.dy != 0 && s.halo_y() <= 0) ||
          (d.dz != 0 && s.halo_z() <= 0)) {
        continue;
      }
      const auto run = ddd::brick_run<Components>(semantic_, stencil_, d);
      if (!run.has_value()) {
        continue;
      }
      const ddd::Region send = ddd::send_region(d);
      const ddd::Region recv = ddd::recv_region(d);

      if (const auto to = ddd::brick_peer(pg, coords, d, mode); to.has_value()) {
        const int32_t peer_rank = pg.rank_of(*to);
        ddd::append_brick_send_refresh(s, Components, d, *run, shadow_[i]);
        const ddd::PencilMessage src = ddd::region_run(s, send, Components, *run);
        if (const auto j = local_index(peer_rank); j.has_value()) {
          const ddd::PencilMessage dst = ddd::region_run(spans[*j], recv, Components, *run);
          if (src.count != dst.count) {
            log::throw_with_log<std::logic_error>(
                "HaloDisgDistributed: brick message size mismatch between local chunks {} and {} "
                "(direction ({}, {}, {}), {} vs {} elements)",
                i, *j, d.dx, d.dy, d.dz, src.count, dst.count);
          }
          if (has_gpu_buffers) {
            intra.push(gpu_buffers[i], gpu_buffers[*j], src.first, dst.first, src.count);
          }
          host_intra_.push_back({i, *j, src.first, dst.first, src.count});
        } else {
          ++host_dist_pending_;
#ifdef CARBON_HAVE_MPI_NCCL
          if (!has_gpu_buffers) {
            log::throw_with_log<std::invalid_argument>(
                "HaloDisgDistributed: cross-rank halo on a host-only field (MemorySide::Host) is "
                "not yet supported; use MemorySide::HostDevice (or a single-rank layout)");
          }
          dist.push(peer_rank, DistOp::Send, src.first, src.count);
#else
          log::throw_with_log<std::invalid_argument>(
              "HaloDisgDistributed: brick peer rank {} is on another process, but Carbon was "
              "built without CARBON_ENABLE_MPI=ON; build with MPI to enable cross-rank halo "
              "exchange",
              peer_rank);
#endif
        }
      }

      if (const auto from = ddd::brick_peer(pg, coords, ddd::BrickDir{-d.dx, -d.dy, -d.dz}, mode);
          from.has_value()) {
        const int32_t peer_rank = pg.rank_of(*from);
        if (!local_index(peer_rank).has_value()) {
          ++host_dist_pending_;
#ifdef CARBON_HAVE_MPI_NCCL
          const ddd::PencilMessage dst = ddd::region_run(s, recv, Components, *run);
          dist.push(peer_rank, DistOp::Recv, dst.first, dst.count);
#endif
        }
      }
    }
  }

  finalise_shadows_();

  intra_ = intra.finalise(grid.system());
#ifdef CARBON_HAVE_MPI_NCCL
  dist_ = dist.finalise(grid.system(), gpu_buffers);
#endif

  if (intra_.src_counts().size() > 0U) {
    System sys = grid.system();
    fence_event_ = sys.event();
  }
}

template <typename T,
          int Components>
HaloDisgDistributed(DenseDisgDistributed::Field<T,
                                                Components>) -> HaloDisgDistributed<T,
                                                                                    Components>;

template <typename T,
          int Components>
HaloDisgDistributed(DenseDisgDistributed::Field<T,
                                                Components>,
                    Stencil,
                    StencilSemantic) -> HaloDisgDistributed<T,
                                                            Components>;

}  // namespace carbon
