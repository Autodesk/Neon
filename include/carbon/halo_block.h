#pragma once

#include <carbon/block.h>
#include <carbon/detail/block/field.h>
#include <carbon/detail/buffer_group.h>
#include <carbon/detail/halo_build.h>
#include <carbon/logging.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include <carbon/dist_transfer.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace carbon {

/**
 * @brief Halo-exchange descriptor for a single @ref Block::Field .
 *
 * @par What it does
 * @ref Block partitions the brick z-axis across GPUs and reserves @b ghost slots at
 * each GPU's lower / upper z-edge (mirrors of retained bricks owned by the adjacent
 * GPU). @c HaloBlock turns the per-adjacency @c halo_pairs_upward /
 * @c halo_pairs_downward plan @ref Block builds at construction into a single
 * intra-process @ref Transfer<T> covering every @c (src_owned → dst_ghost) brick.
 * Each pair copies @c VoxelCount · Components · sizeof(T) bytes (whole brick, all
 * components fused — the same shape Neon's @c bField::initHaloUpdateTable encodes).
 *
 * Cross-rank halo (MPI / NCCL) is wired through @ref DistTransfer<T> when
 * @c CARBON_HAVE_MPI_NCCL is set @b and the @ref Block was built on a
 * distributed @c System (one rank per GPU, cf. @c System::distributed() ).
 * Send / Recv entries are computed from @c Block::halo_dist_send_upward /
 * @c halo_dist_recv_upward / @c halo_dist_send_downward /
 * @c halo_dist_recv_downward at construction; @c operator()(stream) issues
 * the intra-process @c Transfer<T> first, then the cross-rank
 * @c DistTransfer<T> inside one @c ncclGroupStart / @c ncclGroupEnd block,
 * mirroring the structure of @ref Halo .
 *
 * @par Lifecycle
 * Cheap to construct (the @c Transfer just records src/dst byte offsets), copyable,
 * and the field's per-GPU buffers are captured by reference-count semantics through
 * @c Buffer<T> 's underlying @c Allocation . Re-issue via @c operator()(stream) any
 * time the owned bricks change — typical use is one halo per Jacobi iteration before
 * the stencil kernel runs.
 *
 * @tparam T          Element type (must be @c TriviallyCopyable ).
 * @tparam Components Number of components per voxel (@c >= 1 ).
 * @tparam SBlock     @c StaticBlock policy of the underlying @ref Block .
 */
template <typename T, int Components, typename SBlock = detail::block::DefaultStaticBlock>
class HaloBlock {
public:
  using Grid = ::carbon::Block<SBlock>;
  using Field = typename Grid::template Field<T, Components>;
  static constexpr int VoxelCount = SBlock::VoxelCount;

  /** Empty halo: @ref operator() is a no-op. */
  HaloBlock() = default;

  /**
   * Build the halo plan for @p field hosted on @p block . The constructor walks
   * every @c (g, g+1) adjacency, aliases the field's per-GPU device buffers as
   * @c Transfer<T> source / destination per-pair endpoints, and records per-pair
   * byte offsets sized to one full brick (all components fused into a single
   * contiguous transfer, since brick storage is SoA per-block —
   * @c (block_idx · Components + c) · VoxelCount + voxel — so all components for a
   * given brick are contiguous in memory).
   *
   * @throws std::invalid_argument if @p field 's per-GPU device buffer cardinality
   *         does not match @p block 's GPU count.
   */
  HaloBlock(const Grid& block,
            Field& field)
      : field_(field),
        field_uid_(field.gpu_buffers().get_uid()),
        field_name_(field.name()) {
    const size_t n_gpus = block.gpu_count();
    if (field.gpu_buffers().size() != n_gpus) {
      log::throw_with_log<std::invalid_argument>(
          "HaloBlock: field has {} per-GPU device buffers but Block has {} GPUs",
          field.gpu_buffers().size(), n_gpus);
    }
    const size_t brick_count = static_cast<size_t>(VoxelCount) * static_cast<size_t>(Components);

    // Intra-process pairs (single-rank multi-GPU). Build one flat list of
    // (src_alloc, dst_alloc, src_first, dst_first, count) entries that we'll
    // hand to a single Transfer<T> at the end. Aliasing the field's per-GPU
    // Allocation<T> handles into one BufferGroup<T> works because
    // BufferGroup::from_vector accepts any cardinality and routes per-pair
    // through System::async_copy (which dispatches cudaMemcpyPeer when
    // src_gpu != dst_gpu).
    //
    // The same iteration populates @c host_intra_ — a parallel list of
    // (src_gpu, dst_gpu, src_offset, dst_offset, count) entries the host
    // pipeline replays via @c std::memcpy in @ref launch_on_host . Block's
    // host mirror is sized identically to the device buffer (same brick
    // layout, same offsets — see @c Block::newField ), so the offsets reuse
    // the device-side math without translation.
    if (n_gpus >= 2U) {
      std::vector<detail::Allocation<T>> src_allocs;
      std::vector<detail::Allocation<T>> dst_allocs;
      std::vector<size_t> src_firsts;
      std::vector<size_t> src_counts;
      std::vector<size_t> dst_firsts;
      std::vector<size_t> dst_counts;
      for (size_t adj = 0; adj + 1U < n_gpus; ++adj) {
        for (const auto& pair : block.halo_pairs_upward(adj)) {
          const size_t src_first = static_cast<size_t>(pair.src_local_id) * brick_count;
          const size_t dst_first = static_cast<size_t>(pair.dst_local_id) * brick_count;
          src_allocs.push_back(field.gpu_buffers()[adj]);
          dst_allocs.push_back(field.gpu_buffers()[adj + 1]);
          src_firsts.push_back(src_first);
          src_counts.push_back(brick_count);
          dst_firsts.push_back(dst_first);
          dst_counts.push_back(brick_count);
          host_intra_.push_back(HostIntraEntry{adj, adj + 1U, src_first, dst_first, brick_count});
        }
        for (const auto& pair : block.halo_pairs_downward(adj)) {
          const size_t src_first = static_cast<size_t>(pair.src_local_id) * brick_count;
          const size_t dst_first = static_cast<size_t>(pair.dst_local_id) * brick_count;
          src_allocs.push_back(field.gpu_buffers()[adj + 1]);
          dst_allocs.push_back(field.gpu_buffers()[adj]);
          src_firsts.push_back(src_first);
          src_counts.push_back(brick_count);
          dst_firsts.push_back(dst_first);
          dst_counts.push_back(brick_count);
          host_intra_.push_back(HostIntraEntry{adj + 1U, adj, src_first, dst_first, brick_count});
        }
      }
      if (!src_allocs.empty()) {
        detail::halo_build::coalesce_contiguous_transfers(src_allocs, dst_allocs, src_firsts,
                                                          src_counts, dst_firsts, dst_counts);
        detail::halo_build::coalesce_contiguous_host_intra(host_intra_);
      }
      pair_count_ = src_allocs.size();
      if (pair_count_ != 0U) {
        auto src_group = detail::BufferGroup<T>::from_vector(std::move(src_allocs));
        auto dst_group = detail::BufferGroup<T>::from_vector(std::move(dst_allocs));
        transfer_ = Transfer<T>(block.system(), std::move(src_group), std::move(src_firsts),
                                std::move(src_counts), std::move(dst_group), std::move(dst_firsts),
                                std::move(dst_counts));
        // Cross-GPU producer fence backing operator() (see carbon::Halo). Only
        // multi-local-GPU layouts have intra brick copies that can race source-GPU
        // compute; allocate the per-GPU event group from a shallow System copy.
        System sys = block.system();
        fence_event_ = sys.event();
      }
    }

#ifdef CARBON_HAVE_MPI_NCCL
    // Cross-rank entries. Block has already separated Send / Recv into four
    // per-direction lists (halo_dist_*); we flatten them into one
    // DistTransfer<T> posted against this rank's lone field buffer
    // (one-rank-per-GPU). FIFO order across the matching list pairs (this
    // rank's send_upward vs. peer's recv_upward, etc.) is preserved by Block's
    // row-major (bx, by) sweep, so NCCL's positional pairing matches up.
    if (block.system().distributed() != nullptr) {
      if (n_gpus != 1U) {
        log::throw_with_log<std::invalid_argument>(
            "HaloBlock: distributed System must have exactly one GPU per rank "
            "(gpu_count={})",
            n_gpus);
      }
      std::vector<int> dist_peers;
      std::vector<DistOp> dist_ops;
      std::vector<size_t> dist_firsts;
      std::vector<size_t> dist_counts_v;
      auto add = [&](const auto& entries, DistOp op, int dir_channel) {
        size_t dir_idx = 0U;
        for (const auto& e : entries) {
          dist_peers.push_back(e.peer_rank);
          dist_ops.push_back(op);
          dist_firsts.push_back(static_cast<size_t>(e.local_id) * brick_count);
          dist_counts_v.push_back(brick_count);
          host_dist_tags_.push_back((dir_channel << 16) | static_cast<int>(dir_idx));
          ++dir_idx;
        }
      };
      add(block.halo_dist_send_upward(), DistOp::Send, 0);
      add(block.halo_dist_recv_upward(), DistOp::Recv, 0);
      add(block.halo_dist_send_downward(), DistOp::Send, 1);
      add(block.halo_dist_recv_downward(), DistOp::Recv, 1);
      dist_entry_count_ = dist_peers.size();
      host_dist_pending_ = 0U;
      if (dist_entry_count_ != 0U) {
        dist_ =
            DistTransfer<T>(block.system(), field.gpu_buffers(), std::move(dist_peers),
                            std::move(dist_ops), std::move(dist_firsts), std::move(dist_counts_v));
      }
    }
#endif
  }

  /**
   * Issue every brick transfer on @p stream . Async; the caller is responsible for
   * the stream-side ordering (typically: launch on the @c System 's default stream,
   * then queue the dependent stencil kernel on the same stream so the halo lands
   * before the kernel reads ghosts).
   */
  auto
  operator()(const Stream& stream) const -> void {
    if (pair_count_ != 0U) {
      // Cross-GPU producer fence — see the matching note on @c carbon::Halo::operator()
      // and @c StreamGroup::wait_for_all. The intra brick copies read source-GPU owned
      // bricks written by prior compute on the source GPU's stream, but a peer copy on
      // the destination GPU's stream has no implicit ordering against it. fence_event_
      // is empty for single-GPU layouts (pair_count_ == 0 already skips this branch).
      if (!fence_event_.empty()) {
        fence_event_.record(stream);
        stream.wait_for_all(fence_event_);
      }
      transfer_(stream);
    }
#ifdef CARBON_HAVE_MPI_NCCL
    if (dist_entry_count_ != 0U) {
      dist_(stream);
    }
#endif
  }

  /**
   * Host (CPU) variant of @ref operator() — replays the @c host_intra_ entries
   * built at construction as synchronous @c std::memcpy s between adjacent
   * GPUs' host mirrors of the field. The natural pair to a Block kernel
   * dispatched through @c carbon::Kernel::launch_on_host for fully-host
   * pipelines on Block.
   *
   * @par Single-rank vs distributed
   * - Single-rank multi-GPU: works the same way as the device path,
   *   intra-process @c std::memcpy between adjacent GPUs' host mirrors of the
   *   field. The host allocation is sized identically to the device buffer
   *   (same brick layout, same offsets — @c Block::newField sizes
   *   @c cpu_buffers from the same per-GPU count vector as @c gpu_buffers ),
   *   so the offsets are reused without translation.
   * - Cross-rank: replays the cross-rank entries built at construction through
   *   @c dist_transfer_detail::run_host ( @c MPI_Isend / @c MPI_Irecv with tags
   *   that mirror the device @c DistTransfer FIFO pairing per direction), then
   *   runs any intra-process @c std::memcpy entries.
   *
   * Synchronous: returns when every memcpy has completed (no async work is
   * queued). @p stream is waited on first so any pending device work that
   * stages writes into the field's host mirror via @c System::async_copy
   * lands before this method reads it back.
   *
   * @throws std::invalid_argument when the field has no host mirror to write
   *         into (built with @c MemorySide::Device ).
   * @throws std::runtime_error on MPI errors from the cross-rank host exchange.
   */
  auto
  launch_on_host(const Stream& stream) const -> void {
    stream.wait();
    const auto& cpu = field_.cpu_buffers();
    if (cpu.size() == 0U) {
      log::throw_with_log<std::invalid_argument>(
          "HaloBlock::launch_on_host: field has no host mirror (cpu_buffers().size() == 0); "
          "construct the field with MemorySide::Host or MemorySide::HostDevice");
    }
#ifdef CARBON_HAVE_MPI_NCCL
    if (dist_entry_count_ != 0U) {
      std::vector<size_t> first_bytes(dist_entry_count_);
      std::vector<size_t> byte_counts(dist_entry_count_);
      for (size_t i = 0; i < dist_entry_count_; ++i) {
        first_bytes[i] = dist_.first_elements()[i] * sizeof(T);
        byte_counts[i] = dist_.counts()[i] * sizeof(T);
      }
      dist_transfer_detail::run_host(dist_.system(), static_cast<void*>(cpu[0].data()),
                                     dist_.peer_ranks(), dist_.ops(), first_bytes, byte_counts,
                                     host_dist_tags_);
    }
#endif
    for (const HostIntraEntry& e : host_intra_) {
      T* const dst_ptr = cpu[e.dst_gpu].data() + e.dst_offset;
      const T* const src_ptr = cpu[e.src_gpu].data() + e.src_offset;
      std::memcpy(static_cast<void*>(dst_ptr), static_cast<const void*>(src_ptr),
                  e.count * sizeof(T));
    }
  }

  /**
   * Field this descriptor is bound to. Mirrors @ref Halo::field /
   * @ref HaloDisg::field so generic introspection (e.g. @c Skeleton token
   * discovery) can reach the field handle without a Block-specific branch.
   */
  [[nodiscard]] auto
  field() const noexcept -> const Field& {
    return field_;
  }

  /**
   * Buffer-group UID of the field this halo was built over — same value as
   * @c Field::gpu_buffers().get_uid() at construction. Used by
   * @c BlockHaloKernelImpl to emit an @c AccessToken for the dependency
   * tracker.
   */
  [[nodiscard]] auto
  field_uid() const noexcept -> uint64_t {
    return field_uid_;
  }

  /** Diagnostic name of the field this halo was built over (copy of the
   *  field's @c name() at construction). */
  [[nodiscard]] auto
  field_name() const noexcept -> const std::string& {
    return field_name_;
  }

  /** Number of coalesced intra-process device transfers issued by @c operator() . */
  [[nodiscard]] auto
  pair_count() const noexcept -> size_t {
    return pair_count_;
  }

  /**
   * Number of coalesced @c std::memcpy entries the host launch will issue.
   * Matches @ref pair_count on single-rank multi-GPU layouts where each
   * adjacency+direction face is one contiguous run.
   */
  [[nodiscard]] auto
  host_intra_count() const noexcept -> size_t {
    return host_intra_.size();
  }

  /**
   * Number of cross-rank @c ncclSend / @c ncclRecv entries issued by
   * @c operator() . Always zero in non-MPI builds.
   */
  [[nodiscard]] auto
  dist_entry_count() const noexcept -> size_t {
#ifdef CARBON_HAVE_MPI_NCCL
    return dist_entry_count_;
#else
    return 0U;
#endif
  }

private:
  /**
   * Per-direction intra-process host halo entry: a single @c std::memcpy of
   * @c count elements of @p T from @c src_gpu 's host mirror at @c src_offset
   * (in elements) into @c dst_gpu 's host mirror at @c dst_offset .
   *
   * Each entry covers a whole brick (all @c Components fused — the same
   * shape the device @c Transfer<T> path encodes per-pair); @c count is
   * @c VoxelCount * Components for every entry today, but a future
   * asymmetric host plan (cross-rank or per-component) can vary it without
   * touching the launch path.
   */
  struct HostIntraEntry {
    size_t src_gpu{};
    size_t dst_gpu{};
    size_t src_offset{};
    size_t dst_offset{};
    size_t count{};
  };

  Field field_{};
  Transfer<T> transfer_{};
  size_t pair_count_{0};
#ifdef CARBON_HAVE_MPI_NCCL
  DistTransfer<T> dist_{};
  size_t dist_entry_count_{0};
  std::vector<int> host_dist_tags_{};
#endif
  std::vector<HostIntraEntry> host_intra_{};
  // Legacy sentinel — kept for API symmetry with Halo / HaloDisg; always zero once the
  // MPI host fallback is wired (cross-rank entries run through run_host instead).
  size_t host_dist_pending_{0};
  uint64_t field_uid_{0};
  std::string field_name_{};
  // Per-GPU event group backing the cross-GPU producer fence in @ref operator() (see the
  // matching member on @c carbon::Halo). Allocated only when pair_count_ > 0.
  mutable Event fence_event_{};
};

}  // namespace carbon
