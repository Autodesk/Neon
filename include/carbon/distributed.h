#pragma once

#include <carbon/export.h>

#include <cstddef>
#include <memory>

namespace carbon {

namespace detail {
class Communicator;
}  // namespace detail

/**
 * Process-global handle to an MPI session, used to bootstrap NCCL for cross-rank GPU transfers.
 *
 * The distributed transport is gated on the @c CARBON_HAVE_MPI_NCCL compile definition; this
 * header is otherwise unused. Carbon's MPI / NCCL types are kept entirely out of public headers,
 * so user translation units never need @c <mpi.h> or @c <nccl.h>: the implementation lives
 * behind a pImpl.
 *
 * Lifetime
 * --------
 * Calling the constructor checks @c MPI_Initialized and only calls @c MPI_Init_thread if MPI
 * has not already been initialised by the host application. The destructor calls
 * @c MPI_Finalize only when this @c Distributed was the one to initialise MPI; otherwise the
 * host application owns the lifetime and we must not finalise on its behalf.
 *
 * Process / device model
 * ----------------------
 * Carbon's distributed runs are built around the standard MPI+CUDA convention of one rank per
 * GPU. Inside one rank there is no peer copy because a rank owns exactly one device; all
 * cross-GPU transfers go through NCCL on the rank's communicator (which transparently uses
 * NVLink/peer for same-node pairs and IB/RoCE across nodes).
 *
 * Pinning is launcher-driven. The static helper @ref pin_local_device sets the CUDA device for
 * this process; if @c cudaGetDeviceCount reports a single visible device (typical when the
 * launcher already pinned the rank with @c srun --gpus-per-task=1 or @c --map-by ppr:N:node)
 * it selects device 0; otherwise it falls back to @c local_rank() % visible_count and emits a
 * warning recommending launcher-side pinning.
 *
 * Copy / move
 * -----------
 * @c Distributed owns a process-global resource (the MPI lifetime). It is non-copyable and
 * non-movable. Hold one through @c std::shared_ptr<const Distributed> and pass it to
 * @ref System.
 */
class CARBON_API Distributed {
public:
  /**
   * Initialise (or attach to a pre-existing) MPI session. Internally calls
   * @c MPI_Init_thread with @c MPI_THREAD_SERIALIZED only when @c MPI_Initialized reports
   * false. Throws @c std::runtime_error on any MPI error.
   */
  Distributed();

  ~Distributed();

  Distributed(const Distributed&) = delete;
  Distributed&
  operator=(const Distributed&) = delete;
  Distributed(Distributed&&) = delete;
  Distributed&
  operator=(Distributed&&) = delete;

  /** Rank of this process in @c MPI_COMM_WORLD. */
  [[nodiscard]] auto
  world_rank() const noexcept -> int;

  /** Total number of processes in @c MPI_COMM_WORLD. */
  [[nodiscard]] auto
  world_size() const noexcept -> int;

  /**
   * Rank of this process within its node, computed via
   * @c MPI_Comm_split_type(MPI_COMM_TYPE_SHARED). Requires MPI 3.0+.
   */
  [[nodiscard]] auto
  local_rank() const noexcept -> int;

  /**
   * Number of processes sharing this node (the size of the local communicator built by
   * @c MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)).
   */
  [[nodiscard]] auto
  local_size() const noexcept -> int;

  /**
   * Pin the calling thread's CUDA device for the rank's exclusive use. If the launcher has
   * already restricted the rank to a single visible device (e.g. via @c CUDA_VISIBLE_DEVICES),
   * selects device 0; otherwise falls back to @c local_rank() % visible_count and logs a
   * warning. Returns the selected device ordinal.
   */
  [[nodiscard]] auto
  pin_local_device() const -> int;

  /**
   * Whether this @c Distributed instance is the owner of the MPI lifetime (i.e. it called
   * @c MPI_Init_thread itself). The destructor calls @c MPI_Finalize only when this is true.
   */
  [[nodiscard]] auto
  owns_mpi() const noexcept -> bool;

  /**
   * Collective barrier on @c MPI_COMM_WORLD . Wraps @c MPI_Barrier ; throws
   * @c std::runtime_error on any MPI error. Useful when host-side coordination is needed
   * outside the GPU transfer paths (e.g. atomic checkpoint commit, where rank 0's
   * @c manifest.json write must follow every rank's per-shard write).
   */
  void
  barrier() const;

  /**
   * Collective broadcast on @c MPI_COMM_WORLD . Wraps @c MPI_Bcast ; throws
   * @c std::runtime_error on any MPI error. @p root must hold the send buffer on entry;
   * every rank receives the same @p bytes from @p root on exit.
   */
  void
  broadcast(void* data,
            size_t bytes,
            int root) const;

private:
  // Bridge to the underlying MPI_Comm lives in @c src/distributed_internal.h, which is only
  // included from translation units that may already see @c <mpi.h>; this keeps MPI types out
  // of the public umbrella while still letting @c detail::Communicator bootstrap NCCL.
  friend class detail::Communicator;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace carbon
