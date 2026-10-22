#pragma once

#include <carbon/export.h>

#include <memory>

namespace carbon {

class Distributed;

namespace detail {

// Forward declaration of the internal-use bridge that grants @c src/dist_transfer.cpp access
// to the underlying @c ncclComm_t held by @ref Communicator. Declared here so we can friend it
// without dragging @c <nccl.h> into this public header; the actual definition lives in
// @c src/communicator_internal.h, where @c <nccl.h> is already available. The header is not
// installed, so user code cannot reach the back-door.
struct nccl_bridge;

/**
 * One NCCL communicator bound to this process's pinned CUDA device.
 *
 * Built via @ref from_mpi using a @ref Distributed handle: rank 0 calls @c ncclGetUniqueId,
 * the id is broadcast over @c MPI_COMM_WORLD, then every rank calls @c ncclCommInitRank with
 * @c (world_size, world_rank). The destructor calls @c ncclCommDestroy. The header pulls in
 * @c <nccl.h> and @c <cuda_runtime.h> only when its @c .cpp / @c .cu users include the
 * matching internal header (@c communicator_internal.h); the public surface forward-declares
 * the type and keeps @c <nccl.h> out of any user translation unit.
 *
 * @c Communicator must be destroyed before the @c Distributed it was built from
 * (NCCL needs MPI live to drain in-flight ops, and @c ncclCommDestroy must run before
 * @c MPI_Finalize). @ref System::State enforces this through member declaration order.
 */
class CARBON_API Communicator {
public:
  Communicator(const Communicator&) = delete;
  Communicator&
  operator=(const Communicator&) = delete;
  Communicator(Communicator&&) = delete;
  Communicator&
  operator=(Communicator&&) = delete;

  ~Communicator();

  /**
   * Bootstrap one NCCL communicator covering every rank in @p distributed (one rank per
   * process / per GPU under Carbon's distributed convention). The caller must have already
   * pinned the rank's CUDA device via @ref Distributed::pin_local_device or equivalent.
   */
  [[nodiscard]] static auto
  from_mpi(const Distributed& distributed) -> std::shared_ptr<Communicator>;

  /** This communicator's NCCL rank (== @c distributed.world_rank()). */
  [[nodiscard]] auto
  rank() const noexcept -> int;

  /** Total number of NCCL ranks in this communicator (== @c distributed.world_size()). */
  [[nodiscard]] auto
  size() const noexcept -> int;

  /** Local CUDA device ordinal this communicator is bound to. */
  [[nodiscard]] auto
  device() const noexcept -> int;

private:
  Communicator();

  // Friend the namespace-scope bridge declared above so it can call @ref raw_handle without
  // exposing it on the public surface.
  friend struct nccl_bridge;

  /**
   * Underlying @c ncclComm_t as @c void*. Reachable only through @ref nccl_bridge; the
   * bridge performs the @c reinterpret_cast back to @c ncclComm_t in a TU that has the NCCL
   * header included. Not part of the user-facing API.
   */
  [[nodiscard]] auto
  raw_handle() const noexcept -> void*;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace detail
}  // namespace carbon
