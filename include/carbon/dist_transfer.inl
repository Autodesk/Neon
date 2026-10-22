#pragma once

// Definition of DistTransfer<T>::operator(). Forwards to a non-template, byte-typed dispatch
// helper that lives in src/dist_transfer.cpp; that .cpp owns every NCCL include so this header
// never has to pull in <nccl.h>. Available only when Carbon was built with
// CARBON_ENABLE_MPI=ON.

#ifdef CARBON_HAVE_MPI_NCCL

#include <carbon/detail/buffer_group.h>
#include <carbon/detail/stream_group.h>
#include <carbon/dist_transfer.h>
#include <carbon/system.h>

#include <cstddef>
#include <vector>

namespace carbon::dist_transfer_detail {

/**
 * Byte-typed dispatch helper backing every @ref DistTransfer<T>::operator() instantiation. The
 * template only converts element offsets / counts to bytes and forwards the local buffer's
 * raw pointer; this keeps @c <nccl.h> entirely out of public headers. Lives in
 * @c src/dist_transfer.cpp.
 */
CARBON_API auto
run(const System& system,
    void* local_buffer,
    const std::vector<int>& peer_ranks,
    const std::vector<DistOp>& ops,
    const std::vector<size_t>& first_bytes,
    const std::vector<size_t>& byte_counts,
    const Stream& stream) -> void;

/**
 * Synchronous host-side counterpart to @ref run : posts every non-zero entry as
 * @c MPI_Isend / @c MPI_Irecv against @p local_buffer (byte offsets), waits on
 * @c MPI_Waitall , and throws on MPI errors. @p mpi_tags[i] is the MPI message tag
 * for entry @c i (Send/Recv peers must agree on tag + byte count). Used by
 * @c HaloBlock::launch_on_host and siblings for the cross-rank host halo fallback.
 */
CARBON_API auto
run_host(const System& system,
         void* local_buffer,
         const std::vector<int>& peer_ranks,
         const std::vector<DistOp>& ops,
         const std::vector<size_t>& first_bytes,
         const std::vector<size_t>& byte_counts,
         const std::vector<int>& mpi_tags) -> void;

}  // namespace carbon::dist_transfer_detail

namespace carbon {

template <typename T>
inline auto
DistTransfer<T>::operator()(const Stream& stream) const -> void {
  if (counts_.empty()) {
    return;
  }
  std::vector<size_t> first_bytes(counts_.size());
  std::vector<size_t> byte_counts(counts_.size());
  for (size_t i = 0; i < counts_.size(); ++i) {
    first_bytes[i] = first_elements_[i] * sizeof(T);
    byte_counts[i] = counts_[i] * sizeof(T);
  }
  dist_transfer_detail::run(system_, static_cast<void*>(buffer_[0].data()), peer_ranks_, ops_,
                            first_bytes, byte_counts, stream);
}

}  // namespace carbon

#endif  // CARBON_HAVE_MPI_NCCL
