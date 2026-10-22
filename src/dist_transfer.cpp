#include <carbon/dist_transfer.h>

#include <carbon/detail/backend/select.h>
#include <carbon/detail/communicator.h>
#include <carbon/distributed.h>
#include <carbon/logging.h>

#include "communicator_internal.h"
#include "distributed_internal.h"
#include "gpu_runtime_compat.h"

#include <mpi.h>

#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif

#if CARBON_BACKEND_IS_SYCL
#include <sycl/sycl.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace carbon::dist_transfer_detail {

namespace {

[[noreturn]] auto
throw_nccl_error(const char* where,
                 ncclResult_t res) -> void {
  log::throw_with_log<std::runtime_error>("DistTransfer: {} failed: {}", where,
                                          ncclGetErrorString(res));
}

inline auto
check_nccl(const char* where,
           ncclResult_t res) -> void {
  if (res != ncclSuccess) {
    throw_nccl_error(where, res);
  }
}

[[noreturn]] auto
throw_mpi_error(const char* where,
                int err) -> void {
  char buf[MPI_MAX_ERROR_STRING] = {};
  int len = 0;
  if (MPI_Error_string(err, buf, &len) != MPI_SUCCESS) {
    log::throw_with_log<std::runtime_error>("DistTransfer::run_host: {} failed with MPI error {}",
                                            where, err);
  }
  log::throw_with_log<std::runtime_error>("DistTransfer::run_host: {} failed: {}", where, buf);
}

inline auto
check_mpi(const char* where,
          int err) -> void {
  if (err != MPI_SUCCESS) {
    throw_mpi_error(where, err);
  }
}

[[nodiscard]] auto
dist_transport_is_mpi() noexcept -> bool {
  static const bool is_mpi = [] {
    const char* value = std::getenv("CARBON_DIST_TRANSPORT");
    return value != nullptr && std::strcmp(value, "mpi") == 0;
  }();
  return is_mpi;
}

[[nodiscard]] auto
mpi_cuda_aware() noexcept -> bool {
#if defined(OMPI_MAJOR_VERSION)
  static const bool aware = MPIX_Query_cuda_support() != 0;
  return aware;
#else
  return false;
#endif
}

constexpr int k_mpi_device_tag = 0;

auto
run_mpi_device(const System& system,
               void* local_buffer,
               const std::vector<int>& peer_ranks,
               const std::vector<DistOp>& ops,
               const std::vector<size_t>& first_bytes,
               const std::vector<size_t>& byte_counts,
               const Stream& stream) -> void {
  if (byte_counts.empty()) {
    return;
  }
  const auto* dist = system.distributed();
  if (dist == nullptr) {
    log::throw_with_log<std::runtime_error>(
        "DistTransfer::run_mpi_device: System has no Distributed handle (was the Distributed "
        "handle destroyed prematurely?)");
  }
  if (peer_ranks.size() != byte_counts.size() || ops.size() != byte_counts.size() ||
      first_bytes.size() != byte_counts.size()) {
    log::throw_with_log<std::invalid_argument>(
        "DistTransfer::run_mpi_device: parallel vectors disagree (peer_ranks={}, ops={}, "
        "first_bytes={}, byte_counts={})",
        peer_ranks.size(), ops.size(), first_bytes.size(), byte_counts.size());
  }

  // MPI is not stream-ordered like NCCL; drain the per-rank stream so source owned-cells are
  // written before Send and subsequent kernels see completed Recv writes.
  stream.wait();

  MPI_Comm comm = detail::distributed_mpi_world(*dist);
  auto* base = static_cast<std::uint8_t*>(local_buffer);
  const bool cuda_aware = mpi_cuda_aware();

  std::vector<std::vector<std::byte>> staging;
  if (!cuda_aware) {
    staging.resize(byte_counts.size());
  }

  std::vector<MPI_Request> requests;
  requests.reserve(byte_counts.size());

  for (size_t i = 0; i < byte_counts.size(); ++i) {
    if (byte_counts[i] == 0U) {
      continue;
    }
    void* device_ptr = static_cast<void*>(base + first_bytes[i]);
    void* mpi_ptr = device_ptr;
    const int count = static_cast<int>(byte_counts[i]);

    if (!cuda_aware) {
      staging[i].resize(byte_counts[i]);
      mpi_ptr = staging[i].data();
      if (ops[i] == DistOp::Send) {
#if CARBON_BACKEND_IS_CUDALIKE
        using detail::gpu_compat::gpu_error_t;
        using detail::gpu_compat::gpu_get_error_string;
        using detail::gpu_compat::gpu_memcpy;
        using detail::gpu_compat::gpu_memcpy_device_to_host;
        using detail::gpu_compat::gpu_success;
        const gpu_error_t err =
            gpu_memcpy(mpi_ptr, device_ptr, byte_counts[i], gpu_memcpy_device_to_host);
        if (err != gpu_success) {
          log::throw_with_log<std::runtime_error>(
              "DistTransfer::run_mpi_device: device-to-host copy failed: {}",
              gpu_get_error_string(err));
        }
#elif CARBON_BACKEND_IS_SYCL
        sycl::queue& q = stream[0].queue();
        q.memcpy(mpi_ptr, device_ptr, byte_counts[i]).wait_and_throw();
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
      }
    }

    MPI_Request req = MPI_REQUEST_NULL;
    if (ops[i] == DistOp::Recv) {
      check_mpi("MPI_Irecv",
                MPI_Irecv(mpi_ptr, count, MPI_BYTE, peer_ranks[i], k_mpi_device_tag, comm, &req));
    } else {
      check_mpi("MPI_Isend",
                MPI_Isend(mpi_ptr, count, MPI_BYTE, peer_ranks[i], k_mpi_device_tag, comm, &req));
    }
    requests.push_back(req);
  }

  if (!requests.empty()) {
    check_mpi("MPI_Waitall",
              MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
  }

  if (!cuda_aware) {
    for (size_t i = 0; i < byte_counts.size(); ++i) {
      if (byte_counts[i] == 0U) {
        continue;
      }
      if (ops[i] == DistOp::Recv) {
        void* device_ptr = static_cast<void*>(base + first_bytes[i]);
        void* host_ptr = staging[i].data();
#if CARBON_BACKEND_IS_CUDALIKE
        using detail::gpu_compat::gpu_error_t;
        using detail::gpu_compat::gpu_get_error_string;
        using detail::gpu_compat::gpu_memcpy;
        using detail::gpu_compat::gpu_memcpy_host_to_device;
        using detail::gpu_compat::gpu_success;
        const gpu_error_t err =
            gpu_memcpy(device_ptr, host_ptr, byte_counts[i], gpu_memcpy_host_to_device);
        if (err != gpu_success) {
          log::throw_with_log<std::runtime_error>(
              "DistTransfer::run_mpi_device: host-to-device copy failed: {}",
              gpu_get_error_string(err));
        }
#elif CARBON_BACKEND_IS_SYCL
        sycl::queue& q = stream[0].queue();
        q.memcpy(device_ptr, host_ptr, byte_counts[i]).wait_and_throw();
#endif
      }
    }
  }
}

}  // namespace

auto
run(const System& system,
    void* local_buffer,
    const std::vector<int>& peer_ranks,
    const std::vector<DistOp>& ops,
    const std::vector<size_t>& first_bytes,
    const std::vector<size_t>& byte_counts,
    const Stream& stream) -> void {
  const auto* comm_ptr = system.communicator();
  if (comm_ptr == nullptr) {
    log::throw_with_log<std::runtime_error>(
        "DistTransfer::run: System has no Communicator (was the Distributed handle destroyed "
        "prematurely?)");
  }
  if (stream.size() != 1U) {
    log::throw_with_log<std::invalid_argument>(
        "DistTransfer::run: Stream must hold exactly one per-rank stream (got {})", stream.size());
  }
  if (dist_transport_is_mpi()) {
    run_mpi_device(system, local_buffer, peer_ranks, ops, first_bytes, byte_counts, stream);
    return;
  }
  ncclComm_t nccl_comm = detail::communicator_handle(*comm_ptr);
  auto* base = static_cast<std::uint8_t*>(local_buffer);

  // Helper that fires the NCCL/RCCL group once we have the underlying vendor stream
  // (cudaStream_t or hipStream_t) in hand. RCCL is ABI-compatible with NCCL, so the
  // body is identical save for the typedef pulled in by gpu_runtime_compat.h.
  using detail::gpu_compat::gpu_stream_t;
  auto submit_nccl_group = [&](gpu_stream_t cs) {
    check_nccl("ncclGroupStart", ncclGroupStart());
    for (size_t i = 0; i < byte_counts.size(); ++i) {
      if (byte_counts[i] == 0U) {
        continue;
      }
      void* ptr = static_cast<void*>(base + first_bytes[i]);
      if (ops[i] == DistOp::Send) {
        check_nccl("ncclSend",
                   ncclSend(ptr, byte_counts[i], ncclChar, peer_ranks[i], nccl_comm, cs));
      } else {
        check_nccl("ncclRecv",
                   ncclRecv(ptr, byte_counts[i], ncclChar, peer_ranks[i], nccl_comm, cs));
      }
    }
    check_nccl("ncclGroupEnd", ncclGroupEnd());
  };

#if CARBON_BACKEND_IS_CUDALIKE
  // CUDA / native-HIP backend: the per-rank handle is already a cudaStream_t / hipStream_t;
  // submit immediately so the NCCL/RCCL group is ordered after preceding kernel launches
  // and async memcpys on the same stream.
  submit_nccl_group(stream[0].get());
#else
  // SYCL backend: AdaptiveCpp does not expose sycl::get_native<sycl::backend::*>(queue)
  // outside a command group. We wrap the NCCL/RCCL submission inside an
  // AdaptiveCpp_enqueue_custom_operation (AdaptiveCpp's interop equivalent of SYCL 2020
  // host_task — the latter is not exposed on hipsycl::sycl::handler). The interop_handle
  // yields the queue's underlying vendor stream (CUstream under cuda: target,
  // hipStream_t under hip: target); AdaptiveCpp ensures the custom op fires after
  // preceding work on the in-order queue is in flight, and the stream returned is the
  // same one those preceding submissions used, so the NCCL/RCCL group chains correctly
  // without an explicit stream sync. Subsequent SYCL submissions on the queue see the
  // custom op as a dependency, so they automatically wait for the collective to be
  // enqueued (note: enqueued, not completed; the chain stays asynchronous on the device
  // side, which is exactly the CUDA-backend behaviour).
  sycl::queue& q = stream[0].queue();
  q.submit([&](sycl::handler& cgh) {
    cgh.AdaptiveCpp_enqueue_custom_operation([&submit_nccl_group](sycl::interop_handle ih) {
#ifdef CARBON_USE_RCCL
      submit_nccl_group(ih.get_native_queue<sycl::backend::hip>());
#else
      submit_nccl_group(ih.get_native_queue<sycl::backend::cuda>());
#endif
    });
  });
  // Wait for the custom op to enqueue the NCCL/RCCL group before returning, so
  // DistTransfer keeps its synchronous-looking call shape (the user typically pairs
  // operator() with a Stream::wait() at the end of the iteration). This is a host-side
  // wait on the SYCL submission; the device-side ops remain asynchronous on the
  // underlying vendor stream.
  q.wait_and_throw();
#endif
}

auto
run_host(const System& system,
         void* local_buffer,
         const std::vector<int>& peer_ranks,
         const std::vector<DistOp>& ops,
         const std::vector<size_t>& first_bytes,
         const std::vector<size_t>& byte_counts,
         const std::vector<int>& mpi_tags) -> void {
  if (byte_counts.empty()) {
    return;
  }
  const auto* dist = system.distributed();
  if (dist == nullptr) {
    log::throw_with_log<std::runtime_error>(
        "DistTransfer::run_host: System has no Distributed handle (was the Distributed handle "
        "destroyed prematurely?)");
  }
  if (peer_ranks.size() != byte_counts.size() || ops.size() != byte_counts.size() ||
      first_bytes.size() != byte_counts.size() || mpi_tags.size() != byte_counts.size()) {
    log::throw_with_log<std::invalid_argument>(
        "DistTransfer::run_host: parallel vectors disagree (peer_ranks={}, ops={}, "
        "first_bytes={}, byte_counts={}, mpi_tags={})",
        peer_ranks.size(), ops.size(), first_bytes.size(), byte_counts.size(), mpi_tags.size());
  }
  MPI_Comm comm = detail::distributed_mpi_world(*dist);
  auto* base = static_cast<std::uint8_t*>(local_buffer);

  std::vector<MPI_Request> requests;
  requests.reserve(byte_counts.size());
  for (size_t i = 0; i < byte_counts.size(); ++i) {
    if (byte_counts[i] == 0U) {
      continue;
    }
    void* const ptr = static_cast<void*>(base + first_bytes[i]);
    MPI_Request req = MPI_REQUEST_NULL;
    const int count = static_cast<int>(byte_counts[i]);
    if (ops[i] == DistOp::Recv) {
      check_mpi("MPI_Irecv",
                MPI_Irecv(ptr, count, MPI_BYTE, peer_ranks[i], mpi_tags[i], comm, &req));
    } else {
      check_mpi("MPI_Isend",
                MPI_Isend(ptr, count, MPI_BYTE, peer_ranks[i], mpi_tags[i], comm, &req));
    }
    requests.push_back(req);
  }
  if (!requests.empty()) {
    check_mpi("MPI_Waitall",
              MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
  }
}

}  // namespace carbon::dist_transfer_detail
