#include <carbon/reduce_dist.h>

#include <carbon/detail/backend/select.h>
#include <carbon/detail/communicator.h>
#include <carbon/distributed.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include "communicator_internal.h"
#include "distributed_internal.h"
#include "gpu_runtime_compat.h"

#include <mpi.h>

#if CARBON_BACKEND_IS_SYCL
#include <sycl/sycl.hpp>
#endif
#endif  // CARBON_HAVE_MPI_NCCL

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace carbon::reduce {

#ifdef CARBON_HAVE_MPI_NCCL

namespace {

[[noreturn]] auto
throw_nccl_error(const char* where,
                 ncclResult_t res) -> void {
  log::throw_with_log<std::runtime_error>("dist_allreduce_in_place: {} failed: {}", where,
                                          ncclGetErrorString(res));
}

inline auto
check_nccl(const char* where,
           ncclResult_t res) -> void {
  if (res != ncclSuccess) {
    throw_nccl_error(where, res);
  }
}

[[nodiscard]] auto
to_nccl_op(ReduceOpKind op) -> ncclRedOp_t {
  switch (op) {
    case ReduceOpKind::Sum:
      return ncclSum;
    case ReduceOpKind::Min:
      return ncclMin;
    case ReduceOpKind::Max:
      return ncclMax;
    case ReduceOpKind::Prod:
      return ncclProd;
  }
  log::throw_with_log<std::runtime_error>("dist_allreduce_in_place: unknown ReduceOpKind {}",
                                          static_cast<int>(op));
}

[[nodiscard]] auto
to_mpi_op(ReduceOpKind op) -> MPI_Op {
  switch (op) {
    case ReduceOpKind::Sum:
      return MPI_SUM;
    case ReduceOpKind::Min:
      return MPI_MIN;
    case ReduceOpKind::Max:
      return MPI_MAX;
    case ReduceOpKind::Prod:
      return MPI_PROD;
  }
  log::throw_with_log<std::runtime_error>("dist_host_allreduce_in_place: unknown ReduceOpKind {}",
                                          static_cast<int>(op));
}

[[nodiscard]] auto
to_mpi_dtype(ReduceDataKind dt) -> MPI_Datatype {
  switch (dt) {
    case ReduceDataKind::Int8:
      return MPI_INT8_T;
    case ReduceDataKind::Uint8:
      return MPI_UINT8_T;
    case ReduceDataKind::Int32:
      return MPI_INT32_T;
    case ReduceDataKind::Uint32:
      return MPI_UINT32_T;
    case ReduceDataKind::Int64:
      return MPI_INT64_T;
    case ReduceDataKind::Uint64:
      return MPI_UINT64_T;
    case ReduceDataKind::Float32:
      return MPI_FLOAT;
    case ReduceDataKind::Float64:
      return MPI_DOUBLE;
    case ReduceDataKind::Float16:
    case ReduceDataKind::BFloat16:
      // MPI has no canonical fp16 / bf16 datatype; the GPU path covers those via
      // ncclHalf / ncclBfloat16 directly. The host fallback bails so users see a
      // pointed error rather than a silent dtype mismatch on MPI_Allreduce.
      log::throw_with_log<std::runtime_error>(
          "dist_host_allreduce_in_place: MPI has no built-in fp16/bf16 datatype; the host "
          "reduce path does not support these element types — use the device reduce kernel "
          "(stage-3 NCCL allreduce) for fp16/bf16 cross-rank reductions");
  }
  log::throw_with_log<std::runtime_error>("dist_host_allreduce_in_place: unknown ReduceDataKind {}",
                                          static_cast<int>(dt));
}

[[noreturn]] auto
throw_mpi_error(const char* where,
                int code) -> void {
  char msg[MPI_MAX_ERROR_STRING] = {};
  int len = 0;
  ::MPI_Error_string(code, msg, &len);
  log::throw_with_log<std::runtime_error>("dist_host_allreduce_in_place: {} failed: {}", where,
                                          msg);
}

inline auto
check_mpi(const char* where,
          int code) -> void {
  if (code != MPI_SUCCESS) {
    throw_mpi_error(where, code);
  }
}

[[nodiscard]] auto
to_nccl_dtype(ReduceDataKind dt) -> ncclDataType_t {
  switch (dt) {
    case ReduceDataKind::Int8:
      return ncclInt8;
    case ReduceDataKind::Uint8:
      return ncclUint8;
    case ReduceDataKind::Int32:
      return ncclInt32;
    case ReduceDataKind::Uint32:
      return ncclUint32;
    case ReduceDataKind::Int64:
      return ncclInt64;
    case ReduceDataKind::Uint64:
      return ncclUint64;
    case ReduceDataKind::Float32:
      return ncclFloat32;
    case ReduceDataKind::Float64:
      return ncclFloat64;
    case ReduceDataKind::Float16:
      // ncclHalf is the fp16 dtype; carbon's data_kind<float16> has no specialisation
      // today (no float16 type alias in the public API), but the case is here so users
      // who add their own data_kind<__half> specialisation get a working dispatch.
      return ncclHalf;
    case ReduceDataKind::BFloat16:
#ifdef NCCL_BFLOAT16
      return ncclBfloat16;
#else
      log::throw_with_log<std::runtime_error>(
          "dist_allreduce_in_place: BFloat16 dtype requested but the linked NCCL/RCCL build "
          "does not expose ncclBfloat16; rebuild against NCCL >= 2.10 / RCCL with bf16");
#endif
  }
  log::throw_with_log<std::runtime_error>("dist_allreduce_in_place: unknown ReduceDataKind {}",
                                          static_cast<int>(dt));
}

}  // namespace

auto
dist_allreduce_in_place(const System& sys,
                        void* device_ptr,
                        size_t count,
                        ReduceOpKind op,
                        ReduceDataKind dtype,
                        const Stream& stream) -> void {
  if (count == 0U) {
    return;
  }
  const auto* comm_ptr = sys.communicator();
  if (comm_ptr == nullptr) {
    log::throw_with_log<std::runtime_error>(
        "dist_allreduce_in_place: System has no Communicator (was the Distributed handle "
        "destroyed prematurely?)");
  }
  if (stream.size() != 1U) {
    log::throw_with_log<std::invalid_argument>(
        "dist_allreduce_in_place: Stream must hold exactly one per-rank stream "
        "(carbon's distributed convention is one rank per GPU); got {}",
        stream.size());
  }

  ncclComm_t nccl_comm = detail::communicator_handle(*comm_ptr);
  const ncclRedOp_t nccl_op = to_nccl_op(op);
  const ncclDataType_t nccl_dtype = to_nccl_dtype(dtype);

  // Lambda mirrors src/dist_transfer.cpp::submit_nccl_group: same per-backend stream
  // hand-off so the collective queues onto whichever per-rank stream the caller passed
  // in. ncclAllReduce is in-place when sendbuff == recvbuff, which is exactly the
  // mode the reduce KernelImpl uses (single scalar already living on the root GPU).
  using detail::gpu_compat::gpu_stream_t;
  auto submit_allreduce = [&](gpu_stream_t cs) {
    check_nccl("ncclAllReduce",
               ncclAllReduce(device_ptr, device_ptr, count, nccl_dtype, nccl_op, nccl_comm, cs));
  };

#if CARBON_BACKEND_IS_CUDALIKE
  submit_allreduce(stream[0].get());
#else
  // SYCL backend mirrors dist_transfer.cpp's interop hop: AdaptiveCpp does not expose
  // get_native<sycl::backend::*>(queue) outside a command group, so we wrap the NCCL
  // call inside an AdaptiveCpp_enqueue_custom_operation that runs on the queue's
  // current frontier. The host-side q.wait_and_throw() afterwards keeps the call shape
  // synchronous-looking on the host side; the device-side allreduce remains async.
  sycl::queue& q = stream[0].queue();
  q.submit([&](sycl::handler& cgh) {
    cgh.AdaptiveCpp_enqueue_custom_operation([&submit_allreduce](sycl::interop_handle ih) {
#ifdef CARBON_USE_RCCL
      submit_allreduce(ih.get_native_queue<sycl::backend::hip>());
#else
      submit_allreduce(ih.get_native_queue<sycl::backend::cuda>());
#endif
    });
  });
  q.wait_and_throw();
#endif
}

auto
dist_host_allreduce_in_place(const System& sys,
                             void* host_ptr,
                             size_t count,
                             ReduceOpKind op,
                             ReduceDataKind dtype) -> void {
  if (count == 0U) {
    return;
  }
  const auto* dist_ptr = sys.distributed();
  if (dist_ptr == nullptr) {
    log::throw_with_log<std::runtime_error>(
        "dist_host_allreduce_in_place: System has no Distributed handle (this is the host "
        "MPI fallback used by the OpenMP reduce path; single-process Systems should not "
        "reach here)");
  }

  const MPI_Comm comm = detail::distributed_mpi_world(*dist_ptr);
  const MPI_Op mpi_op = to_mpi_op(op);
  const MPI_Datatype mpi_dt = to_mpi_dtype(dtype);

  // count is size_t but MPI_Allreduce takes int; reject the (extremely unlikely)
  // overflow rather than silently truncate. Single-scalar reductions (count == 1) are
  // the only call site today, but the API is shaped to be reusable for vector reduces.
  if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    log::throw_with_log<std::invalid_argument>(
        "dist_host_allreduce_in_place: count={} exceeds MPI_Allreduce's int element limit", count);
  }
  const int icount = static_cast<int>(count);

  check_mpi("MPI_Allreduce", ::MPI_Allreduce(MPI_IN_PLACE, host_ptr, icount, mpi_dt, mpi_op, comm));
}

#else  // !CARBON_HAVE_MPI_NCCL

auto
dist_allreduce_in_place(const System& sys,
                        void* /*device_ptr*/,
                        size_t /*count*/,
                        ReduceOpKind /*op*/,
                        ReduceDataKind /*dtype*/,
                        const Stream& /*stream*/) -> void {
  if (sys.distributed() != nullptr) {
    // Defensive: a non-null Distributed should be unreachable when carbon is built
    // without CARBON_HAVE_MPI_NCCL (the Distributed class is gated on the same define),
    // but we throw rather than silently no-op so a future refactor that loosens that
    // gate gets a loud failure here instead of a silent loss of cross-rank reduction.
    log::throw_with_log<std::runtime_error>(
        "dist_allreduce_in_place: carbon was built without CARBON_HAVE_MPI_NCCL but the "
        "System carries a Distributed handle; rebuild with CARBON_ENABLE_MPI=ON");
  }
}

auto
dist_host_allreduce_in_place(const System& sys,
                             void* /*host_ptr*/,
                             size_t /*count*/,
                             ReduceOpKind /*op*/,
                             ReduceDataKind /*dtype*/) -> void {
  if (sys.distributed() != nullptr) {
    // Same defensive stance as dist_allreduce_in_place above; the host path also requires
    // MPI to be linked in (via CARBON_HAVE_MPI_NCCL), so a non-null Distributed here
    // means the user managed to construct one without rebuilding carbon for MPI.
    log::throw_with_log<std::runtime_error>(
        "dist_host_allreduce_in_place: carbon was built without CARBON_HAVE_MPI_NCCL but the "
        "System carries a Distributed handle; rebuild with CARBON_ENABLE_MPI=ON");
  }
}

#endif  // CARBON_HAVE_MPI_NCCL

}  // namespace carbon::reduce
