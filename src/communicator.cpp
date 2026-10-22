#include <carbon/detail/communicator.h>

#include <carbon/distributed.h>
#include <carbon/logging.h>

#include "communicator_internal.h"
#include "distributed_internal.h"
#include "gpu_runtime_compat.h"

#include <mpi.h>

#include <stdexcept>

namespace carbon::detail {

namespace {

[[noreturn]] auto
throw_nccl_error(const char* where,
                 ncclResult_t res) -> void {
  log::throw_with_log<std::runtime_error>("Communicator: {} failed: {}", where,
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
    log::throw_with_log<std::runtime_error>("Communicator: {} failed with MPI error {}", where,
                                            err);
  }
  log::throw_with_log<std::runtime_error>("Communicator: {} failed: {}", where, buf);
}

inline auto
check_mpi(const char* where,
          int err) -> void {
  if (err != MPI_SUCCESS) {
    throw_mpi_error(where, err);
  }
}

}  // namespace

struct Communicator::Impl {
  ncclComm_t comm{nullptr};
  int rank{0};
  int size{1};
  int device{0};
};

Communicator::Communicator()
    : impl_(std::make_unique<Impl>()) {}

Communicator::~Communicator() {
  if (impl_ && impl_->comm != nullptr) {
    ncclResult_t res = ncclCommDestroy(impl_->comm);
    if (res != ncclSuccess) {
      CARBON_LOG_ERROR("Communicator: ncclCommDestroy failed: {}", ncclGetErrorString(res));
    }
  }
}

auto
Communicator::from_mpi(const Distributed& distributed) -> std::shared_ptr<Communicator> {
  auto comm = std::shared_ptr<Communicator>(new Communicator());
  comm->impl_->rank = distributed.world_rank();
  comm->impl_->size = distributed.world_size();

  using namespace carbon::detail::gpu_compat;
  gpu_error_t cerr = gpu_get_device(&comm->impl_->device);
  if (cerr != gpu_success) {
    log::throw_with_log<std::runtime_error>(
        "Communicator::from_mpi: gpu_get_device failed: {} ({})", static_cast<int>(cerr),
        gpu_get_error_string(cerr));
  }

  ncclUniqueId id{};
  if (distributed.world_rank() == 0) {
    check_nccl("ncclGetUniqueId", ncclGetUniqueId(&id));
  }

  MPI_Comm mpi_comm = distributed_mpi_world(distributed);

  check_mpi("MPI_Bcast(ncclUniqueId)", MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm));
  // Required so late ranks finish receiving the id before any rank starts the collective
  // ncclCommInitRank; matches the load-bearing barrier in Neon-nccl's Backend.cpp::initNCCL.
  check_mpi("MPI_Barrier(after Bcast)", MPI_Barrier(mpi_comm));

  check_nccl("ncclCommInitRank", ncclCommInitRank(&comm->impl_->comm, distributed.world_size(), id,
                                                  distributed.world_rank()));

  CARBON_LOG_INFO("Communicator: rank={}/{} bound to device {}", comm->impl_->rank,
                  comm->impl_->size, comm->impl_->device);
  return comm;
}

auto
Communicator::rank() const noexcept -> int {
  return impl_->rank;
}

auto
Communicator::size() const noexcept -> int {
  return impl_->size;
}

auto
Communicator::device() const noexcept -> int {
  return impl_->device;
}

auto
Communicator::raw_handle() const noexcept -> void* {
  return reinterpret_cast<void*>(impl_->comm);
}

}  // namespace carbon::detail
