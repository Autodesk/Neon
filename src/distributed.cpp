#include <carbon/distributed.h>

#include <carbon/logging.h>

#include "gpu_runtime_compat.h"

#include <mpi.h>

#include <limits>
#include <stdexcept>

namespace carbon {

namespace {

/** Convert an MPI error code to a string and throw std::runtime_error with a Carbon log. */
[[noreturn]] auto
throw_mpi_error(const char* where,
                int err) -> void {
  char buf[MPI_MAX_ERROR_STRING] = {};
  int len = 0;
  if (MPI_Error_string(err, buf, &len) != MPI_SUCCESS) {
    log::throw_with_log<std::runtime_error>("Distributed: {} failed with MPI error {}", where, err);
  }
  log::throw_with_log<std::runtime_error>("Distributed: {} failed: {}", where, buf);
}

inline auto
check_mpi(const char* where,
          int err) -> void {
  if (err != MPI_SUCCESS) {
    throw_mpi_error(where, err);
  }
}

}  // namespace

struct Distributed::Impl {
  bool owned_init{false};
  int world_rank{0};
  int world_size{1};
  int local_rank{0};
  int local_size{1};
  // We own this comm only when owned_init is true on construction below; even when MPI was
  // already initialised by the host, the local-rank communicator is created here and must be
  // freed in the destructor.
  MPI_Comm local_comm{MPI_COMM_NULL};
};

Distributed::Distributed()
    : impl_(std::make_unique<Impl>()) {
  int already = 0;
  check_mpi("MPI_Initialized", MPI_Initialized(&already));
  if (!already) {
    int provided = 0;
    check_mpi("MPI_Init_thread",
              MPI_Init_thread(nullptr, nullptr, MPI_THREAD_SERIALIZED, &provided));
    if (provided < MPI_THREAD_SERIALIZED) {
      CARBON_LOG_WARN(
          "Distributed: MPI provided thread level {} below requested SERIALIZED ({}); "
          "concurrent MPI use from multiple host threads is unsupported.",
          provided, static_cast<int>(MPI_THREAD_SERIALIZED));
    }
    impl_->owned_init = true;
  }

  check_mpi("MPI_Comm_rank", MPI_Comm_rank(MPI_COMM_WORLD, &impl_->world_rank));
  check_mpi("MPI_Comm_size", MPI_Comm_size(MPI_COMM_WORLD, &impl_->world_size));

  check_mpi("MPI_Comm_split_type",
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, impl_->world_rank,
                                MPI_INFO_NULL, &impl_->local_comm));
  check_mpi("MPI_Comm_rank(local)", MPI_Comm_rank(impl_->local_comm, &impl_->local_rank));
  check_mpi("MPI_Comm_size(local)", MPI_Comm_size(impl_->local_comm, &impl_->local_size));

  CARBON_LOG_INFO("Distributed: world_rank={} world_size={} local_rank={} local_size={}",
                  impl_->world_rank, impl_->world_size, impl_->local_rank, impl_->local_size);
}

Distributed::~Distributed() {
  if (!impl_) {
    return;
  }
  if (impl_->local_comm != MPI_COMM_NULL) {
    int finalized = 0;
    if (MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized) {
      MPI_Comm_free(&impl_->local_comm);
    }
  }
  if (impl_->owned_init) {
    int finalized = 0;
    if (MPI_Finalized(&finalized) == MPI_SUCCESS && !finalized) {
      MPI_Finalize();
    }
  }
}

auto
Distributed::world_rank() const noexcept -> int {
  return impl_->world_rank;
}

auto
Distributed::world_size() const noexcept -> int {
  return impl_->world_size;
}

auto
Distributed::local_rank() const noexcept -> int {
  return impl_->local_rank;
}

auto
Distributed::local_size() const noexcept -> int {
  return impl_->local_size;
}

auto
Distributed::owns_mpi() const noexcept -> bool {
  return impl_->owned_init;
}

void
Distributed::barrier() const {
  check_mpi("MPI_Barrier", MPI_Barrier(MPI_COMM_WORLD));
}

void
Distributed::broadcast(void* data,
                       size_t bytes,
                       int root) const {
  if (bytes == 0U) {
    return;
  }
  if (data == nullptr) {
    log::throw_with_log<std::invalid_argument>(
        "Distributed::broadcast: data must be non-null when bytes > 0");
  }
  if (root < 0 || root >= impl_->world_size) {
    log::throw_with_log<std::invalid_argument>(
        "Distributed::broadcast: root {} out of range for world_size {}", root, impl_->world_size);
  }
  int count = 0;
  if (bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
    log::throw_with_log<std::invalid_argument>(
        "Distributed::broadcast: bytes {} exceeds MPI_Bcast int limit", bytes);
  }
  count = static_cast<int>(bytes);
  check_mpi("MPI_Bcast", MPI_Bcast(data, count, MPI_BYTE, root, MPI_COMM_WORLD));
}

auto
Distributed::pin_local_device() const -> int {
  using namespace carbon::detail::gpu_compat;
  int visible = 0;
  gpu_error_t err = gpu_get_device_count(&visible);
  if (err != gpu_success) {
    log::throw_with_log<std::runtime_error>(
        "Distributed::pin_local_device: gpu_get_device_count failed: {} ({})",
        static_cast<int>(err), gpu_get_error_string(err));
  }
  if (visible <= 0) {
    log::throw_with_log<std::runtime_error>(
        "Distributed::pin_local_device: no visible GPU devices on rank {}", impl_->world_rank);
  }
  int chosen = 0;
  if (visible == 1) {
    chosen = 0;
  } else {
    chosen = impl_->local_rank % visible;
    CARBON_LOG_WARN(
        "Distributed::pin_local_device: rank {} sees {} visible devices; falling back to "
        "device {} = local_rank {} %% {}. Prefer launcher-side pinning "
        "(srun --gpus-per-task=1 or mpirun --map-by ppr:N:node:PE=...).",
        impl_->world_rank, visible, chosen, impl_->local_rank, visible);
  }
  err = gpu_set_device(chosen);
  if (err != gpu_success) {
    log::throw_with_log<std::runtime_error>(
        "Distributed::pin_local_device: gpu_set_device({}) failed: {} ({})", chosen,
        static_cast<int>(err), gpu_get_error_string(err));
  }
  return chosen;
}

}  // namespace carbon

#include "distributed_internal.h"

namespace carbon::detail {

auto
distributed_mpi_world(const Distributed& /*d*/) noexcept -> MPI_Comm {
  return MPI_COMM_WORLD;
}

}  // namespace carbon::detail
