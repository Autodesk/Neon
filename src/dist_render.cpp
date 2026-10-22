#include <carbon/render_dist.h>

#include <carbon/distributed.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#ifdef CARBON_HAVE_MPI_NCCL
#include "distributed_internal.h"

#include <mpi.h>
#endif

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace carbon::render {

#ifdef CARBON_HAVE_MPI_NCCL

namespace {

[[noreturn]] auto
throw_mpi_error(const char* where,
                int err) -> void {
  char msg[MPI_MAX_ERROR_STRING]{};
  int len = 0;
  (void)MPI_Error_string(err, msg, &len);
  log::throw_with_log<std::runtime_error>("dist_gather_segments: {} failed: {}", where, msg);
}

inline auto
check_mpi(const char* where,
          int err) -> void {
  if (err != MPI_SUCCESS) {
    throw_mpi_error(where, err);
  }
}

}  // namespace

auto
dist_gather_segments(const System& sys,
                     const float* local_color_r,
                     const float* local_color_g,
                     const float* local_color_b,
                     const float* local_trans,
                     const float* local_depth,
                     float* all_color_r,
                     float* all_color_g,
                     float* all_color_b,
                     float* all_trans,
                     float* all_depth,
                     size_t pixel_count) -> void {
  if (pixel_count == 0U) {
    return;
  }
  const auto* dist_ptr = sys.distributed();
  if (dist_ptr == nullptr) {
    log::throw_with_log<std::runtime_error>(
        "dist_gather_segments: System has no Distributed handle");
  }

  if (pixel_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    log::throw_with_log<std::invalid_argument>(
        "dist_gather_segments: pixel_count={} exceeds MPI's int element limit", pixel_count);
  }
  const int icount = static_cast<int>(pixel_count);
  const MPI_Comm comm = detail::distributed_mpi_world(*dist_ptr);

  check_mpi("MPI_Allgather(color_r)", ::MPI_Allgather(local_color_r, icount, MPI_FLOAT, all_color_r,
                                                      icount, MPI_FLOAT, comm));
  check_mpi("MPI_Allgather(color_g)", ::MPI_Allgather(local_color_g, icount, MPI_FLOAT, all_color_g,
                                                      icount, MPI_FLOAT, comm));
  check_mpi("MPI_Allgather(color_b)", ::MPI_Allgather(local_color_b, icount, MPI_FLOAT, all_color_b,
                                                      icount, MPI_FLOAT, comm));
  check_mpi("MPI_Allgather(trans)",
            ::MPI_Allgather(local_trans, icount, MPI_FLOAT, all_trans, icount, MPI_FLOAT, comm));
  check_mpi("MPI_Allgather(t_enter)",
            ::MPI_Allgather(local_depth, icount, MPI_FLOAT, all_depth, icount, MPI_FLOAT, comm));
}

#else

auto
dist_gather_segments(const System& sys,
                     const float* /*local_color_r*/,
                     const float* /*local_color_g*/,
                     const float* /*local_color_b*/,
                     const float* /*local_trans*/,
                     const float* /*local_depth*/,
                     float* /*all_color_r*/,
                     float* /*all_color_g*/,
                     float* /*all_color_b*/,
                     float* /*all_trans*/,
                     float* /*all_depth*/,
                     size_t /*pixel_count*/) -> void {
  if (sys.distributed() != nullptr) {
    log::throw_with_log<std::runtime_error>(
        "dist_gather_segments: carbon was built without CARBON_HAVE_MPI_NCCL but the "
        "System carries a Distributed handle; rebuild with CARBON_ENABLE_MPI=ON");
  }
}

#endif  // CARBON_HAVE_MPI_NCCL

}  // namespace carbon::render
