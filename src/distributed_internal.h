#pragma once

// Bridge header used only inside src/. Lets translation units that already see <mpi.h> recover
// the MPI_Comm associated with a carbon::Distributed without dragging MPI types through any
// public header. Not installed.

#include <carbon/distributed.h>

#include <mpi.h>

namespace carbon::detail {

/**
 * Returns the @c MPI_Comm associated with @p d (currently always @c MPI_COMM_WORLD; the
 * indirection leaves room for a future @c MPI_Comm_dup or split-comm handle without changing
 * the friend boundary).
 */
[[nodiscard]] auto
distributed_mpi_world(const Distributed& d) noexcept -> MPI_Comm;

}  // namespace carbon::detail
