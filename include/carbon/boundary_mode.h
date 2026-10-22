#pragma once

#include <cstdint>

namespace carbon {

/**
 * Boundary semantics applied to a @ref Dense / @ref DenseDisg grid (and propagated into
 * every per-slab @c Span).
 *
 * Drives two pieces of behaviour:
 * - @c Partition::neighbour : how out-of-domain neighbour reads are reported. Under
 *   @ref BoundaryMode::Open they are marked invalid; under @ref BoundaryMode::PeriodicZ
 *   the grid wraps and the read is valid from the opposite z face.
 * - Halo exchange: which slab-pair transfers to schedule. Under @ref BoundaryMode::Open
 *   the top-most slab has no upper-halo source and the bottom-most slab has no lower-halo
 *   source; under @ref BoundaryMode::PeriodicZ they wrap around to each other.
 *
 * Both modes are wired end-to-end for @ref Dense (intra-process @c Halo, cross-rank
 * NCCL halo, host @c launch_on_host , and @c Partition::neighbour ) and for
 * @ref DenseDisg ( @c HaloDisg builds the wrap-around adjacency entry; cross-rank
 * NCCL is deferred per the @ref HaloDisg header).
 *
 * @note @ref BoundaryMode is @b not exposed on @ref carbon::Block — sparse grids
 * express boundary semantics through the active-block predicate (cells outside
 * the retained pattern simply have no neighbour) rather than a toroidal wrap.
 */
enum class BoundaryMode : std::uint8_t {
  /**
   * Non-toroidal grid (default). Out-of-domain reads return @c valid = false; the top and
   * bottom slabs in z carry no halo source and the halo exchange leaves their outer halos
   * untouched.
   */
  Open = 0,

  /**
   * Toroidal in z only: the global grid wraps so global @c z = -1 aliases @c z = global.z - 1
   * and global @c z = global.z aliases @c z = 0. The halo exchange pairs the top slab's
   * upper halo with the bottom slab's first owned z (and vice versa);
   * @c Partition::neighbour applies the same wrap so kernels see a contiguous toroidal
   * neighbourhood.
   *
   * Tested by @c test_halo_periodic_z (single-process), @c test_halo_host (OpenMP
   * launch_on_host pass 2), and @c test_halo_dist_periodic_z (MPI + NCCL).
   */
  PeriodicZ = 1,
};

}  // namespace carbon
