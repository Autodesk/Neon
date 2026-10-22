#pragma once

/**
 * Q-criterion scalar field from a vector velocity field on Dense / DenseDisg grids.
 *
 * @ref make_qcriterion_kernel builds a compute @ref Kernel that fills a single-component
 * field with @c Q = ½(‖Ω‖² − ‖S‖²) , where @c S and @c Ω are the symmetric and
 * antisymmetric parts of the velocity-gradient tensor @c J_ij = ∂u_i/∂x_j computed by
 * central differences. Cells whose stencil neighbours are unavailable write @c 0 .
 */

#include <carbon/cuda_annotation.h>
#include <carbon/kernel.h>

namespace carbon {

/**
 * Build a compute @ref Kernel that writes the Q-criterion from @p velocity into @p q_out .
 *
 * @p velocity must have exactly three components (@c u_x , @c u_y , @c u_z ). @p q_out
 * receives one scalar per cell. @p velocity_scale multiplies each velocity component
 * before differencing (e.g. @c 1/ulb for dimensionless @c Q in @c (1/length)² units).
 */
template <typename Grid,
          typename T = float,
          int VelocityComponents = 3>
[[nodiscard]] auto
make_qcriterion_kernel(const Grid& grid,
                       typename Grid::template Field<T,
                                                     VelocityComponents>& velocity,
                       typename Grid::template Field<T,
                                                     1>& q_out,
                       T velocity_scale = T{1}) -> Kernel;

}  // namespace carbon

#include <carbon/detail/qcriterion.inl>
