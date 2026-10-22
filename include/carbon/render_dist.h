#pragma once

#include <carbon/export.h>
#include <carbon/system.h>

#include <cstddef>

namespace carbon::render {

/**
 * MPI_Allgather of per-pixel ray-march segment data from every rank into @p all_* buffers laid
 * out rank-major: index @c r * pixel_count + i is rank @c r pixel @c i .
 *
 * Requires @c System::distributed() and a carbon build with @c CARBON_HAVE_MPI_NCCL .
 * No-op when @p pixel_count is zero.
 */
CARBON_API auto
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
                     size_t pixel_count) -> void;

}  // namespace carbon::render
