#pragma once

#include <carbon/detail/dense_disg/idx.h>

namespace carbon::detail::dense_disg_distributed {

/**
 * Cell index shared with the Z-slab @c DenseDisg family: buffer-relative @c (x, y, z) plus the
 * three precomputed region fields ( @c offset_local_no_card , @c region_first_z ,
 * @c region_z_dim ) the disaggregated pitch consumes. On slab storage the fields mean exactly
 * what they mean for @c DenseDisg (counts of xy planes); on pencil storage the two region
 * fields count x rows instead (see @c Span::storage_unit ).
 */
using Idx = ::carbon::detail::dense_disg::Idx;

}  // namespace carbon::detail::dense_disg_distributed
