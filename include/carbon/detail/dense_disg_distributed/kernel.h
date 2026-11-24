#pragma once

/**
 * Backend-dispatching shim for the @c DenseDisgDistributed kernel function template.
 * Mirrors @c carbon/detail/dense_disg/kernel.h .
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/dense_disg_distributed_kernel.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/dense_disg_distributed_launch.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
