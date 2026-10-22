#pragma once

/**
 * Backend-dispatching shim for the block-grid kernel function template (the per-cell
 * device callable concept @c block_kernel_fn and the @c __global__ launcher under
 * CUDA). Symmetric with @c detail/dense/kernel.h .
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/block_kernel.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/block_kernel.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
