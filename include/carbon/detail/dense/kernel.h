#pragma once

/**
 * Backend-dispatching shim for the dense-grid kernel function template (the per-cell
 * device callable concept @c dense_kernel_fn and the @c __global__ launcher under CUDA).
 * The SYCL counterpart will provide the same concept and a SYCL-friendly callable
 * structure (no @c __global__) used by @c q.parallel_for at the launch site.
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/dense_kernel.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/dense_kernel.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
