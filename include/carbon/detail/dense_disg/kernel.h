#pragma once

/**
 * Backend-dispatching shim for the disaggregated-dense kernel function template.
 * Mirrors @c carbon/detail/dense/kernel.h but for the disg backend headers.
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/dense_disg_kernel.h>
#elif CARBON_BACKEND_IS_SYCL
// Concept lives in the SYCL launch header (no separate kernel TU for SYCL).
#include <carbon/detail/backend/sycl/dense_disg_launch.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
