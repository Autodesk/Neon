#pragma once

/**
 * Backend-dispatching shim for the dense-grid kernel launch helper (see
 * @c carbon::detail::dense::launch_dense_grid). Hides the launch-syntax difference
 * between @c <<<...>>> (CUDA) and @c q.parallel_for (SYCL).
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/dense_launch.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/dense_launch.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
