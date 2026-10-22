#pragma once

/**
 * Backend dispatch for pixel-parallel render kernels. Selected at compile time via
 * @c CARBON_BACKEND_IS_* ; only @c detail::RayMarchKernelImpl reaches in here.
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/render_launch.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/render_launch.h>
#endif
