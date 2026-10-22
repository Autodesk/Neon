#pragma once

/**
 * Backend-dispatching shim for the block-grid kernel launch helper (see
 * @c carbon::detail::block::launch_block_grid ). Symmetric with
 * @c detail/dense/launch.h .
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/block_launch.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/block_launch.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
