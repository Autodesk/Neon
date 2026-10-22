#pragma once

/**
 * Backend-dispatching shim for @c carbon::detail::atomic_add and @c atomic_min .
 *
 * The actual implementation lives under @c detail/backend/<backend>/atomic.h and is
 * selected at compile time by @c CARBON_BACKEND_IS_CUDALIKE / @c CARBON_BACKEND_IS_SYCL
 * (see @c detail/backend/select.h). User code and detail/ orchestration code @c #include
 * this file; they never include the backend-specific header directly.
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/atomic.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/atomic.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
