#pragma once

/**
 * Backend-dispatching shim for the per-GPU event wrapper. See @c detail/stream.h for the
 * dispatch model and the SYCL roadmap.
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/event.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/event.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
