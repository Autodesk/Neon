#pragma once

/**
 * Backend-dispatching shim for @c carbon::detail::Gpu — the per-device handle that owns
 * a @c cudaStream_t / @c cudaEvent_t pool, exposes typed_launch, and reports per-device
 * properties. The SYCL counterpart will wrap @c sycl::device + @c sycl::context.
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/gpu.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/gpu.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
