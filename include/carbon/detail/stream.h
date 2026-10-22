#pragma once

/**
 * Backend-dispatching shim for the per-GPU stream wrapper @c carbon::detail::CudaStream.
 *
 * The actual implementation lives under @c detail/backend/<backend>/stream.h and is selected
 * at compile time by @c CARBON_BACKEND_IS_CUDA / @c CARBON_BACKEND_IS_SYCL (see
 * @c detail/backend/select.h). User code and detail/ orchestration code @c #include this
 * file (the public path); they never include the backend-specific header directly.
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/stream.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/stream.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
