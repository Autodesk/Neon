#pragma once

// Backend-dispatching shim for the inline definitions belonging to detail::Gpu (see
// detail/gpu.h). Same dispatch model as the rest of detail/.

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/gpu.inl>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/gpu.inl>
#endif
