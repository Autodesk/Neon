#pragma once

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_CUDALIKE
#include <carbon/detail/backend/cudalike/dense_disg_launch.h>
#elif CARBON_BACKEND_IS_SYCL
#include <carbon/detail/backend/sycl/dense_disg_launch.h>
#else
#error "carbon: no compute backend selected (see carbon/detail/backend/select.h)."
#endif
