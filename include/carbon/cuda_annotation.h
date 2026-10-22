#pragma once

/** Marks functions (and similar) for both host and device when built with a cuda-like device compiler (nvcc / HIP clang); empty on host-only TUs. */
#if defined(__CUDACC__) || defined(__HIPCC__)
#define CARBON_HOST_DEVICE __host__ __device__
#else
#define CARBON_HOST_DEVICE
#endif

/** Legacy alias; same expansion as @ref CARBON_HOST_DEVICE. */
#define CARBON_HD CARBON_HOST_DEVICE
