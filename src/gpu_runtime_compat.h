#pragma once

// Private compatibility shim used by carbon's MPI/distributed sources to keep one body
// of code working against either NCCL+CUDA or RCCL+HIP. Only included from .cpp files
// under src/; never installed and never seen by carbon's public API.
//
// Selection is made at cmake time:
//   - Default: NCCL + CUDA (covers CARBON_BACKEND=Cuda + CARBON_BACKEND=Sycl with cuda: target).
//   - When CARBON_USE_RCCL is defined (auto-set by CMakeLists.txt when CARBON_BACKEND=Sycl
//     and CARBON_SYCL_TARGETS contains a hip: entry): RCCL + HIP. RCCL is ABI-compatible with
//     NCCL — same ncclSend / ncclRecv / ncclComm_t / ncclResult_t / etc. — so the bulk of
//     each file stays unchanged; only the runtime API names (cudaXxx vs hipXxx), the
//     stream typedef, and one SYCL-interop backend tag flip.

#include <carbon/detail/backend/select.h>

#ifdef CARBON_DIST_HOST_MPI
// Host-MPI build (CARBON_BACKEND=Sycl, omp-only target): no vendor runtime, no NCCL/RCCL.
// "Device" memory is ordinary host RAM and the single compute device is the OpenMP host.
#include <cstddef>
#include <cstring>
#elif defined(CARBON_USE_RCCL)
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#else
#include <cuda_runtime.h>
#include <nccl.h>
#endif

namespace carbon::detail::gpu_compat {

#ifdef CARBON_DIST_HOST_MPI
using gpu_stream_t = void*;
using gpu_error_t = int;
inline constexpr gpu_error_t gpu_success = 0;

inline auto
gpu_get_device(int* dev) noexcept -> gpu_error_t {
  *dev = 0;
  return gpu_success;
}
inline auto
gpu_set_device(int /*dev*/) noexcept -> gpu_error_t {
  return gpu_success;
}
inline auto
gpu_get_device_count(int* n) noexcept -> gpu_error_t {
  *n = 1;
  return gpu_success;
}
inline auto
gpu_get_error_string(gpu_error_t /*e*/) noexcept -> const char* {
  return "host-mpi (no vendor runtime)";
}
inline auto
gpu_memcpy(void* dst,
           const void* src,
           size_t count,
           int /*kind*/) noexcept -> gpu_error_t {
  std::memcpy(dst, src, count);
  return gpu_success;
}
inline constexpr int gpu_memcpy_host_to_device = 0;
inline constexpr int gpu_memcpy_device_to_host = 1;
#elif defined(CARBON_USE_RCCL)
using gpu_stream_t = hipStream_t;
using gpu_error_t = hipError_t;
inline constexpr gpu_error_t gpu_success = hipSuccess;

inline auto
gpu_get_device(int* dev) noexcept -> gpu_error_t {
  return hipGetDevice(dev);
}
inline auto
gpu_set_device(int dev) noexcept -> gpu_error_t {
  return hipSetDevice(dev);
}
inline auto
gpu_get_device_count(int* n) noexcept -> gpu_error_t {
  return hipGetDeviceCount(n);
}
inline auto
gpu_get_error_string(gpu_error_t e) noexcept -> const char* {
  return hipGetErrorString(e);
}
inline auto
gpu_memcpy(void* dst,
           const void* src,
           size_t count,
           int kind) noexcept -> gpu_error_t {
  return hipMemcpy(dst, src, count, static_cast<hipMemcpyKind>(kind));
}
inline constexpr int gpu_memcpy_host_to_device = hipMemcpyHostToDevice;
inline constexpr int gpu_memcpy_device_to_host = hipMemcpyDeviceToHost;
#else
using gpu_stream_t = cudaStream_t;
using gpu_error_t = cudaError_t;
inline constexpr gpu_error_t gpu_success = cudaSuccess;

inline auto
gpu_get_device(int* dev) noexcept -> gpu_error_t {
  return cudaGetDevice(dev);
}
inline auto
gpu_set_device(int dev) noexcept -> gpu_error_t {
  return cudaSetDevice(dev);
}
inline auto
gpu_get_device_count(int* n) noexcept -> gpu_error_t {
  return cudaGetDeviceCount(n);
}
inline auto
gpu_get_error_string(gpu_error_t e) noexcept -> const char* {
  return cudaGetErrorString(e);
}
inline auto
gpu_memcpy(void* dst,
           const void* src,
           size_t count,
           int kind) noexcept -> gpu_error_t {
  return cudaMemcpy(dst, src, count, static_cast<cudaMemcpyKind>(kind));
}
inline constexpr int gpu_memcpy_host_to_device = cudaMemcpyHostToDevice;
inline constexpr int gpu_memcpy_device_to_host = cudaMemcpyDeviceToHost;
#endif

}  // namespace carbon::detail::gpu_compat
