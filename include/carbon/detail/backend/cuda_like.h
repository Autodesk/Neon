#pragma once

/**
 * Backend-neutral runtime aliases shared by the CUDA and HIP backends.
 *
 * CUDA and HIP expose a near-identical runtime surface: every @c cudaXxx call has a
 * @c hipXxx twin with the same signature, @c dim3 / @c __global__ / @c <<<>>> are spelled
 * the same, and @c cub maps to @c hipcub one-to-one. Rather than fork the backend tree into
 * a cuda and a hip copy, both cuda-like backends share a @b single implementation tree
 * (@c detail/backend/cudalike/), selected by @ref CARBON_BACKEND_IS_CUDALIKE, that talks to
 * the runtime exclusively through the aliases defined here:
 *
 *  - @c carbon::detail::gpu_stream_t        — @c cudaStream_t / @c hipStream_t.
 *  - @c carbon::detail::gpu_error_t         — @c cudaError_t / @c hipError_t.
 *  - @c carbon::detail::gpu_success         — @c cudaSuccess / @c hipSuccess.
 *  - @c carbon::detail::gpu_error_string(e) — @c cudaGetErrorString / @c hipGetErrorString.
 *  - @c carbon::detail::gpu_set_device(d)   — @c cudaSetDevice / @c hipSetDevice.
 *  - @c carbon::detail::gpu_event_t         — @c cudaEvent_t / @c hipEvent_t.
 *  - @c carbon::detail::gpu_error_not_supported / gpu_error_invalid_value /
 *    gpu_error_invalid_resource_handle — the @c cudaErrorXxx / @c hipErrorXxx codes carbon
 *    returns from its unsupported-path stubs and resource-handle diagnostics.
 *  - @c carbon::detail::gpu_event_disable_timing — the timing-disable event flag.
 *  - @c carbon::detail::gpu_get_last_error / gpu_malloc / gpu_free / gpu_stream_create /
 *    gpu_stream_destroy / gpu_event_create{,_with_flags} / gpu_event_destroy /
 *    gpu_event_record / gpu_stream_wait_event / gpu_runtime_launch_kernel — thin inline
 *    wrappers over the matching @c cudaXxx / @c hipXxx runtime call. (The launch wrapper is
 *    named @c gpu_runtime_launch_kernel — not @c gpu_launch_kernel — to avoid colliding with
 *    the file-local @c gpu_launch_kernel marshalling helper in @c cudalike/gpu.inl.)
 *  - @c carbon::detail::gpucub               — namespace alias for @c cub / @c hipcub
 *                                             (only declared inside a cuda-like device TU,
 *                                             since cub/hipcub require the device compiler).
 *
 * The runtime header (@c <cuda_runtime.h> / @c <hip/hip_runtime.h>) is includable from host
 * translation units too — the same way @c detail/backend/cudalike/gpu.h already pulls it in —
 * so the type aliases are available regardless of whether the TU is a device TU. Only the
 * @c gpucub alias and the cub/hipcub headers are gated on @ref CARBON_CUDALIKE_DEVICE_TU.
 *
 * Under the SYCL backend this header expands to nothing: the shared headers only reference
 * these aliases inside @c CARBON_BACKEND_IS_CUDALIKE arms, which are dead under SYCL.
 *
 * @par One tree, two backends
 * The @c detail/backend/cudalike/ headers spell @b only these neutral @c gpu_* aliases and
 * the @c gpucub namespace, never a raw @c cudaXxx / @c hipXxx name. That is what lets a
 * single source tree serve both the CUDA backend (where the aliases resolve to @c cudaXxx +
 * @c cub) and the HIP backend (@c hipXxx + @c hipcub, including the HIP-on-NVIDIA platform).
 * Launch syntax (@c <<<>>>), @c __global__, @c __shared__ and @c dim3 are spelled identically
 * by both toolchains, so they appear verbatim in the tree.
 */

#include <carbon/detail/backend/select.h>

#if CARBON_BACKEND_IS_HIP
#include <hip/hip_runtime.h>
#elif CARBON_BACKEND_IS_CUDA
#include <cuda_runtime.h>
#endif

#if CARBON_BACKEND_IS_CUDALIKE && CARBON_CUDALIKE_DEVICE_TU
#if CARBON_BACKEND_IS_HIP
#include <hipcub/hipcub.hpp>
#else
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>
#endif
#endif

#if CARBON_BACKEND_IS_CUDALIKE

namespace carbon {
namespace detail {

#if CARBON_BACKEND_IS_HIP
using gpu_stream_t = hipStream_t;
using gpu_error_t = hipError_t;
using gpu_event_t = hipEvent_t;
inline constexpr gpu_error_t gpu_success = hipSuccess;
inline constexpr gpu_error_t gpu_error_not_supported = hipErrorNotSupported;
inline constexpr gpu_error_t gpu_error_invalid_value = hipErrorInvalidValue;
// HIP's canonical spelling is hipErrorInvalidHandle (hipErrorInvalidResourceHandle is a
// deprecated alias); under __HIP_PLATFORM_NVIDIA__ it forwards to cudaErrorInvalidResourceHandle.
inline constexpr gpu_error_t gpu_error_invalid_resource_handle = hipErrorInvalidHandle;
inline constexpr unsigned int gpu_event_disable_timing = hipEventDisableTiming;
#else
using gpu_stream_t = cudaStream_t;
using gpu_error_t = cudaError_t;
using gpu_event_t = cudaEvent_t;
inline constexpr gpu_error_t gpu_success = cudaSuccess;
inline constexpr gpu_error_t gpu_error_not_supported = cudaErrorNotSupported;
inline constexpr gpu_error_t gpu_error_invalid_value = cudaErrorInvalidValue;
inline constexpr gpu_error_t gpu_error_invalid_resource_handle = cudaErrorInvalidResourceHandle;
inline constexpr unsigned int gpu_event_disable_timing = cudaEventDisableTiming;
#endif

[[nodiscard]] inline auto
gpu_error_string(gpu_error_t e) noexcept -> const char* {
#if CARBON_BACKEND_IS_HIP
  return hipGetErrorString(e);
#else
  return cudaGetErrorString(e);
#endif
}

[[nodiscard]] inline auto
gpu_set_device(int device_id) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipSetDevice(device_id);
#else
  return cudaSetDevice(device_id);
#endif
}

[[nodiscard]] inline auto
gpu_get_device(int* device_id) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipGetDevice(device_id);
#else
  return cudaGetDevice(device_id);
#endif
}

[[nodiscard]] inline auto
gpu_get_last_error() noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipGetLastError();
#else
  return cudaGetLastError();
#endif
}

[[nodiscard]] inline auto
gpu_malloc(void** ptr,
           size_t size_bytes) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipMalloc(ptr, size_bytes);
#else
  return cudaMalloc(ptr, size_bytes);
#endif
}

// Returns are not [[nodiscard]] on the teardown wrappers below: several call sites discard
// the status on a best-effort destroy/free path (mirroring the original cuda* usage).
inline auto
gpu_free(void* ptr) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipFree(ptr);
#else
  return cudaFree(ptr);
#endif
}

#if CARBON_BACKEND_IS_HIP
inline constexpr unsigned int gpu_host_alloc_default = hipHostAllocDefault;
#else
inline constexpr unsigned int gpu_host_alloc_default = cudaHostAllocDefault;
#endif

inline auto
gpu_host_alloc(void** ptr,
               size_t size_bytes,
               unsigned int flags) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipHostMalloc(ptr, size_bytes, flags);
#else
  return cudaHostAlloc(ptr, size_bytes, flags);
#endif
}

inline auto
gpu_host_free(void* ptr) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipHostFree(ptr);
#else
  return cudaFreeHost(ptr);
#endif
}

inline auto
gpu_stream_create(gpu_stream_t* stream) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipStreamCreate(stream);
#else
  return cudaStreamCreate(stream);
#endif
}

inline auto
gpu_stream_create_with_priority(gpu_stream_t* stream,
                                int least_priority,
                                int greatest_priority) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  (void)least_priority;
  return hipStreamCreateWithPriority(stream, hipStreamDefault, greatest_priority);
#else
  (void)least_priority;
  return cudaStreamCreateWithPriority(stream, cudaStreamDefault, greatest_priority);
#endif
}

inline auto
gpu_stream_priority_range(int* least_priority,
                          int* greatest_priority) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipDeviceGetStreamPriorityRange(least_priority, greatest_priority);
#else
  return cudaDeviceGetStreamPriorityRange(least_priority, greatest_priority);
#endif
}

inline auto
gpu_stream_destroy(gpu_stream_t stream) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipStreamDestroy(stream);
#else
  return cudaStreamDestroy(stream);
#endif
}

inline auto
gpu_event_create(gpu_event_t* event) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipEventCreate(event);
#else
  return cudaEventCreate(event);
#endif
}

inline auto
gpu_event_create_with_flags(gpu_event_t* event,
                            unsigned int flags) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipEventCreateWithFlags(event, flags);
#else
  return cudaEventCreateWithFlags(event, flags);
#endif
}

inline auto
gpu_event_destroy(gpu_event_t event) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipEventDestroy(event);
#else
  return cudaEventDestroy(event);
#endif
}

inline auto
gpu_event_record(gpu_event_t event,
                 gpu_stream_t stream) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipEventRecord(event, stream);
#else
  return cudaEventRecord(event, stream);
#endif
}

inline auto
gpu_stream_wait_event(gpu_stream_t stream,
                      gpu_event_t event,
                      unsigned int flags) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipStreamWaitEvent(stream, event, flags);
#else
  return cudaStreamWaitEvent(stream, event, flags);
#endif
}

inline auto
gpu_event_synchronize(gpu_event_t event) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipEventSynchronize(event);
#else
  return cudaEventSynchronize(event);
#endif
}

inline auto
gpu_event_query(gpu_event_t event) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipEventQuery(event);
#else
  return cudaEventQuery(event);
#endif
}

inline auto
gpu_runtime_launch_kernel(const void* func,
                          dim3 grid,
                          dim3 block,
                          void** args,
                          size_t shared_mem,
                          gpu_stream_t stream) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipLaunchKernel(func, grid, block, args, shared_mem, stream);
#else
  return cudaLaunchKernel(func, grid, block, args, shared_mem, stream);
#endif
}

// ---------------------------------------------------------------------------
// Synchronisation + blocking-copy surface. Used by host .cpp files that need a
// self-contained "do this now and tell me the result" sequence outside the
// Stream / Buffer plumbing — notably the peer-access validation probe in
// detail/backend/cudalike/peer_access.h, which has to run during GpuGroup
// construction before any of that plumbing exists.
// ---------------------------------------------------------------------------

inline auto
gpu_device_synchronize() noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipDeviceSynchronize();
#else
  return cudaDeviceSynchronize();
#endif
}

inline auto
gpu_stream_synchronize(gpu_stream_t stream) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipStreamSynchronize(stream);
#else
  return cudaStreamSynchronize(stream);
#endif
}

/** Blocking copy with the direction inferred from the pointers (unified addressing). */
inline auto
gpu_memcpy_default(void* dst,
                   const void* src,
                   size_t size_bytes) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipMemcpy(dst, src, size_bytes, hipMemcpyDefault);
#else
  return cudaMemcpy(dst, src, size_bytes, cudaMemcpyDefault);
#endif
}

/** Stream-ordered copy with the direction inferred from the pointers. */
inline auto
gpu_memcpy_async_default(void* dst,
                         const void* src,
                         size_t size_bytes,
                         gpu_stream_t stream) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipMemcpyAsync(dst, src, size_bytes, hipMemcpyDefault, stream);
#else
  return cudaMemcpyAsync(dst, src, size_bytes, cudaMemcpyDefault, stream);
#endif
}

inline auto
gpu_memset(void* dst,
           int value,
           size_t size_bytes) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipMemset(dst, value, size_bytes);
#else
  return cudaMemset(dst, value, size_bytes);
#endif
}

// ---------------------------------------------------------------------------
// Peer-access surface. Carbon treats every answer these give as a *claim* to be
// validated rather than a fact — see detail/backend/cudalike/peer_access.h for
// why (a platform can report capability it does not have, and the resulting
// peer DMA then silently transfers nothing).
// ---------------------------------------------------------------------------

#if CARBON_BACKEND_IS_HIP
inline constexpr gpu_error_t gpu_error_peer_access_already_enabled =
    hipErrorPeerAccessAlreadyEnabled;
#else
inline constexpr gpu_error_t gpu_error_peer_access_already_enabled =
    cudaErrorPeerAccessAlreadyEnabled;
#endif

inline auto
gpu_device_can_access_peer(int* can_access,
                           int device_id,
                           int peer_id) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipDeviceCanAccessPeer(can_access, device_id, peer_id);
#else
  return cudaDeviceCanAccessPeer(can_access, device_id, peer_id);
#endif
}

/** Enables the @b current context to reach @p peer_id 's memory. */
inline auto
gpu_device_enable_peer_access(int peer_id) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipDeviceEnablePeerAccess(peer_id, 0);
#else
  return cudaDeviceEnablePeerAccess(peer_id, 0);
#endif
}

inline auto
gpu_device_disable_peer_access(int peer_id) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipDeviceDisablePeerAccess(peer_id);
#else
  return cudaDeviceDisablePeerAccess(peer_id);
#endif
}

/**
 * Does the @p src_id -> @p dst_id link support native atomics? Wrapped rather than
 * exposing the attribute enum because the two runtimes spell both the enum type and the
 * enumerator differently.
 */
inline auto
gpu_device_native_atomic_supported(int* value,
                                   int src_id,
                                   int dst_id) noexcept -> gpu_error_t {
#if CARBON_BACKEND_IS_HIP
  return hipDeviceGetP2PAttribute(value, hipDevP2PAttrNativeAtomicSupported, src_id, dst_id);
#else
  return cudaDeviceGetP2PAttribute(value, cudaDevP2PAttrNativeAtomicSupported, src_id, dst_id);
#endif
}

// ---------------------------------------------------------------------------
// CUDA-graph capture / replay surface. These are host-callable runtime-API
// entry points (no device compiler needed), so a host .cpp linked against the
// runtime — e.g. src/skeleton.cpp — can drive stream capture and graph launch
// through the same neutral gpu_* aliases the rest of the tree uses. CUDA and
// HIP expose an identical shape (hipGraph_t / hipStreamBeginCapture / ...).
using gpu_graph_t =
#if CARBON_BACKEND_IS_HIP
    hipGraph_t;
#else
    cudaGraph_t;
#endif
using gpu_graph_exec_t =
#if CARBON_BACKEND_IS_HIP
    hipGraphExec_t;
#else
    cudaGraphExec_t;
#endif
using gpu_stream_capture_mode_t =
#if CARBON_BACKEND_IS_HIP
    hipStreamCaptureMode;
#else
    cudaStreamCaptureMode;
#endif
inline constexpr gpu_stream_capture_mode_t gpu_stream_capture_mode_thread_local =
#if CARBON_BACKEND_IS_HIP
    hipStreamCaptureModeThreadLocal;
#else
    cudaStreamCaptureModeThreadLocal;
#endif
// Process-wide capture: work enqueued on the capturing stream(s) from *any*
// thread is captured. Required for the multi-GPU fork/join capture, where the
// per-GPU kernel launches fan out across an OpenMP team (transient worker
// threads enqueue onto the captured peer streams).
inline constexpr gpu_stream_capture_mode_t gpu_stream_capture_mode_global =
#if CARBON_BACKEND_IS_HIP
    hipStreamCaptureModeGlobal;
#else
    cudaStreamCaptureModeGlobal;
#endif

// HIP's NVIDIA platform shim aliases the graph *types* to their CUDA counterparts but does
// not forward the graph *entry points*, so `hipStreamBeginCapture` and friends are undeclared
// under -DCMAKE_HIP_PLATFORM=nvidia. Because the types are already the CUDA ones there, the
// CUDA entry points take them directly — this is only about which name is spelled.
#if CARBON_BACKEND_IS_HIP && !defined(__HIP_PLATFORM_NVIDIA__)
#define CARBON_GRAPH_USE_HIP_ENTRY_POINTS 1
#else
#define CARBON_GRAPH_USE_HIP_ENTRY_POINTS 0
#endif

// That platform also keeps hipError_t a distinct enum from cudaError_t, so a CUDA entry
// point's status goes through HIP's own mapping helper before it can be returned as
// gpu_error_t. Identity everywhere else.
#if CARBON_BACKEND_IS_HIP && defined(__HIP_PLATFORM_NVIDIA__)
#define CARBON_GRAPH_STATUS(expr) hipCUDAErrorTohipError(expr)
#else
#define CARBON_GRAPH_STATUS(expr) (expr)
#endif

inline auto
gpu_stream_begin_capture(gpu_stream_t stream,
                         gpu_stream_capture_mode_t mode) noexcept -> gpu_error_t {
#if CARBON_GRAPH_USE_HIP_ENTRY_POINTS
  return hipStreamBeginCapture(stream, mode);
#else
  return CARBON_GRAPH_STATUS(cudaStreamBeginCapture(stream, mode));
#endif
}

inline auto
gpu_stream_end_capture(gpu_stream_t stream,
                       gpu_graph_t* graph) noexcept -> gpu_error_t {
#if CARBON_GRAPH_USE_HIP_ENTRY_POINTS
  return hipStreamEndCapture(stream, graph);
#else
  return CARBON_GRAPH_STATUS(cudaStreamEndCapture(stream, graph));
#endif
}

inline auto
gpu_graph_instantiate(gpu_graph_exec_t* exec,
                      gpu_graph_t graph) noexcept -> gpu_error_t {
#if CARBON_GRAPH_USE_HIP_ENTRY_POINTS
  return hipGraphInstantiateWithFlags(exec, graph, 0);
#else
  return CARBON_GRAPH_STATUS(cudaGraphInstantiateWithFlags(exec, graph, 0));
#endif
}

inline auto
gpu_graph_launch(gpu_graph_exec_t exec,
                 gpu_stream_t stream) noexcept -> gpu_error_t {
#if CARBON_GRAPH_USE_HIP_ENTRY_POINTS
  return hipGraphLaunch(exec, stream);
#else
  return CARBON_GRAPH_STATUS(cudaGraphLaunch(exec, stream));
#endif
}

inline auto
gpu_graph_destroy(gpu_graph_t graph) noexcept -> gpu_error_t {
#if CARBON_GRAPH_USE_HIP_ENTRY_POINTS
  return hipGraphDestroy(graph);
#else
  return CARBON_GRAPH_STATUS(cudaGraphDestroy(graph));
#endif
}

inline auto
gpu_graph_exec_destroy(gpu_graph_exec_t exec) noexcept -> gpu_error_t {
#if CARBON_GRAPH_USE_HIP_ENTRY_POINTS
  return hipGraphExecDestroy(exec);
#else
  return CARBON_GRAPH_STATUS(cudaGraphExecDestroy(exec));
#endif
}

#undef CARBON_GRAPH_USE_HIP_ENTRY_POINTS
#undef CARBON_GRAPH_STATUS

#if CARBON_CUDALIKE_DEVICE_TU
#if CARBON_BACKEND_IS_HIP
namespace gpucub = ::hipcub;
#else
namespace gpucub = ::cub;
#endif
#endif

}  // namespace detail
}  // namespace carbon

#endif  // CARBON_BACKEND_IS_CUDALIKE
