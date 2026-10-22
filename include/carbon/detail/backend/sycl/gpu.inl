#pragma once

// Inline definitions for the SYCL-backend Gpu class. Selected by the dispatcher in
// detail/gpu.inl when CARBON_BACKEND_IS_SYCL. The non-inline parts (Gpu::Gpu,
// Gpu::device_count, Gpu::set_device, Gpu::report, Gpu::launch_smoke_kernel,
// Gpu::allocated_bytes) live in src/gpu_sycl.cpp.

#include <carbon/detail/allocation.h>
// Pull in CudaStream / Event through the public dispatch path so the same shim layer
// resolves to the SYCL backend headers next to this file. Going through the dispatcher
// (rather than including detail/backend/sycl/{stream,event}.h directly) keeps the SYCL
// gpu.inl symmetric with the CUDA gpu.inl, which uses the same indirection.
#include <carbon/detail/event.h>
#include <carbon/detail/gpu_allocation_tracking.h>
#include <carbon/detail/stream.h>
#include <carbon/logging.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

namespace carbon {
namespace detail {

inline auto
Gpu::stream() const -> CudaStream {
  // sycl::queue construction is the closest analogue to cudaStreamCreate. We use the
  // in_order property because carbon's existing single-stream-per-device call sites
  // (dense_kernel_impl, reduce_kernel_impl, transfer) issue work serially and rely on
  // ordering between consecutive submissions. AdaptiveCpp + cuda:sm_8x lowers in_order
  // queues to a single CUDA stream per queue, matching the CUDA backend's wire
  // semantics. Out-of-order queues (the SYCL default) can be revisited later when an
  // explicit dependency graph is wired through carbon::Event.
  auto q = std::make_shared<sycl::queue>(context_, device_, sycl::property::queue::in_order{});
  return CudaStream(std::move(q));
}

inline auto
Gpu::event() const -> Event {
  // sycl::event has no "create empty" form like cudaEventCreate — events are produced
  // by submissions. We allocate a default-constructed sycl::event (representing "nothing
  // recorded yet") and let the eventual submission overwrite it. This matches the CUDA
  // wrapper's "event handle exists but is unsignalled" state.
  auto e = std::make_shared<sycl::event>();
  return Event(std::move(e));
}

template <typename T>
auto
Gpu::buffer(size_t count,
            int relative_gpu_id) const -> Allocation<T> {
  const size_t size_bytes = count * sizeof(T);
  void* ptr = nullptr;
  if (size_bytes > 0U) {
    // Device-only USM allocation. Pinning the context here means the allocation can be
    // freed from any host thread later (sycl::free takes the same context). We use
    // device_ptr so cross-rank peer transfers get a stable address per device, matching
    // the CUDA backend's cudaMalloc-then-cudaMemcpyPeerAsync model. AdaptiveCpp's CUDA
    // implementation lowers this to cudaMalloc on the targeted device.
    ptr = sycl::malloc_device(size_bytes, device_, context_);
    if (ptr == nullptr) {
      ::carbon::log::throw_with_log<std::runtime_error>(
          "carbon::detail::Gpu::buffer: sycl::malloc_device failed for {} bytes on device {}",
          size_bytes, device_id_);
    }
    gpu_track_allocation(device_id_, size_bytes);
  }

  const int dev = device_id_;
  // Capture the SYCL context by value so the deleter can free even after this Gpu has
  // been destroyed. sycl::context is a refcounted handle so the copy is O(1).
  sycl::context ctx = context_;
  std::function<void(void*)> free_fn = [dev, size_bytes, ctx](void* p) mutable {
    if (p == nullptr) {
      return;
    }
    if (size_bytes > 0U) {
      gpu_track_deallocation(dev, size_bytes);
    }
    sycl::free(p, ctx);
  };

  return Allocation<T>(AllocationLocation::Cuda, count, device_id_, relative_gpu_id, ptr, free_fn);
}

}  // namespace detail
}  // namespace carbon
