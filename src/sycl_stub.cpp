// SYCL counterpart of src/cuda_stub.cu. Provides the same `extern "C"` symbols carbon's
// rest-of-library expects (carbon_cuda_runtime_linked, carbon_launch_smoke_kernel) but
// implemented through sycl::queue::parallel_for instead of __global__ + <<<>>>.
//
// "cuda" in the symbol names is residual: those symbols are defined in the CUDA stub and
// kept here unchanged so call sites elsewhere (Gpu::launch_smoke_kernel via gpu_sycl.cpp,
// and any external linkage probe) do not need to fork on backend at the symbol layer.
//
// The file lives outside detail/backend/sycl/ because its job is to satisfy specific
// host-visible symbols from the SYCL build of libcarbon.so; consumers do not include it.

#include <carbon/cuda_annotation.h>
#include <carbon/detail/dense/span.h>
#include <carbon/integer3d.h>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <vector>

namespace {

/**
 * Pick the same "default" device the SYCL gpu_count() exposes as device 0. Matches
 * gpu_sycl.cpp's all_sycl_devices() ordering by enumerating platforms and skipping CPU
 * devices.
 */
auto
first_gpu_or_default() -> sycl::device {
  for (const auto& platform : sycl::platform::get_platforms()) {
    for (const auto& dev : platform.get_devices(sycl::info::device_type::all)) {
      if (!dev.is_cpu()) {
        return dev;
      }
    }
  }
  // Fall back to whatever the default selector picks (typically the host CPU). Worst
  // case the smoke parallel_for runs on the CPU, which is a useful diagnostic in itself
  // (the toolchain works, no GPU is visible to AdaptiveCpp).
  return sycl::device{sycl::default_selector_v};
}

}  // namespace

/**
 * Probe that exercises the SYCL toolchain and the carbon types reachable from device
 * code (Span / int64_3d). Returns the visible GPU device count, or -1 on failure. Same
 * contract as the CUDA stub at src/cuda_stub.cu, just routed through SYCL.
 */
extern "C" int
carbon_cuda_runtime_linked(void) {
  try {
    sycl::device dev = first_gpu_or_default();
    sycl::queue q(dev, sycl::property::queue::in_order{});

    unsigned* dev_out = sycl::malloc_device<unsigned>(1, q);
    if (dev_out == nullptr) {
      return -1;
    }

    // Span on-device construction smoke (mirrors cuda_stub's first kernel).
    const carbon::detail::dense::Span span(carbon::int64_3d{4, 5, 1}, carbon::int64_3d{4, 5, 1}, 32,
                                           7);
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
       const carbon::detail::dense::Idx idx = span.make_idx(0, 0, 0);
       const auto g = span.global_location(idx);
       *dev_out = static_cast<unsigned>(g.z + span.xy_slab_elements());
     }).wait();

    // int64_3d on-device smoke (mirrors cuda_stub's third kernel).
    const carbon::int64_3d g{4, 5, 6};
    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
       const carbon::int64_3d slab = g.with_z(1);
       *dev_out = static_cast<unsigned>(slab.volume());
     }).wait();

    unsigned host_out = 0U;
    q.memcpy(&host_out, dev_out, sizeof(unsigned)).wait();
    (void)host_out;

    sycl::free(dev_out, q);

    // Count GPU+accelerator devices the same way gpu_sycl.cpp does.
    int n = 0;
    for (const auto& platform : sycl::platform::get_platforms()) {
      for (const auto& d : platform.get_devices(sycl::info::device_type::all)) {
        if (!d.is_cpu()) {
          ++n;
        }
      }
    }
    return n;
  } catch (...) {
    // Mirror the CUDA stub's "any error returns -1" contract.
    return -1;
  }
}

/**
 * Smoke kernel launcher used by Gpu::launch_smoke_kernel. Submits a 1-thread
 * parallel_for on the device matching @p device_id and waits. Returns 0 on success,
 * non-zero on any thrown SYCL exception. The return type is plain @c int rather than
 * the CUDA stub's @c cudaError_t to avoid pulling cuda_runtime.h into this TU.
 */
extern "C" int
carbon_launch_smoke_kernel(int device_id) {
  try {
    int idx = 0;
    sycl::device chosen{};
    bool found = false;
    for (const auto& platform : sycl::platform::get_platforms()) {
      for (const auto& d : platform.get_devices(sycl::info::device_type::all)) {
        if (d.is_cpu()) {
          continue;
        }
        if (idx == device_id) {
          chosen = d;
          found = true;
          break;
        }
        ++idx;
      }
      if (found) {
        break;
      }
    }
    if (!found) {
      return 1;
    }
    sycl::queue q(chosen, sycl::property::queue::in_order{});
    q.parallel_for(sycl::range<1>(1), [](sycl::id<1>) {
     }).wait();
    return 0;
  } catch (...) {
    return 2;
  }
}
