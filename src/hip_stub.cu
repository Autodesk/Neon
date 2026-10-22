// HIP counterpart of src/cuda_stub.cu. Provides the same `extern "C"` symbols carbon's
// rest-of-library expects (carbon_cuda_runtime_linked, carbon_launch_smoke_kernel),
// implemented with __global__ kernels + <<<>>> through the HIP device compiler.
//
// "cuda" in the carbon_cuda_runtime_linked symbol name is residual (kept so external
// linkage probes do not need to fork on backend at the symbol layer).

#include <carbon/cuda_annotation.h>
#include <carbon/detail/dense/kernel.h>
#include <carbon/detail/dense/span.h>
#include <carbon/integer3d.h>

#include <hip/hip_runtime.h>

namespace {

/** Functor satisfies @c dense_kernel_fn for host-side concept checking and device launch. */
struct carbon_dense_grid_smoke_fn {
  CARBON_HOST_DEVICE void
  operator()(carbon::detail::dense::Span s,
             carbon::detail::dense::Idx i) const {
    (void)s;
    (void)i;
  }
};

/** Compile-time check that carbon::detail::dense::Span / Idx are usable from device code. */
__global__ void
carbon_dense_layout_device_smoke_kernel(carbon::detail::dense::Span span,
                                        unsigned* out) {
  const carbon::detail::dense::Idx idx =
      span.make_idx(static_cast<int64_t>(threadIdx.x), static_cast<int64_t>(blockIdx.x), 0);
  if (threadIdx.x == 0U && blockIdx.x == 0U) {
    const auto g = span.global_location(idx);
    *out = static_cast<unsigned>(g.z + span.xy_slab_elements());
  }
}

/** Compile-time check that carbon::Integer3d is usable from device code. */
__global__ void
carbon_integer3d_device_smoke_kernel(const carbon::int64_3d ext,
                                     unsigned* out) {
  if (threadIdx.x == 0U && blockIdx.x == 0) {
    const carbon::int64_3d slab = ext.with_z(1);
    *out = static_cast<unsigned>(slab.volume());
  }
}

}  // namespace

/**
 * Placeholder HIP translation unit so the project links against the HIP runtime and the
 * carbon device types are exercised from device code.
 */
extern "C" int
carbon_cuda_runtime_linked(void) {
  int deviceCount = 0;
  const hipError_t err = hipGetDeviceCount(&deviceCount);
  if (err != hipSuccess) {
    return -1;
  }
  if (deviceCount > 0) {
    unsigned host_out = 0U;
    unsigned* dev_out = nullptr;
    if (hipMalloc(&dev_out, sizeof(unsigned)) == hipSuccess && dev_out != nullptr) {
      const carbon::detail::dense::Span span(carbon::int64_3d{4, 5, 1}, carbon::int64_3d{4, 5, 1},
                                             32, 7);
      carbon_dense_layout_device_smoke_kernel<<<1, 32>>>(span, dev_out);
      {
        const carbon::detail::dense::Span span_grid(carbon::int64_3d{2, 2, 2},
                                                    carbon::int64_3d{2, 2, 2}, 4, 0);
        carbon::detail::dense::dense_grid_kernel<<<dim3(1, 1, 1), dim3(2, 2, 2)>>>(
            span_grid, carbon::DataView::Standard, carbon_dense_grid_smoke_fn{});
        (void)hipGetLastError();
      }
      const carbon::int64_3d g{4, 5, 6};
      carbon_integer3d_device_smoke_kernel<<<1, 32>>>(g, dev_out);
      if (hipMemcpy(&host_out, dev_out, sizeof(unsigned), hipMemcpyDeviceToHost) == hipSuccess) {
        (void)host_out;  // slab z=1 => volume 4*5*1 = 20
      }
      (void)hipFree(dev_out);
    }
  }
  return deviceCount;
}

namespace {

__global__ void
carbon_smoke_kernel_impl() {}

}  // namespace

extern "C" hipError_t
carbon_launch_smoke_kernel(int device_id) {
  hipError_t err = hipSetDevice(device_id);
  if (err != hipSuccess) {
    return err;
  }
  carbon_smoke_kernel_impl<<<1, 32>>>();
  err = hipGetLastError();
  if (err != hipSuccess) {
    return err;
  }
  return hipDeviceSynchronize();
}
