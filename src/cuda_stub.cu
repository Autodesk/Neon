#include <carbon/cuda_annotation.h>
#include <carbon/detail/dense/kernel.h>
#include <carbon/detail/dense/span.h>
#include <carbon/integer3d.h>
#include <cuda_runtime.h>

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
 * Placeholder CUDA translation unit so the project links against the CUDA runtime.
 * Real device code will live alongside the host API in later iterations.
 */
extern "C" int
carbon_cuda_runtime_linked(void) {
  int deviceCount = 0;
  const cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess) {
    return -1;
  }
  if (deviceCount > 0) {
    unsigned host_out = 0U;
    unsigned* dev_out = nullptr;
    if (cudaMalloc(&dev_out, sizeof(unsigned)) == cudaSuccess && dev_out != nullptr) {
      const carbon::detail::dense::Span span(carbon::int64_3d{4, 5, 1}, carbon::int64_3d{4, 5, 1},
                                             32, 7);
      carbon_dense_layout_device_smoke_kernel<<<1, 32>>>(span, dev_out);
      {
        const carbon::detail::dense::Span span_grid(carbon::int64_3d{2, 2, 2},
                                                    carbon::int64_3d{2, 2, 2}, 4, 0);
        carbon::detail::dense::dense_grid_kernel<<<dim3(1, 1, 1), dim3(2, 2, 2)>>>(
            span_grid, carbon::DataView::Standard, carbon_dense_grid_smoke_fn{});
        (void)cudaGetLastError();
      }
      const carbon::int64_3d g{4, 5, 6};
      carbon_integer3d_device_smoke_kernel<<<1, 32>>>(g, dev_out);
      if (cudaMemcpy(&host_out, dev_out, sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess) {
        (void)host_out;  // slab z=1 => volume 4*5*1 = 20
      }
      (void)cudaFree(dev_out);
    }
  }
  return deviceCount;
}

namespace {

__global__ void
carbon_smoke_kernel_impl() {}

}  // namespace

extern "C" cudaError_t
carbon_launch_smoke_kernel(int device_id) {
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    return err;
  }
  carbon_smoke_kernel_impl<<<1, 32>>>();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }
  return cudaDeviceSynchronize();
}
