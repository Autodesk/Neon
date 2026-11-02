// Packed halo gather/scatter kernels (CUDA and HIP backends). See include/carbon/detail/halo_pack.h.
//
// Each __global__ is a one-line forwarder to the host+device functions in halo_pack_plan.h, which
// tests/test_halo_pack_plan.cpp exercises on the host with the same arguments. Keep the kernels
// trivial: any logic added here is NOT covered by that test.

#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/gpu.h>
#include <carbon/detail/halo_pack.h>

#include <cstddef>
#include <cstdint>

namespace carbon::detail::halo_pack {
namespace {

struct alignas(16) Word16 {
  std::uint64_t lo;
  std::uint64_t hi;
};

template <typename W>
__global__ void
gather_kernel(const W* __restrict__ field,
              W* __restrict__ stage,
              PackKernelArgs a) {
  pack_gather_thread<W>(field, stage, a, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
                        blockDim.x, gridDim.y);
}

template <typename W>
__global__ void
scatter_kernel(const W* __restrict__ stage,
               W* __restrict__ field,
               PackKernelArgs a) {
  pack_scatter_thread<W>(stage, field, a, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x,
                         blockDim.x, gridDim.y);
}

template <typename W>
auto
gather_as(const Gpu& gpu,
          gpu_stream_t stream,
          const void* field,
          void* stage,
          const PackKernelArgs& a) -> gpu_error_t {
  const PackLaunchShape s = pack_launch_shape(a);
  return gpu.template typed_launch<&gather_kernel<W>>(
      dim3(s.grid_x, s.grid_y, s.grid_z), dim3(s.block_x, 1U, 1U), 0U, stream,
      static_cast<const W*>(field), static_cast<W*>(stage), a);
}

template <typename W>
auto
scatter_as(const Gpu& gpu,
           gpu_stream_t stream,
           const void* stage,
           void* field,
           const PackKernelArgs& a) -> gpu_error_t {
  const PackLaunchShape s = pack_launch_shape(a);
  return gpu.template typed_launch<&scatter_kernel<W>>(
      dim3(s.grid_x, s.grid_y, s.grid_z), dim3(s.block_x, 1U, 1U), 0U, stream,
      static_cast<const W*>(stage), static_cast<W*>(field), a);
}

}  // namespace

auto
launch_gather(const Gpu& gpu,
              gpu_stream_t stream,
              std::size_t elem_bytes,
              const void* field_base,
              void* stage_region,
              const PackKernelArgs& a) -> gpu_error_t {
  switch (elem_bytes) {
    case 1U:
      return gather_as<std::uint8_t>(gpu, stream, field_base, stage_region, a);
    case 2U:
      return gather_as<std::uint16_t>(gpu, stream, field_base, stage_region, a);
    case 4U:
      return gather_as<std::uint32_t>(gpu, stream, field_base, stage_region, a);
    case 8U:
      return gather_as<std::uint64_t>(gpu, stream, field_base, stage_region, a);
    case 16U:
      return gather_as<Word16>(gpu, stream, field_base, stage_region, a);
    default:
      return gpu_error_invalid_value;
  }
}

auto
launch_scatter(const Gpu& gpu,
               gpu_stream_t stream,
               std::size_t elem_bytes,
               const void* stage_region,
               void* field_base,
               const PackKernelArgs& a) -> gpu_error_t {
  switch (elem_bytes) {
    case 1U:
      return scatter_as<std::uint8_t>(gpu, stream, stage_region, field_base, a);
    case 2U:
      return scatter_as<std::uint16_t>(gpu, stream, stage_region, field_base, a);
    case 4U:
      return scatter_as<std::uint32_t>(gpu, stream, stage_region, field_base, a);
    case 8U:
      return scatter_as<std::uint64_t>(gpu, stream, stage_region, field_base, a);
    case 16U:
      return scatter_as<Word16>(gpu, stream, stage_region, field_base, a);
    default:
      return gpu_error_invalid_value;
  }
}

}  // namespace carbon::detail::halo_pack
