// Row-copy kernel for the DenseDisgDistributed pencil halo (CUDA and HIP backends). See
// include/carbon/detail/dense_disg_distributed/row_copy.h.
//
// The __global__ is a one-line forwarder to row_copy_thread, which the host tests exercise through
// row_copy_host's identical job walk. Keep it trivial.

#include <carbon/detail/backend/cuda_like.h>
#include <carbon/detail/dense_disg_distributed/row_copy.h>
#include <carbon/detail/gpu.h>

#include <cstddef>
#include <cstdint>

namespace carbon::detail::dense_disg_distributed {
namespace {

struct alignas(16) Word16 {
  std::uint64_t lo;
  std::uint64_t hi;
};

template <typename W>
__global__ void
row_copy_kernel(W* base,
                RowCopyArgs a) {
  const int64_t x = static_cast<int64_t>(blockIdx.x) * kRowCopyBlockX + threadIdx.x;
  row_copy_thread<W>(base, a, blockIdx.z, blockIdx.y, x);
}

template <typename W>
auto
row_copy_as(const Gpu& gpu,
            gpu_stream_t stream,
            void* base,
            const RowCopyArgs& a) -> gpu_error_t {
  const auto blocks_x =
      static_cast<unsigned>((a.max_len + static_cast<int64_t>(kRowCopyBlockX) - 1) /
                            static_cast<int64_t>(kRowCopyBlockX));
  return gpu.template typed_launch<&row_copy_kernel<W>>(
      dim3(blocks_x, static_cast<unsigned>(a.max_rows), static_cast<unsigned>(a.job_count)),
      dim3(kRowCopyBlockX, 1U, 1U), 0U, stream, static_cast<W*>(base), a);
}

}  // namespace

auto
launch_row_copy(const Gpu& gpu,
                gpu_stream_t stream,
                std::size_t elem_bytes,
                void* base,
                const RowCopyArgs& a) -> gpu_error_t {
  if (a.empty()) {
    return gpu_success;
  }
  switch (elem_bytes) {
    case 1U:
      return row_copy_as<std::uint8_t>(gpu, stream, base, a);
    case 2U:
      return row_copy_as<std::uint16_t>(gpu, stream, base, a);
    case 4U:
      return row_copy_as<std::uint32_t>(gpu, stream, base, a);
    case 8U:
      return row_copy_as<std::uint64_t>(gpu, stream, base, a);
    case 16U:
      return row_copy_as<Word16>(gpu, stream, base, a);
    default:
      return gpu_error_invalid_value;
  }
}

}  // namespace carbon::detail::dense_disg_distributed
