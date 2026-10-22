// Smoke test for the dense kernel launch path.
//
// Allocates one carbon::Buffer<uint32_t> per GPU sized to that GPU's slab logic-extent,
// runs a kernel that writes 1U into every owned cell via the captured raw device pointer,
// downloads through a Transfer back to a host mirror, and verifies every cell == 1U. The
// raw-pointer capture is intentional: this test exercises the launch + per-cell index math
// without the convenience of a Field/Partition wrapper, which is why it lives at this
// layer rather than higher up in the stack.

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <vector>

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_dense_kernel: no devices visible");
    return 1;
  }

  carbon::System sys(carbon::get_all_devices());
  const carbon::Dense dense(sys, carbon::int64_3d{4, 4, 4});

  if (dense.gpu_count() == 0U) {
    CARBON_LOG_ERROR("test_dense_kernel: Dense has zero GPUs");
    return 1;
  }

  // One device buffer per GPU sized to that slab's logic-extent (no halo on this grid).
  const size_t gpu_n = dense.gpu_count();
  std::vector<size_t> counts;
  counts.reserve(gpu_n);
  for (size_t g = 0; g < gpu_n; ++g) {
    counts.push_back(dense.linear_elements(g));
  }
  carbon::Buffer<std::uint32_t> d_bufs = sys.gpu_group().buffer<std::uint32_t>(counts);

  // Capture raw per-GPU device pointers up front — the kernel closure dereferences these
  // by relative_gpu_id without going through Field/Partition (this is the dense-kernel
  // smoke test, not a Partition test).
  std::vector<std::uint32_t*> d_ptrs(gpu_n, nullptr);
  for (size_t g = 0; g < gpu_n; ++g) {
    d_ptrs[g] = d_bufs[g].data();
  }

  // Zero-fill each device buffer via a transfer from a host-side zero buffer; replaces the
  // legacy cudaMemset path with the same backend-neutral primitive used by every other
  // upload in carbon's tests.
  std::vector<carbon::detail::Allocation<std::uint32_t>> h_zero_v;
  h_zero_v.reserve(gpu_n);
  for (size_t g = 0; g < gpu_n; ++g) {
    auto h = carbon::Cpu::buffer<std::uint32_t>(counts[g]);
    for (size_t i = 0; i < counts[g]; ++i) {
      h.data()[i] = 0U;
    }
    h_zero_v.push_back(std::move(h));
  }
  carbon::Buffer<std::uint32_t> h_zero =
      carbon::detail::BufferGroup<std::uint32_t>::from_vector(std::move(h_zero_v));

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<std::uint32_t> zero_upload(sys, h_zero, d_bufs);
    zero_upload(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_kernel: zero upload failed: {}", e.what());
    return 1;
  }

  const auto kernel = dense.make_kernel(
      [&d_ptrs](carbon::KernelLoader& loader) {
        std::uint32_t* const d_buf = d_ptrs[loader.relative_gpu_id()];
        return [d_buf] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const carbon::int64_3d c = idx.cell();
          const carbon::int64_3d lim = s.logic_extent();
          const int64_t pitch = s.xy_slab_elements();
          const int64_t owned_z = c.z - s.halo_z();
          const int64_t off = owned_z * pitch + c.y * lim.x + c.x;
          d_buf[static_cast<size_t>(off)] = 1U;
        };
      },
      0U);

  try {
    kernel.launch(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_kernel: kernel launch / wait failed: {}", e.what());
    return 1;
  }

  // Download the result through a backend-neutral Transfer; same-shape host BufferGroup as
  // the zero-fill above.
  std::vector<carbon::detail::Allocation<std::uint32_t>> h_out_v;
  h_out_v.reserve(gpu_n);
  for (size_t g = 0; g < gpu_n; ++g) {
    h_out_v.push_back(carbon::Cpu::buffer<std::uint32_t>(counts[g]));
  }
  carbon::Buffer<std::uint32_t> h_out =
      carbon::detail::BufferGroup<std::uint32_t>::from_vector(std::move(h_out_v));

  try {
    carbon::Transfer<std::uint32_t> download(sys, d_bufs, h_out);
    download(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_kernel: download / wait failed: {}", e.what());
    return 1;
  }

  for (size_t g = 0; g < gpu_n; ++g) {
    const size_t n_g = counts[g];
    const std::uint32_t* host = h_out[g].data();
    for (size_t i = 0; i < n_g; ++i) {
      if (host[i] != 1U) {
        CARBON_LOG_ERROR("test_dense_kernel: gpu {} host[{}] == {} (expected 1)", g, i, host[i]);
        return 1;
      }
    }
  }

  return 0;
}
