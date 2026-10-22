// Verifies Partition::read returns the encoded payload for every (component, cell) pair.
//
// Allocates one device-side carbon::Buffer<uint32_t> per GPU sized to that slab's
// haloed storage * cardinality, uploads a deterministic encoded pattern, runs a kernel
// whose closure builds a Partition<uint32_t, 1> over the captured device pointer and
// reads every component back, writing 1 into a per-cell result buffer iff every read
// matches the encoding. The result buffer is downloaded through a Transfer and verified
// host-side.

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace {

auto
encoded_value(int64_t x,
              int64_t y,
              int64_t z,
              int64_t c) -> std::uint32_t {
  return static_cast<std::uint32_t>(1000000 * c + 10000 * z + 100 * y + x);
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_partition_read: no devices visible");
    return 1;
  }

  carbon::System sys(carbon::get_all_devices());
  const carbon::Dense dense(sys, carbon::int64_3d{4, 4, 4});
  const size_t gpu_n = dense.gpu_count();
  if (gpu_n == 0U) {
    CARBON_LOG_ERROR("test_partition_read: Dense has zero GPUs");
    return 1;
  }

  constexpr int cardinality = 2;

  // Per-GPU sizes for the (haloed storage * cardinality) field buffer and the
  // (logic-extent) result buffer.
  std::vector<size_t> field_counts;
  std::vector<size_t> cell_counts;
  field_counts.reserve(gpu_n);
  cell_counts.reserve(gpu_n);
  for (size_t g = 0; g < gpu_n; ++g) {
    const carbon::Dense::Span& span = dense.spans()[g];
    field_counts.push_back(static_cast<size_t>(span.linear_storage_elements()) * cardinality);
    cell_counts.push_back(dense.linear_elements(g));
  }

  carbon::Buffer<std::uint32_t> d_fields = sys.gpu_group().buffer<std::uint32_t>(field_counts);
  carbon::Buffer<std::uint32_t> d_results = sys.gpu_group().buffer<std::uint32_t>(cell_counts);

  // Stage the encoded pattern + a zero-filled result buffer host-side, then upload.
  std::vector<carbon::detail::Allocation<std::uint32_t>> h_field_v;
  std::vector<carbon::detail::Allocation<std::uint32_t>> h_zero_v;
  h_field_v.reserve(gpu_n);
  h_zero_v.reserve(gpu_n);
  for (size_t g = 0; g < gpu_n; ++g) {
    auto h_field = carbon::Cpu::buffer<std::uint32_t>(field_counts[g]);
    const carbon::Dense::Span& span = dense.spans()[g];
    const carbon::int64_3d local = span.haloed_extent();
    const int64_t plane = span.xy_slab_elements();
    const int64_t per_component = span.linear_storage_elements();
    for (int64_t c = 0; c < cardinality; ++c) {
      for (int64_t z = 0; z < local.z; ++z) {
        for (int64_t y = 0; y < local.y; ++y) {
          for (int64_t x = 0; x < local.x; ++x) {
            const int64_t off = c * per_component + z * plane + y * local.x + x;
            h_field.data()[static_cast<size_t>(off)] = encoded_value(x, y, z + span.z_offset(), c);
          }
        }
      }
    }
    h_field_v.push_back(std::move(h_field));

    auto h_zero = carbon::Cpu::buffer<std::uint32_t>(cell_counts[g]);
    for (size_t i = 0; i < cell_counts[g]; ++i) {
      h_zero.data()[i] = 0U;
    }
    h_zero_v.push_back(std::move(h_zero));
  }
  carbon::Buffer<std::uint32_t> h_field =
      carbon::detail::BufferGroup<std::uint32_t>::from_vector(std::move(h_field_v));
  carbon::Buffer<std::uint32_t> h_zero =
      carbon::detail::BufferGroup<std::uint32_t>::from_vector(std::move(h_zero_v));

  carbon::Stream streams = sys.stream();
  try {
    carbon::Transfer<std::uint32_t> upload_field(sys, h_field, d_fields);
    upload_field(streams);
    carbon::Transfer<std::uint32_t> upload_zero(sys, h_zero, d_results);
    upload_zero(streams);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_partition_read: upload failed: {}", e.what());
    return 1;
  }

  // Capture raw device pointers for the kernel closure: this test exercises Partition::read
  // built from a raw pointer (so the per-component offset math is what's under test).
  std::vector<std::uint32_t*> d_field_ptrs(gpu_n, nullptr);
  std::vector<std::uint32_t*> d_result_ptrs(gpu_n, nullptr);
  for (size_t g = 0; g < gpu_n; ++g) {
    d_field_ptrs[g] = d_fields[g].data();
    d_result_ptrs[g] = d_results[g].data();
  }

  const auto kernel = dense.make_kernel(
      [&d_field_ptrs, &d_result_ptrs](carbon::KernelLoader& loader) {
        const size_t rel = loader.relative_gpu_id();
        std::uint32_t* const field_ptr = d_field_ptrs[rel];
        std::uint32_t* const result_ptr = d_result_ptrs[rel];
        const carbon::Dense::Partition<std::uint32_t, 1> part(field_ptr);
        return [part, result_ptr] CARBON_HOST_DEVICE(carbon::Dense::Span span,
                                                     carbon::Dense::Idx idx) {
          const carbon::int64_3d cell = idx.cell();
          const carbon::int64_3d lim = span.logic_extent();
          const int64_t owned_z = cell.z - span.halo_z();
          const int64_t out_off = owned_z * lim.x * lim.y + cell.y * lim.x + cell.x;

          std::uint32_t ok = 1U;
          for (int c = 0; c < cardinality; ++c) {
            const std::uint32_t got = part.read(span, idx, c);
            const std::uint32_t expected = static_cast<std::uint32_t>(
                1000000 * c + 10000 * span.global_location(idx).z + 100 * cell.y + cell.x);
            if (got != expected) {
              ok = 0U;
            }
          }
          result_ptr[static_cast<size_t>(out_off)] = ok;
        };
      },
      0U);

  try {
    kernel.launch(streams);
    streams.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_partition_read: kernel launch / wait failed: {}", e.what());
    return 1;
  }

  // Download the per-cell result via a backend-neutral Transfer and verify every cell == 1.
  std::vector<carbon::detail::Allocation<std::uint32_t>> h_result_v;
  h_result_v.reserve(gpu_n);
  for (size_t g = 0; g < gpu_n; ++g) {
    h_result_v.push_back(carbon::Cpu::buffer<std::uint32_t>(cell_counts[g]));
  }
  carbon::Buffer<std::uint32_t> h_result =
      carbon::detail::BufferGroup<std::uint32_t>::from_vector(std::move(h_result_v));

  try {
    carbon::Transfer<std::uint32_t> download(sys, d_results, h_result);
    download(streams);
    streams.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_partition_read: download / wait failed: {}", e.what());
    return 1;
  }

  int rc = 0;
  for (size_t g = 0; g < gpu_n; ++g) {
    const std::uint32_t* host = h_result[g].data();
    for (size_t i = 0; i < cell_counts[g]; ++i) {
      if (host[i] != 1U) {
        CARBON_LOG_ERROR("test_partition_read: gpu {} cell {} mismatch", g, i);
        rc = 1;
        break;
      }
    }
  }
  return rc;
}
