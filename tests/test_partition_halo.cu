// Halo-aware Partition + Span test (Stencil radius > 0).
//
// Builds a Dense grid with a radius-1 stencil so each per-GPU Span carries a 1-thick z halo
// on both sides. Allocates a single-component field over the haloed storage, fills the entire
// haloed buffer (lower halo + owned + upper halo) on the host with values that encode the
// global (x, y, z) coordinates, and launches a kernel that, for every owned cell, reads
// itself and its (0, 0, +/-1) z neighbours through Partition::read. Both neighbours are
// guaranteed to live inside the haloed buffer (the lower halo for the first owned z layer,
// the upper halo for the last), so a successful read here means the halo arithmetic on
// Span (xy_slab_elements + linear_offset + z_offset) and Partition::read are all consistent.

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/stencil.h>
#include <carbon/transfer.h>

#include "test_domain.h"

#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace {

CARBON_HD auto
encoded_value(int64_t global_x,
              int64_t global_y,
              int64_t global_z) -> std::uint32_t {
  return static_cast<std::uint32_t>(10000 * (global_z + 100) + 100 * (global_y + 10) +
                                    (global_x + 10));
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_partition_halo: no devices visible");
    return 1;
  }

  // 7-point-style stencil sized for a radius of 1 on z. Center is dropped by the stencil.
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  if (stencil.radius() != 1) {
    CARBON_LOG_ERROR("test_partition_halo: unexpected stencil radius {}", stencil.radius());
    return 1;
  }

  carbon::System sys(carbon::get_all_devices());
  // nz must be at least 2 per GPU once split, so the upper-halo neighbour of the topmost owned
  // layer and the lower-halo neighbour of the bottom layer have something useful to read.
  const int64_t nz = carbon::test::dense_slab_z(8, sys.gpu_group().size(), stencil.radius());
  const carbon::Dense dense(sys, carbon::int64_3d{4, 4, nz}, std::move(stencil));
  const size_t gpu_n = dense.gpu_count();
  if (gpu_n == 0U) {
    CARBON_LOG_ERROR("test_partition_halo: zero GPUs");
    return 1;
  }

  // Sanity-check the Span shape on every GPU before involving any device work.
  for (size_t g = 0; g < gpu_n; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const auto logic = span.logic_extent();
    if (span.halo_z() != 1) {
      CARBON_LOG_ERROR("test_partition_halo: gpu {} halo_z {} != 1", g, span.halo_z());
      return 1;
    }
    if (haloed.z != logic.z + 2) {
      CARBON_LOG_ERROR("test_partition_halo: gpu {} haloed.z={} logic.z={} (expected +2)", g,
                       haloed.z, logic.z);
      return 1;
    }
    if (span.xy_slab_elements() != haloed.x * haloed.y) {
      CARBON_LOG_ERROR("test_partition_halo: gpu {} xy_slab_elements {} != haloed.x*haloed.y {}", g,
                       span.xy_slab_elements(), haloed.x * haloed.y);
      return 1;
    }
  }

  // Per-GPU sizes for the haloed-storage field buffer and the per-cell result buffer.
  std::vector<size_t> field_counts;
  std::vector<size_t> cell_counts;
  field_counts.reserve(gpu_n);
  cell_counts.reserve(gpu_n);
  for (size_t g = 0; g < gpu_n; ++g) {
    field_counts.push_back(static_cast<size_t>(dense.spans()[g].linear_storage_elements()));
    cell_counts.push_back(dense.linear_elements(g));
  }
  carbon::Buffer<std::uint32_t> d_fields = sys.gpu_group().buffer<std::uint32_t>(field_counts);
  carbon::Buffer<std::uint32_t> d_results = sys.gpu_group().buffer<std::uint32_t>(cell_counts);

  // Stage the encoded haloed-buffer pattern + zero result host-side, then upload through
  // a backend-neutral Transfer.
  std::vector<carbon::detail::Allocation<std::uint32_t>> h_field_v;
  std::vector<carbon::detail::Allocation<std::uint32_t>> h_zero_v;
  h_field_v.reserve(gpu_n);
  h_zero_v.reserve(gpu_n);
  for (size_t g = 0; g < gpu_n; ++g) {
    auto h_field = carbon::Cpu::buffer<std::uint32_t>(field_counts[g]);
    const auto& span = dense.spans()[g];
    const carbon::int64_3d haloed = span.haloed_extent();
    const int64_t plane = span.xy_slab_elements();
    // Buffer-local z=0 maps to global span.z_offset(), so the lower halo (z=0), owned cells
    // (z in [1, 1+local_z)), and upper halo all carry the same encoding scheme.
    for (int64_t z = 0; z < haloed.z; ++z) {
      const int64_t global_z = span.z_offset() + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * plane + y * haloed.x + x;
          h_field.data()[static_cast<size_t>(off)] = encoded_value(x, y, global_z);
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
    CARBON_LOG_ERROR("test_partition_halo: upload failed: {}", e.what());
    return 1;
  }

  // Capture raw device pointers for the kernel closure (this test exercises Partition::read
  // built directly from a raw pointer + the Span halo math).
  std::vector<std::uint32_t*> d_field_ptrs(gpu_n, nullptr);
  std::vector<std::uint32_t*> d_result_ptrs(gpu_n, nullptr);
  for (size_t g = 0; g < gpu_n; ++g) {
    d_field_ptrs[g] = d_fields[g].data();
    d_result_ptrs[g] = d_results[g].data();
  }

  const auto kernel = dense.make_kernel(
      [&d_field_ptrs, &d_result_ptrs](carbon::KernelLoader& loader) {
        using Idx = carbon::Dense::Idx;
        const size_t rel = loader.relative_gpu_id();
        std::uint32_t* const field_ptr = d_field_ptrs[rel];
        std::uint32_t* const result_ptr = d_result_ptrs[rel];
        const carbon::Dense::Partition<std::uint32_t, 1> part(field_ptr);
        return [part, result_ptr] CARBON_HOST_DEVICE(carbon::Dense::Span span, Idx idx) {
          const carbon::int64_3d cell = idx.cell();
          const carbon::int64_3d lim = span.logic_extent();
          // Owned cells live at buffer-local z in [halo_z, halo_z + lim.z); the launch grid
          // already iterates only owned cells, so cell.z is in that range.
          const int64_t owned_z = cell.z - span.halo_z();
          const int64_t out_off = owned_z * lim.x * lim.y + cell.y * lim.x + cell.x;
          const int64_t global_z = span.global_location(idx).z;

          std::uint32_t ok = 1U;
          // Self
          if (part.read(span, idx, 0) != encoded_value(cell.x, cell.y, global_z)) {
            ok = 0U;
          }
          // -z neighbour (always inside the haloed buffer because owned z >= halo_z = 1)
          {
            const Idx ng(cell.x, cell.y, cell.z - 1);
            if (part.read(span, ng, 0) != encoded_value(cell.x, cell.y, global_z - 1)) {
              ok = 0U;
            }
          }
          // +z neighbour (always inside the haloed buffer because owned z < halo_z + lim.z, so
          // cell.z + 1 < haloed.z = lim.z + 2)
          {
            const Idx ng(cell.x, cell.y, cell.z + 1);
            if (part.read(span, ng, 0) != encoded_value(cell.x, cell.y, global_z + 1)) {
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
    CARBON_LOG_ERROR("test_partition_halo: kernel.launch / wait failed: {}", e.what());
    return 1;
  }

  // Download the per-cell result through a backend-neutral Transfer and verify.
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
    CARBON_LOG_ERROR("test_partition_halo: download / wait failed: {}", e.what());
    return 1;
  }

  int rc = 0;
  for (size_t g = 0; g < gpu_n; ++g) {
    const std::uint32_t* host = h_result[g].data();
    for (size_t i = 0; i < cell_counts[g]; ++i) {
      if (host[i] != 1U) {
        CARBON_LOG_ERROR("test_partition_halo: gpu {} cell {} mismatch", g, i);
        rc = 1;
        break;
      }
    }
  }
  return rc;
}
