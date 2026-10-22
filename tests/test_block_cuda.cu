// CUDA correctness test for the carbon::Block (block-sparse) grid: builds a Block over
// a sphere predicate, allocates a Field<int32_t, 2>, runs a hand-rolled CUDA kernel that
// walks (block_idx, intra_xyz) over the brick lattice and writes
//
//   value = encode(global_x, global_y, global_z, component)
//
// into every voxel the bitmask says is active (Span::set_and_validate must agree with
// the host predicate); finally copies the field back to the host and verifies that
//   - every active voxel carries the expected encoded value, and
//   - every inactive voxel still carries the post-allocation zero (i.e. the kernel
//     correctly skipped it via the bitmask).
//
// Phase 1 is single-GPU; the test reuses System::get_all_devices() and trims to the
// first GPU. A multi-GPU variant lands together with the Z-block partitioner port.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/system.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<int32_t, 2>;
using PartitionT = BlockGrid::Partition<int32_t, 2>;

[[nodiscard]] CARBON_HD constexpr auto
encode(int gx,
       int gy,
       int gz,
       int c) noexcept -> int32_t {
  return ((gx + 1) * 73856093) ^ ((gy + 1) * 19349663) ^ ((gz + 1) * 83492791) ^ (c + 1);
}

[[nodiscard]] auto
sphere_predicate(int64_t domain_n,
                 const carbon::int64_3d& gc) -> bool {
  const double cx = (domain_n - 1) * 0.5;
  const double cy = (domain_n - 1) * 0.5;
  const double cz = (domain_n - 1) * 0.5;
  const double r = domain_n * 0.45;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

__global__ auto
fill_active_kernel(Span span,
                   PartitionT field,
                   const carbon::int32_3d* origins) -> void {
  const uint32_t block_idx = blockIdx.x;
  const int ix = static_cast<int>(threadIdx.x);
  const int iy = static_cast<int>(threadIdx.y);
  const int iz = static_cast<int>(threadIdx.z);
  Idx idx{};
  if (!span.set_and_validate(block_idx, ix, iy, iz, idx)) {
    return;
  }
  const auto origin = origins[block_idx];
  const int gx = origin.x + ix;
  const int gy = origin.y + iy;
  const int gz = origin.z + iz;
  field.write(span, idx, 0, encode(gx, gy, gz, 0));
  field.write(span, idx, 1, encode(gx, gy, gz, 1));
}

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("test_block_cuda: no CUDA devices visible — skipping\n");
    return 0;
  }
  // Phase 1 is single-GPU; trim to one device.
  if (devices.size() > 1U) {
    devices.erase(devices.begin() + 1, devices.end());
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  constexpr int64_t domain_n = 24;
  const carbon::int64_3d domain_extent{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };

  BlockGrid block(system, domain_extent, predicate, "test_block_cuda");

  std::printf(
      "test_block_cuda: domain=%ldx%ldx%ld brick=%dx%dx%d -> %u retained blocks "
      "(%ld active voxels)\n",
      static_cast<long>(domain_extent.x), static_cast<long>(domain_extent.y),
      static_cast<long>(domain_extent.z), SB::MemX, SB::MemY, SB::MemZ, block.block_count(),
      static_cast<long>(block.active_voxel_count()));
  if (block.block_count() == 0U) {
    std::fprintf(stderr, "test_block_cuda: predicate did not retain any block\n");
    return 1;
  }

  // Cross-check: ceil(24 / 8)^3 = 27 candidate bricks. The 0.45-radius sphere should
  // retain every brick whose centre is closer than r + sqrt(3) * brick_diag/2 to the
  // domain centre, which for a 24-cube is "everything" -- but we'll lock the lower
  // bound at "at least one brick" rather than baking in an exact count and being
  // brittle to predicate tweaks.
  if (block.candidate_block_extent().x != 3 || block.candidate_block_extent().y != 3 ||
      block.candidate_block_extent().z != 3) {
    std::fprintf(stderr, "test_block_cuda: candidate extent (%ld,%ld,%ld) != (3,3,3)\n",
                 static_cast<long>(block.candidate_block_extent().x),
                 static_cast<long>(block.candidate_block_extent().y),
                 static_cast<long>(block.candidate_block_extent().z));
    return 1;
  }

  // Cross-check: counted active voxels must equal a fresh predicate sweep.
  int64_t expected_active = 0;
  for (int64_t z = 0; z < domain_extent.z; ++z) {
    for (int64_t y = 0; y < domain_extent.y; ++y) {
      for (int64_t x = 0; x < domain_extent.x; ++x) {
        if (predicate(carbon::int64_3d{x, y, z})) {
          ++expected_active;
        }
      }
    }
  }
  if (expected_active != block.active_voxel_count()) {
    std::fprintf(stderr, "test_block_cuda: active-voxel mismatch (expected %ld got %ld)\n",
                 static_cast<long>(expected_active), static_cast<long>(block.active_voxel_count()));
    return 1;
  }

  FieldT field = block.template newField<int32_t, 2>(carbon::MemorySide::Device, "test_field");

  // Allocate the device-origins pointer for the kernel via a small staging buffer.
  // Re-uploading is acceptable here because Block keeps a host-side copy too; the
  // alternative (exposing block.device_origins()) is reserved for the multi-GPU port
  // when origins become a partition-table entry.
  auto host_origins_alloc = carbon::Cpu::buffer<carbon::int32_3d>(block.block_count());
  for (uint32_t b = 0; b < block.block_count(); ++b) {
    host_origins_alloc.data()[b] = block.host_origins()[b];
  }
  const std::vector<size_t> origins_counts{static_cast<size_t>(block.block_count())};
  carbon::Buffer<carbon::int32_3d> dev_origins =
      system.gpu_group().buffer<carbon::int32_3d>(origins_counts);
  carbon::Stream stream = system.stream();
  system.async_copy(host_origins_alloc, dev_origins[0],
                    sizeof(carbon::int32_3d) * block.block_count(), stream);
  stream.wait();

  // Launch one CUDA block per retained brick, MemX*MemY*MemZ threads per block.
  cudaSetDevice(system.gpu_group().devices()[0].device_id());
  const dim3 grid(block.block_count(), 1, 1);
  const dim3 thread_block(SB::MemX, SB::MemY, SB::MemZ);
  fill_active_kernel<<<grid, thread_block>>>(block.span(), field.device_partition(),
                                             dev_origins[0].data());
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "test_block_cuda: cudaDeviceSynchronize -> %s\n", cudaGetErrorString(err));
    return 1;
  }

  // Copy the whole field back and check every voxel cell-by-cell. The field has
  // (block_count * VoxelCount * Components) elements. Inactive voxels (or out-of-domain
  // tail-voxels of edge bricks) must still carry their post-allocation zero value
  // (carbon::Buffer is allocated with cudaMalloc + we never wrote to them).
  const size_t total_elements = static_cast<size_t>(block.block_count()) * SB::VoxelCount * 2U;
  auto host_field_alloc = carbon::Cpu::buffer<int32_t>(total_elements);
  // Zero-init the host buffer so the read-back lands cleanly even on the unused tail.
  for (size_t i = 0; i < total_elements; ++i) {
    host_field_alloc.data()[i] = 0;
  }
  // Pre-zero the device side too so we test the kernel writes against a known background.
  cudaMemset(field.device_buffer()[0].data(), 0, total_elements * sizeof(int32_t));
  // Re-launch (the cudaMemset wiped the previous run).
  fill_active_kernel<<<grid, thread_block>>>(block.span(), field.device_partition(),
                                             dev_origins[0].data());
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "test_block_cuda: post-memset launch err -> %s\n",
                 cudaGetErrorString(err));
    return 1;
  }
  system.async_copy(field.device_buffer()[0], host_field_alloc, total_elements * sizeof(int32_t),
                    stream);
  stream.wait();

  int64_t active_checked = 0;
  int64_t inactive_checked = 0;
  for (uint32_t b = 0; b < block.block_count(); ++b) {
    const auto& bm = block.host_bitmasks()[b];
    const auto origin = block.host_origins()[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          for (int c = 0; c < 2; ++c) {
            const size_t lin =
                (static_cast<size_t>(b) * 2U + c) * SB::VoxelCount + x + 8 * y + 64 * z;
            const int32_t got = host_field_alloc.data()[lin];
            const bool active = bm.is_active(x, y, z);
            if (active) {
              const int32_t want = encode(origin.x + x, origin.y + y, origin.z + z, c);
              if (got != want) {
                std::fprintf(stderr,
                             "test_block_cuda: ACTIVE mismatch at b=%u (%d,%d,%d) c=%d "
                             "got %d want %d (origin=(%d,%d,%d))\n",
                             b, x, y, z, c, got, want, origin.x, origin.y, origin.z);
                return 1;
              }
              ++active_checked;
            } else {
              if (got != 0) {
                std::fprintf(stderr,
                             "test_block_cuda: INACTIVE leak at b=%u (%d,%d,%d) c=%d "
                             "got %d\n",
                             b, x, y, z, c, got);
                return 1;
              }
              ++inactive_checked;
            }
          }
        }
      }
    }
  }
  // active_checked counts both components (×2) so divide before comparing.
  if (active_checked / 2 != block.active_voxel_count()) {
    std::fprintf(stderr,
                 "test_block_cuda: per-component active count %ld / 2 != "
                 "block.active_voxel_count() %ld\n",
                 static_cast<long>(active_checked), static_cast<long>(block.active_voxel_count()));
    return 1;
  }
  std::printf(
      "test_block_cuda: OK (active voxels written: %ld, inactive zero-checked: "
      "%ld)\n",
      static_cast<long>(active_checked / 2), static_cast<long>(inactive_checked / 2));
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
