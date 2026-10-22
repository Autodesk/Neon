// Multi-GPU smoke test for the carbon::Block partitioner. Builds a Block over a
// sphere predicate on a 64-cube domain with the brick z-axis split across all
// visible GPUs, then exercises Block::make_kernel on every device:
//
//   - Aggregate retained-block count across GPUs equals the single-GPU reference.
//   - Each GPU's [bz_begin, bz_end) range matches a fresh partition_extent_z computed
//     here.
//   - For every retained brick on every GPU, host_origins(g)[b].z falls in
//     [bz_begin(g) * MemZ, bz_end(g) * MemZ).
//   - Active-voxel count summed over the per-GPU bitmasks equals the total reported
//     by Block::active_voxel_count() and the dense host predicate sweep.
//   - A compute kernel through Block::make_kernel writes
//     encode(global_x, global_y, global_z) into every active voxel; reading back
//     each GPU's slab and decoding confirms cell-by-cell agreement, while inactive
//     voxels remain zero. The kernel uses Block::device_origins_ptr(g) to recover
//     global coordinates without any extra side-channel buffer (so the multi-GPU
//     accessor is exercised end-to-end).
//
// Skipped (with a note printed) when fewer than 2 CUDA devices are visible.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<int32_t, 1>;

[[nodiscard]] CARBON_HD constexpr auto
encode(int gx,
       int gy,
       int gz) noexcept -> int32_t {
  return ((gx + 1) * 73856093) ^ ((gy + 1) * 19349663) ^ ((gz + 1) * 83492791);
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

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.size() < 2U) {
    std::printf("test_block_multi_gpu: fewer than 2 CUDA devices visible (%zu) — skipping\n",
                devices.size());
    return 0;
  }
  const size_t n_gpus = devices.size();
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  constexpr int64_t domain_n = 64;
  const carbon::int64_3d domain_extent{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };
  BlockGrid block(system, domain_extent, predicate, "test_block_multi_gpu");
  std::printf(
      "test_block_multi_gpu: domain=%ldx%ldx%ld brick=%dx%dx%d gpus=%zu -> %u retained "
      "blocks (%ld active voxels)\n",
      static_cast<long>(domain_extent.x), static_cast<long>(domain_extent.y),
      static_cast<long>(domain_extent.z), SB::MemX, SB::MemY, SB::MemZ, n_gpus, block.block_count(),
      static_cast<long>(block.active_voxel_count()));

  // Cross-check: per-GPU block_count sums to total and partitioning is contiguous in
  // bz across GPUs (no gaps, no overlaps, covers [0, candidate_block_extent.z)).
  uint32_t sum_per_gpu = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    sum_per_gpu += block.block_count(g);
  }
  if (sum_per_gpu != block.block_count()) {
    std::fprintf(stderr, "test_block_multi_gpu: per-GPU block count sum %u != total %u\n",
                 sum_per_gpu, block.block_count());
    return 1;
  }
  int64_t prev_end = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    if (block.bz_begin(g) != prev_end) {
      std::fprintf(stderr,
                   "test_block_multi_gpu: bz_begin(%zu) = %ld, expected %ld (gap or overlap)\n", g,
                   static_cast<long>(block.bz_begin(g)), static_cast<long>(prev_end));
      return 1;
    }
    if (block.bz_end(g) < block.bz_begin(g)) {
      std::fprintf(stderr, "test_block_multi_gpu: bz_end(%zu) = %ld < bz_begin = %ld\n", g,
                   static_cast<long>(block.bz_end(g)), static_cast<long>(block.bz_begin(g)));
      return 1;
    }
    prev_end = block.bz_end(g);
  }
  if (prev_end != block.candidate_block_extent().z) {
    std::fprintf(stderr, "test_block_multi_gpu: bz_end(N-1) = %ld != candidate_extent.z = %ld\n",
                 static_cast<long>(prev_end), static_cast<long>(block.candidate_block_extent().z));
    return 1;
  }
  // Each retained brick's origin.z must lie in its GPU's bz range.
  for (size_t g = 0; g < n_gpus; ++g) {
    const int64_t lo_z = block.bz_begin(g) * SB::MemZ;
    const int64_t hi_z = block.bz_end(g) * SB::MemZ;
    for (uint32_t b = 0; b < block.block_count(g); ++b) {
      const auto origin = block.host_origins(g)[b];
      if (origin.z < lo_z || origin.z >= hi_z) {
        std::fprintf(stderr,
                     "test_block_multi_gpu: brick %u on GPU %zu has origin.z = %d outside "
                     "[%ld, %ld)\n",
                     b, g, origin.z, static_cast<long>(lo_z), static_cast<long>(hi_z));
        return 1;
      }
    }
  }

  // Active-voxel cross-check against a fresh dense predicate sweep.
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
    std::fprintf(stderr, "test_block_multi_gpu: active voxel mismatch (block=%ld vs sweep=%ld)\n",
                 static_cast<long>(block.active_voxel_count()), static_cast<long>(expected_active));
    return 1;
  }

  // Allocate a multi-GPU field, zero-init both slabs, run init kernel.
  FieldT u = block.template newField<int32_t, 1>(carbon::MemorySide::Device, "u");
  carbon::Stream stream = system.stream();
  for (size_t g = 0; g < n_gpus; ++g) {
    const size_t total_g =
        static_cast<size_t>(block.block_count(g)) * static_cast<size_t>(SB::VoxelCount);
    if (total_g == 0U) {
      continue;
    }
    system.async_memset(u.gpu_buffers()[g], 0, total_g * sizeof(int32_t), stream);
  }
  stream.wait();

  carbon::Kernel init = block.make_kernel([u, &block](carbon::KernelLoader& loader) {
    auto field = loader.write(u);
    const carbon::int32_3d* origins_dev = block.device_origins_ptr(loader.relative_gpu_id());
    return [field, origins_dev] CARBON_HD(Span s, Idx idx) -> void {
      const auto intra = idx.intra();
      const carbon::int32_3d origin = origins_dev[idx.block_idx()];
      const int gx = origin.x + intra.x;
      const int gy = origin.y + intra.y;
      const int gz = origin.z + intra.z;
      field.write(s, idx, 0, encode(gx, gy, gz));
    };
  });

  init.launch(stream);
  stream.wait();

  // Read every GPU's slab back and verify cell-by-cell.
  // Use Block::download_to_host for the per-GPU readback so the helper is
  // exercised here too. One async_copy per GPU is queued on `stream`; the call
  // returns owned-only buffers (no ghost slots), which is what the verification
  // loop wants.
  auto host_per_gpu = block.template download_to_host<int32_t, 1>(u, stream);
  stream.wait();
  int64_t active_checked = 0;
  int64_t inactive_checked = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_blocks_g = block.block_count(g);
    if (n_blocks_g == 0U) {
      continue;
    }
    const int32_t* host = host_per_gpu[g].data();
    for (uint32_t b = 0; b < n_blocks_g; ++b) {
      const auto& bm = block.host_bitmasks(g)[b];
      const auto origin = block.host_origins(g)[b];
      for (int z = 0; z < SB::MemZ; ++z) {
        for (int y = 0; y < SB::MemY; ++y) {
          for (int x = 0; x < SB::MemX; ++x) {
            const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                               static_cast<size_t>(x) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                   static_cast<size_t>(z);
            const int32_t got = host[lin];
            if (bm.is_active(x, y, z)) {
              const int32_t want = encode(origin.x + x, origin.y + y, origin.z + z);
              if (got != want) {
                std::fprintf(stderr,
                             "test_block_multi_gpu: ACTIVE mismatch g=%zu b=%u (%d,%d,%d) "
                             "got %d want %d origin=(%d,%d,%d)\n",
                             g, b, x, y, z, got, want, origin.x, origin.y, origin.z);
                return 1;
              }
              ++active_checked;
            } else {
              if (got != 0) {
                std::fprintf(stderr,
                             "test_block_multi_gpu: INACTIVE leak g=%zu b=%u (%d,%d,%d) got %d\n",
                             g, b, x, y, z, got);
                return 1;
              }
              ++inactive_checked;
            }
          }
        }
      }
    }
  }
  if (active_checked != block.active_voxel_count()) {
    std::fprintf(stderr, "test_block_multi_gpu: active checked %ld != active_voxel_count %ld\n",
                 static_cast<long>(active_checked), static_cast<long>(block.active_voxel_count()));
    return 1;
  }
  std::printf("test_block_multi_gpu: OK (active=%ld inactive_zero=%ld)\n",
              static_cast<long>(active_checked), static_cast<long>(inactive_checked));
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
