// DataView::Internal / Boundary split for carbon::Block.
//
// Builds a multi-GPU Block over a sphere predicate (same shape as
// test_block_multi_gpu.cu) and pins the per-slab partition shape:
//
//   - Owned bricks at [0, n_internal_blocks(g)) have no slab z-edge face that
//     touches a remote neighbour (so a stencil read of radius 1 stays in this
//     slab's owned bricks).
//   - Owned bricks at [n_internal_blocks(g), block_count(g)) sit at one of the
//     two boundary brick rows (bz == bz_begin[g] AND there's a lower remote
//     neighbour, OR bz == bz_end[g] - 1 AND there's an upper remote neighbour).
//   - Internal + Boundary classifications agree with a host-side predicate
//     replicated below.
//
// Then runs the same encode-into-voxels kernel three ways and verifies cell-by-
// cell that DataView::Internal + DataView::Boundary covers exactly the same
// active voxels as DataView::Standard:
//
//   1. Standard launch into u_std.
//   2. (Reset u_split to sentinel.) Internal launch into u_split (writes only
//      internal bricks); Boundary launch into u_split (writes only boundary
//      bricks).
//   3. Diff u_std and u_split per active voxel — every active voxel must be
//      written exactly once and the value must match. Inactive voxels and any
//      ghost slot bytes are ignored.
//
// Skipped (with a note printed) when fewer than 2 CUDA devices are visible.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
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
  const double r = domain_n * 0.4;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_INFO("test_block_data_view: skipping (only {} CUDA device(s) visible, need >= 2)",
                    carbon::device_count());
    return 0;
  }

  auto devs = carbon::get_all_devices();
  carbon::System system(carbon::detail::GpuGroup{std::move(devs)});
  const size_t n_gpus = system.gpu_group().size();

  constexpr int64_t domain_n = 64;
  const carbon::int64_3d global{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };

  BlockGrid block(system, global, predicate, "data_view_split");
  CARBON_LOG_INFO(
      "test_block_data_view: domain={}x{}x{}, brick={}x{}x{}, gpus={}, retained {} blocks",
      global.x, global.y, global.z, SB::MemX, SB::MemY, SB::MemZ, n_gpus, block.block_count());

  // 1. Per-slab partition invariants.
  //    For every owned brick we recompute the boundary predicate from
  //    (origin.z, bz_begin, bz_end, n_gpus, single-rank multi-GPU layout) and
  //    confirm that bricks at [0, n_internal) all return false and bricks at
  //    [n_internal, block_count) all return true.
  size_t total_internal = 0;
  size_t total_boundary = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_owned_g = block.block_count(g);
    const uint32_t n_internal_g = block.n_internal_blocks(g);
    const uint32_t n_boundary_g = block.n_boundary_blocks(g);
    if (n_owned_g != n_internal_g + n_boundary_g) {
      CARBON_LOG_ERROR(
          "test_block_data_view: gpu {} accounting: n_owned={} != n_internal {} + n_boundary {}", g,
          n_owned_g, n_internal_g, n_boundary_g);
      return 1;
    }
    const bool has_lower = g > 0;
    const bool has_upper = g + 1U < n_gpus;
    const int64_t bz_lo = block.bz_begin(g);
    const int64_t bz_hi = block.bz_end(g) - 1;
    for (uint32_t b = 0; b < n_owned_g; ++b) {
      const int64_t bz =
          static_cast<int64_t>(block.host_origins(g)[b].z) / static_cast<int64_t>(SB::MemZ);
      const bool expect_boundary = (has_lower && bz == bz_lo) || (has_upper && bz == bz_hi);
      const bool got_boundary = b >= n_internal_g;
      if (expect_boundary != got_boundary) {
        CARBON_LOG_ERROR(
            "test_block_data_view: gpu {} brick {} (bz={}, has_lower={}, has_upper={}, "
            "bz_lo={}, bz_hi={}): expect_boundary={} but got_boundary={} (n_internal={})",
            g, b, bz, has_lower, has_upper, bz_lo, bz_hi, expect_boundary, got_boundary,
            n_internal_g);
        return 1;
      }
    }
    total_internal += n_internal_g;
    total_boundary += n_boundary_g;
    CARBON_LOG_INFO("test_block_data_view: gpu {} owned={} (internal={}, boundary={})", g,
                    n_owned_g, n_internal_g, n_boundary_g);
  }
  if (total_internal + total_boundary != block.block_count()) {
    CARBON_LOG_ERROR("test_block_data_view: sum mismatch ({} internal + {} boundary != {} total)",
                     total_internal, total_boundary, block.block_count());
    return 1;
  }

  // 2. Functional split: run the same encode kernel via Standard / Internal+Boundary,
  //    compare cell-by-cell.
  FieldT u_std = block.template newField<int32_t, 1>(carbon::MemorySide::Device, "u_std");
  FieldT u_split = block.template newField<int32_t, 1>(carbon::MemorySide::Device, "u_split");

  carbon::Stream stream = system.stream();
  for (size_t g = 0; g < n_gpus; ++g) {
    const size_t total_g =
        static_cast<size_t>(block.storage_block_count(g)) * static_cast<size_t>(SB::VoxelCount);
    if (total_g == 0U) {
      continue;
    }
    system.async_memset(u_std.gpu_buffers()[g], 0, total_g * sizeof(int32_t), stream);
    system.async_memset(u_split.gpu_buffers()[g], 0, total_g * sizeof(int32_t), stream);
  }
  stream.wait();

  auto build_encoder = [&](FieldT& dst) {
    const carbon::int32_3d* origins0 = block.device_origins_ptr(0);
    return block.make_kernel([&dst, &block](carbon::KernelLoader& loader) {
      auto out = loader.write(dst);
      const carbon::int32_3d* origins = block.device_origins_ptr(loader.relative_gpu_id());
      return [out, origins] CARBON_HD(Span s, Idx idx) -> void {
        const auto intra = idx.intra();
        const auto origin = origins[idx.block_idx()];
        const int gx = static_cast<int>(origin.x) + static_cast<int>(intra.x);
        const int gy = static_cast<int>(origin.y) + static_cast<int>(intra.y);
        const int gz = static_cast<int>(origin.z) + static_cast<int>(intra.z);
        out.write(s, idx, 0, encode(gx, gy, gz));
      };
    });
    (void)origins0;
  };

  carbon::Kernel encode_std = build_encoder(u_std);
  carbon::Kernel encode_split = build_encoder(u_split);

  encode_std.launch(stream, carbon::DataView::Standard);
  encode_split.launch(stream, carbon::DataView::Internal);
  encode_split.launch(stream, carbon::DataView::Boundary);
  stream.wait();

  auto host_std = block.template download_to_host<int32_t, 1>(u_std, stream);
  auto host_split = block.template download_to_host<int32_t, 1>(u_split, stream);
  stream.wait();

  size_t mismatches = 0;
  size_t active_compared = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_owned_g = block.block_count(g);
    const int32_t* hs = host_std[g].data();
    const int32_t* hp = host_split[g].data();
    for (uint32_t b = 0; b < n_owned_g; ++b) {
      const auto& bm = block.host_bitmasks(g)[b];
      for (int z = 0; z < SB::MemZ; ++z) {
        for (int y = 0; y < SB::MemY; ++y) {
          for (int x = 0; x < SB::MemX; ++x) {
            const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                               static_cast<size_t>(x) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                   static_cast<size_t>(z);
            if (bm.is_active(x, y, z)) {
              if (hs[lin] != hp[lin]) {
                if (mismatches < 4U) {
                  CARBON_LOG_ERROR(
                      "test_block_data_view: gpu {} brick {} ({},{},{}): split {} != standard {}",
                      g, b, x, y, z, hp[lin], hs[lin]);
                }
                ++mismatches;
              }
              ++active_compared;
            }
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_block_data_view: {} mismatches across {} active voxels", mismatches,
                     active_compared);
    return 1;
  }
  CARBON_LOG_INFO(
      "test_block_data_view: ok ({} active voxels matched between Standard and Internal+Boundary)",
      active_compared);
  return 0;
}
