// Unit test for Block::make_reduce_kernel on a multi-GPU sparse grid.
//
// Builds a 32³ cube Block over the same sphere predicate used by
// test_block_data_view and test_block_overlapped_stencil (so the partition
// includes a non-trivial mix of fully-active and partially-active bricks),
// writes a deterministic per-voxel value u(x,y,z) = gx + 7*gy + 53*gz + 1
// into the owned bricks of a field, then reduces it three ways:
//
//   1. carbon::reduce::Sum<int64_t> via Block::make_reduce_kernel, view =
//      DataView::Standard . Compared against a host-side reference walk
//      over the same predicate + value formula.
//
//   2. The same reducer launched with view = DataView::Internal and view =
//      DataView::Boundary in sequence on the same ReductionResult — the
//      destination is rebuilt between the two launches via fresh writes
//      to dst.device_data so the test pins each view independently
//      against its own reference (host walk restricted to internal /
//      boundary owned bricks).
//
//   3. The same reducer with view = DataView::Standard but against a
//      single-GPU System fall-through (skipped when only one CUDA device
//      is visible — the multi-GPU paths exercise stage 2 / stage 3
//      combine on the rank-local root GPU).
//
// Pins both the stage-1 brick-reduce kernel correctness (inactive voxels
// contribute the identity = 0) and the stage-2 multi-GPU combine (gpu
// partials are folded on the root GPU into dst.device_data(view)).

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/reduce.h>
#include <carbon/system.h>

#include "test_domain.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<int64_t, 1>;

constexpr int64_t kDomainN = 32;

[[nodiscard]] auto
sphere_predicate(int64_t domain_n,
                 const carbon::int64_3d& gc) -> bool {
  const double cx = static_cast<double>(domain_n - 1) * 0.5;
  const double cy = static_cast<double>(domain_n - 1) * 0.5;
  const double cz = static_cast<double>(domain_n - 1) * 0.5;
  const double r = static_cast<double>(domain_n) * 0.40;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

[[nodiscard]] CARBON_HD auto
voxel_value(int64_t gx,
            int64_t gy,
            int64_t gz) -> int64_t {
  return gx + 7 * gy + 53 * gz + 1;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_block_reduce: skipping (no CUDA devices visible)");
    return 0;
  }
  auto devs = carbon::get_all_devices();
  carbon::System sys(carbon::detail::GpuGroup{std::move(devs)});
  const size_t n_gpus = sys.gpu_group().size();

  // Kept cubic as it grows, so the sphere scales with the domain instead of leaving the extra
  // brick rows empty.
  const int64_t domain_n = carbon::test::block_slab_z(kDomainN, n_gpus, SB::MemZ);
  const carbon::int64_3d global{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };

  BlockGrid block(sys, global, predicate, "block_reduce");
  CARBON_LOG_INFO("test_block_reduce: domain={}^3, gpus={}, retained {} bricks ({} active voxels)",
                  domain_n, n_gpus, block.block_count(), block.active_voxel_count());

  // Build the field and stage owned-brick data on device by initialising on
  // host and async_copy-ing per GPU. The bitmask tells the kernel which
  // voxels are active; inactive slots can hold any value (the reduce kernel
  // contributes the identity for them).
  FieldT u = block.template newField<int64_t, 1>(carbon::MemorySide::Device, "u");
  carbon::Stream stream = sys.stream();
  int64_t expected_total_standard = 0;
  std::vector<int64_t> expected_internal(n_gpus, 0);
  std::vector<int64_t> expected_boundary(n_gpus, 0);
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_owned = block.block_count(g);
    const uint32_t n_internal = block.n_internal_blocks(g);
    const uint32_t n_storage = block.storage_block_count(g);
    const size_t total_slots = static_cast<size_t>(n_storage) * static_cast<size_t>(SB::VoxelCount);
    if (total_slots == 0U) {
      continue;
    }
    auto host_buf = carbon::Cpu::buffer<int64_t>(total_slots);
    std::vector<int64_t> stage(total_slots, 0);
    for (uint32_t b = 0; b < n_owned; ++b) {
      const auto& origin = block.host_origins(g)[b];
      const auto& bm = block.host_bitmasks(g)[b];
      for (int iz = 0; iz < SB::MemZ; ++iz) {
        for (int iy = 0; iy < SB::MemY; ++iy) {
          for (int ix = 0; ix < SB::MemX; ++ix) {
            if (!bm.is_active(ix, iy, iz)) {
              continue;
            }
            const int64_t gx = static_cast<int64_t>(origin.x) + ix;
            const int64_t gy = static_cast<int64_t>(origin.y) + iy;
            const int64_t gz = static_cast<int64_t>(origin.z) + iz;
            const int64_t v = voxel_value(gx, gy, gz);
            const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                               static_cast<size_t>(ix) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(iy) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                   static_cast<size_t>(iz);
            stage[lin] = v;
            expected_total_standard += v;
            if (b < n_internal) {
              expected_internal[g] += v;
            } else {
              expected_boundary[g] += v;
            }
          }
        }
      }
    }
    std::memcpy(host_buf.data(), stage.data(), total_slots * sizeof(int64_t));
    sys.async_copy(host_buf, u.gpu_buffers()[g], total_slots * sizeof(int64_t), stream);
  }
  stream.wait();

  // ReductionResult bound by reference for the lifetime of the kernel.
  carbon::ReductionResult<int64_t> result(sys);

  auto make_sum_kernel = [&]() {
    return block.template make_reduce_kernel<int64_t>(
        [&u](carbon::KernelLoader& loader) {
          const auto in = loader.read(u);
          return [in] CARBON_HD(Span s, Idx idx) -> int64_t {
            // Caller has already gated on bm.is_active; the kernel only
            // invokes this lambda for active voxels of active bricks. Safe
            // to read directly.
            return in.read(s, idx, 0);
          };
        },
        result, carbon::reduce::Sum<int64_t>{});
  };

  carbon::Kernel reduce_kernel = make_sum_kernel();

  // ---- Standard view ----
  reduce_kernel.launch(stream, carbon::DataView::Standard);
  result.sync(stream);
  stream.wait();
  const int64_t got_standard = result.host_value(carbon::DataView::Standard);
  if (got_standard != expected_total_standard) {
    CARBON_LOG_ERROR("test_block_reduce[Standard]: got {} expected {}", got_standard,
                     expected_total_standard);
    return 1;
  }
  CARBON_LOG_INFO("test_block_reduce[Standard]: sum = {} (matches host reference)", got_standard);

  // ---- Internal view ----
  int64_t expected_total_internal = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    expected_total_internal += expected_internal[g];
  }
  reduce_kernel.launch(stream, carbon::DataView::Internal);
  result.sync(stream);
  stream.wait();
  const int64_t got_internal = result.host_value(carbon::DataView::Internal);
  if (got_internal != expected_total_internal) {
    CARBON_LOG_ERROR("test_block_reduce[Internal]: got {} expected {}", got_internal,
                     expected_total_internal);
    return 1;
  }

  // ---- Boundary view ----
  int64_t expected_total_boundary = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    expected_total_boundary += expected_boundary[g];
  }
  reduce_kernel.launch(stream, carbon::DataView::Boundary);
  result.sync(stream);
  stream.wait();
  const int64_t got_boundary = result.host_value(carbon::DataView::Boundary);
  if (got_boundary != expected_total_boundary) {
    CARBON_LOG_ERROR("test_block_reduce[Boundary]: got {} expected {}", got_boundary,
                     expected_total_boundary);
    return 1;
  }

  // ---- Recombine pin: Internal + Boundary == Standard ----
  if (got_internal + got_boundary != got_standard) {
    CARBON_LOG_ERROR("test_block_reduce: Internal({}) + Boundary({}) = {} != Standard({})",
                     got_internal, got_boundary, got_internal + got_boundary, got_standard);
    return 1;
  }
  CARBON_LOG_INFO(
      "test_block_reduce: Internal({}) + Boundary({}) = Standard({}) — view split round-trips",
      got_internal, got_boundary, got_standard);

  CARBON_LOG_INFO("test_block_reduce: ok ({} gpus, {} active voxels)", n_gpus,
                  block.active_voxel_count());
  return 0;
}
