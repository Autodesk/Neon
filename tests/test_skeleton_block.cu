// Skeleton OCC rewrite over carbon::Block.
//
// Pins that the existing Skeleton::OverlapMode::Standard rewrite (originally
// validated for Dense grids) carries over to Block grids unchanged. The
// rewrite is grid-agnostic by design — it triggers on KernelKind::Halo +
// KernelKind::Compute + an AccessPattern::Stencil token on the field the
// halo writes. After item 1 wired Block::Field::make_halo_kernel through
// HaloBlock, all three signals are emitted by Block-built kernels too, so a
// Skeleton bound to a Block grid + OverlapMode::Standard should:
//
//   1. Auto-insert a halo (KernelLoader::read_stencil snapshots a
//      TypedHaloFactory<Block::Field> at user-add time).
//   2. Split the stencil into Internal + Boundary halves around the halo.
//   3. Move the halo onto a dedicated stream so Internal can run alongside.
//
// We verify functional convergence within FP tolerance (same race-aware
// tolerance test_block_overlapped_stencil uses — the underlying HaloBlock
// has a known cudaMemcpyPeerAsync cross-stream race that produces tiny
// drift over many iterations, see <carbon/halo.h>).
//
// Skipped (with a note printed) when fewer than 2 CUDA devices are visible.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/devices.h>
#include <carbon/halo_block.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/skeleton.h>
#include <carbon/system.h>

#include "test_domain.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<float, 1>;

[[nodiscard]] auto
sphere_predicate(int64_t domain_n,
                 const carbon::int64_3d& gc) -> bool {
  const double cx = (domain_n - 1) * 0.5;
  const double cy = (domain_n - 1) * 0.5;
  const double cz = (domain_n - 1) * 0.5;
  const double r = domain_n * 0.42;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

constexpr int kIters = 10;

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_INFO("test_skeleton_block: skipping (only {} CUDA device(s) visible, need >= 2)",
                    carbon::device_count());
    return 0;
  }

  auto devs = carbon::get_all_devices();
  carbon::System sys(carbon::detail::GpuGroup{std::move(devs)});
  const size_t n_gpus = sys.gpu_group().size();

  // Kept cubic as it grows, so the sphere scales with the domain instead of leaving the extra
  // brick rows empty.
  const int64_t domain_n = carbon::test::block_slab_z(48, n_gpus, SB::MemZ);
  const carbon::int64_3d global{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };

  BlockGrid block(sys, global, predicate, "skeleton_block");
  CARBON_LOG_INFO(
      "test_skeleton_block: domain={}^3, brick={}x{}x{}, gpus={}, retained {} blocks "
      "({} active voxels)",
      domain_n, SB::MemX, SB::MemY, SB::MemZ, n_gpus, block.block_count(),
      block.active_voxel_count());

  FieldT u_ref = block.template newField<float, 1>(carbon::MemorySide::Device, "u_ref");
  FieldT u_ref_next = block.template newField<float, 1>(carbon::MemorySide::Device, "u_ref_next");
  FieldT u_occ = block.template newField<float, 1>(carbon::MemorySide::Device, "u_occ");
  FieldT u_occ_next = block.template newField<float, 1>(carbon::MemorySide::Device, "u_occ_next");

  carbon::Stream stream = sys.stream();
  for (size_t g = 0; g < n_gpus; ++g) {
    const size_t total_slots_g =
        static_cast<size_t>(block.storage_block_count(g)) * static_cast<size_t>(SB::VoxelCount);
    if (total_slots_g == 0U) {
      continue;
    }
    sys.async_memset(u_ref.gpu_buffers()[g], 0, total_slots_g * sizeof(float), stream);
    sys.async_memset(u_ref_next.gpu_buffers()[g], 0, total_slots_g * sizeof(float), stream);
    sys.async_memset(u_occ.gpu_buffers()[g], 0, total_slots_g * sizeof(float), stream);
    sys.async_memset(u_occ_next.gpu_buffers()[g], 0, total_slots_g * sizeof(float), stream);
  }
  stream.wait();

  // Stencil-flagged compute: read_stencil(in) triggers the auto-halo
  // insertion + Internal/Boundary split under OverlapMode::Standard.
  auto build_step = [&](FieldT& u_in, FieldT& u_out) {
    return carbon::Kernel::compute(
        block,
        [&u_in, &u_out](carbon::KernelLoader& loader) {
          const auto in = loader.read_stencil(u_in);
          const auto out = loader.write(u_out);
          return [in, out] CARBON_HD(Span s, Idx idx) -> void {
            const auto xp = in.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0);
            const auto xm = in.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0);
            const auto yp = in.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0);
            const auto ym = in.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0);
            const auto zp = in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
            const auto zm = in.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
            if (!zm.valid) {
              out.write(s, idx, 0, +1.0F);
              return;
            }
            if (!zp.valid) {
              out.write(s, idx, 0, -1.0F);
              return;
            }
            if (!(xp.valid && xm.valid && yp.valid && ym.valid)) {
              out.write(s, idx, 0, 0.0F);
              return;
            }
            const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
            out.write(s, idx, 0, sum * (1.0F / 6.0F));
          };
        },
        "jacobi_block");
  };

  carbon::Kernel step_ref_a = build_step(u_ref, u_ref_next);
  carbon::Kernel step_ref_b = build_step(u_ref_next, u_ref);
  carbon::Kernel step_occ_a = build_step(u_occ, u_occ_next);
  carbon::Kernel step_occ_b = build_step(u_occ_next, u_occ);

  // Reference Skeleton: OverlapMode::None — auto-inserts the halo as a
  // serial predecessor on the same stream, no OCC split.
  carbon::Skeleton sk_ref(sys, carbon::Skeleton::Options(carbon::OverlapMode::None));
  for (int iter = 0; iter < kIters; ++iter) {
    sk_ref.add((iter % 2 == 0) ? step_ref_a : step_ref_b);
  }
  sk_ref.compile();
  sk_ref.run(carbon::DataView::Standard);
  sk_ref.wait();

  // OCC Skeleton: OverlapMode::Standard splits each stencil into
  // Internal + Boundary and moves the halo onto a dedicated stream.
  carbon::Skeleton sk_occ(sys, carbon::Skeleton::Options(carbon::OverlapMode::Standard));
  for (int iter = 0; iter < kIters; ++iter) {
    sk_occ.add((iter % 2 == 0) ? step_occ_a : step_occ_b);
  }
  sk_occ.compile();
  sk_occ.run(carbon::DataView::Standard);
  sk_occ.wait();

  FieldT& ref_final = (kIters % 2 == 0) ? u_ref : u_ref_next;
  FieldT& occ_final = (kIters % 2 == 0) ? u_occ : u_occ_next;
  auto host_ref = block.template download_to_host<float, 1>(ref_final, stream);
  auto host_occ = block.template download_to_host<float, 1>(occ_final, stream);
  stream.wait();

  size_t out_of_tol = 0;
  size_t active_checked = 0;
  float max_abs_diff = 0.0F;
  // Same race-aware tolerance test_block_overlapped_stencil uses; the OCC
  // rewrite reuses the same HaloBlock under the hood, so the same
  // cudaMemcpyPeerAsync cross-stream drift bound applies.
  constexpr float kTol = 5.0e-3F;
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_owned_g = block.block_count(g);
    const float* hr = host_ref[g].data();
    const float* ho = host_occ[g].data();
    for (uint32_t b = 0; b < n_owned_g; ++b) {
      const auto& bm = block.host_bitmasks(g)[b];
      for (int z = 0; z < SB::MemZ; ++z) {
        for (int y = 0; y < SB::MemY; ++y) {
          for (int x = 0; x < SB::MemX; ++x) {
            if (!bm.is_active(static_cast<uint8_t>(x), static_cast<uint8_t>(y),
                              static_cast<uint8_t>(z))) {
              continue;
            }
            const size_t off = (static_cast<size_t>(b) * SB::VoxelCount) +
                               (static_cast<size_t>(z) * SB::MemY * SB::MemX) +
                               (static_cast<size_t>(y) * SB::MemX) + static_cast<size_t>(x);
            const float a = hr[off];
            const float b_ = ho[off];
            const float diff = std::abs(a - b_);
            if (diff > max_abs_diff) {
              max_abs_diff = diff;
            }
            if (diff > kTol) {
              ++out_of_tol;
            }
            ++active_checked;
          }
        }
      }
    }
  }
  CARBON_LOG_INFO(
      "test_skeleton_block: checked {} active voxels, max |ref - occ| = {:.6e}, "
      "out-of-tol (>{:.1e}) = {}",
      active_checked, static_cast<double>(max_abs_diff), static_cast<double>(kTol), out_of_tol);
  if (out_of_tol > 0U) {
    CARBON_LOG_ERROR(
        "test_skeleton_block: {} active voxel(s) exceed FP tolerance kTol={} (max |diff| = {})",
        out_of_tol, static_cast<double>(kTol), static_cast<double>(max_abs_diff));
    return 1;
  }
  CARBON_LOG_INFO("test_skeleton_block: PASS");
  return 0;
}
