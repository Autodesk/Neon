// Overlapped-stencil parity test for carbon::Block.
//
// Builds a multi-GPU Block over a sphere predicate (same shape as
// test_block_data_view.cu) and runs N iterations of a 7-point Jacobi twice:
//
//   1. Reference path: explicit halo() + step.launch(stream, DataView::Standard).
//   2. Overlapped path: carbon::OverlappedStencil with the SAME halo Kernel
//      and the SAME compute Kernel, which under the hood splits the launch
//      into halo on a halo stream || Internal on a compute stream, then
//      Boundary fenced behind the halo via an event.
//
// Pins:
//   - Block::Field::make_halo_kernel returns a real HaloBlock-backed Kernel
//     (it used to be a no-op anchor for default-constructed fields; with the
//     back-reference wired by Block::newField it now routes through
//     Kernel::halo_block).
//   - The Internal + Boundary split applied to a stencil converges to the
//     same answer as a single Standard launch within FP tolerance. Strict
//     bit-for-bit equality is NOT enforced because the multi-GPU halo has
//     a documented cross-stream race (see <carbon/halo.h> "Synchronisation
//     contract"): the halo on @c stream[dest_g] does not wait for the
//     previous compute on @c stream[src_g] , so the bytes the halo reads
//     from the source GPU are deterministic in their final value but not
//     bit-wise consistent in their @b transient value across launches.
//     The same race exists when a user does the halo + step pairing by hand.
//     Iter 1, 2, 3 match bit-for-bit (the initial all-zero u_in masks the
//     race); drift kicks in starting around iter 7 when neighbour reads
//     touch slowly-converging values. A bounded absolute tolerance is what
//     we can verify here; the convergence pin (identical RMS over hundreds
//     of iterations) lives in the Block branch of jacobi_unified.
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
#include <carbon/overlapped_stencil.h>
#include <carbon/system.h>

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

constexpr int kIters = 20;

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_INFO(
        "test_block_overlapped_stencil: skipping (only {} CUDA device(s) visible, need >= 2)",
        carbon::device_count());
    return 0;
  }

  auto devs = carbon::get_all_devices();
  carbon::System sys(carbon::detail::GpuGroup{std::move(devs)});
  const size_t n_gpus = sys.gpu_group().size();

  constexpr int64_t domain_n = 64;
  const carbon::int64_3d global{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };

  BlockGrid block(sys, global, predicate, "overlapped_stencil_parity");
  CARBON_LOG_INFO(
      "test_block_overlapped_stencil: domain={}x{}x{}, brick={}x{}x{}, gpus={}, retained "
      "{} blocks ({} active voxels)",
      global.x, global.y, global.z, SB::MemX, SB::MemY, SB::MemZ, n_gpus, block.block_count(),
      block.active_voxel_count());

  // Two pairs of ping-pong fields: one fed to the reference path, one fed to
  // the overlapped path. Both start with identical bytes so any divergence is
  // attributable to the launch path.
  FieldT u_ref = block.template newField<float, 1>(carbon::MemorySide::Device, "u_ref");
  FieldT u_ref_next = block.template newField<float, 1>(carbon::MemorySide::Device, "u_ref_next");
  FieldT u_ovl = block.template newField<float, 1>(carbon::MemorySide::Device, "u_ovl");
  FieldT u_ovl_next = block.template newField<float, 1>(carbon::MemorySide::Device, "u_ovl_next");

  carbon::Stream stream = sys.stream();
  for (size_t g = 0; g < n_gpus; ++g) {
    const size_t total_slots_g =
        static_cast<size_t>(block.storage_block_count(g)) * static_cast<size_t>(SB::VoxelCount);
    if (total_slots_g == 0U) {
      continue;
    }
    sys.async_memset(u_ref.gpu_buffers()[g], 0, total_slots_g * sizeof(float), stream);
    sys.async_memset(u_ref_next.gpu_buffers()[g], 0, total_slots_g * sizeof(float), stream);
    sys.async_memset(u_ovl.gpu_buffers()[g], 0, total_slots_g * sizeof(float), stream);
    sys.async_memset(u_ovl_next.gpu_buffers()[g], 0, total_slots_g * sizeof(float), stream);
  }
  stream.wait();

  // Halo Kernels for both fields in both copies of the ping-pong pair.
  carbon::Kernel halo_u_ref = u_ref.make_halo_kernel("halo_u_ref");
  carbon::Kernel halo_u_ref_next = u_ref_next.make_halo_kernel("halo_u_ref_next");
  carbon::Kernel halo_u_ovl = u_ovl.make_halo_kernel("halo_u_ovl");
  carbon::Kernel halo_u_ovl_next = u_ovl_next.make_halo_kernel("halo_u_ovl_next");

  if (halo_u_ref.kind() != carbon::KernelKind::Halo) {
    CARBON_LOG_ERROR(
        "test_block_overlapped_stencil: make_halo_kernel did not return a Halo Kernel "
        "(got kind={}); back-reference plumbing may not be wired",
        static_cast<int>(halo_u_ref.kind()));
    return 1;
  }

  auto build_step = [&](FieldT& u_in, FieldT& u_out) {
    return block.make_kernel([&u_in, &u_out](carbon::KernelLoader& loader) {
      const auto in = loader.read(u_in);
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
    });
  };

  carbon::Kernel step_ref_a = build_step(u_ref, u_ref_next);
  carbon::Kernel step_ref_b = build_step(u_ref_next, u_ref);
  carbon::Kernel step_ovl_a = build_step(u_ovl, u_ovl_next);
  carbon::Kernel step_ovl_b = build_step(u_ovl_next, u_ovl);

  // Reference path: explicit halo + Standard launch on one stream, per iter.
  for (int iter = 0; iter < kIters; ++iter) {
    const carbon::Kernel& halo_k = (iter % 2 == 0) ? halo_u_ref : halo_u_ref_next;
    const carbon::Kernel& step_k = (iter % 2 == 0) ? step_ref_a : step_ref_b;
    halo_k.launch(stream, carbon::DataView::Standard);
    step_k.launch(stream, carbon::DataView::Standard);
  }
  stream.wait();

  // Overlapped path: OverlappedStencil step (halo || Internal, then Boundary).
  carbon::OverlappedStencil overlap(sys);
  for (int iter = 0; iter < kIters; ++iter) {
    const carbon::Kernel& halo_k = (iter % 2 == 0) ? halo_u_ovl : halo_u_ovl_next;
    const carbon::Kernel& step_k = (iter % 2 == 0) ? step_ovl_a : step_ovl_b;
    overlap.step(halo_k, step_k);
  }
  overlap.wait();

  // Pull both final fields back and compare per active voxel.
  FieldT& ref_final = (kIters % 2 == 0) ? u_ref : u_ref_next;
  FieldT& ovl_final = (kIters % 2 == 0) ? u_ovl : u_ovl_next;
  auto host_ref = block.template download_to_host<float, 1>(ref_final, stream);
  auto host_ovl = block.template download_to_host<float, 1>(ovl_final, stream);
  stream.wait();

  size_t out_of_tol = 0;
  size_t active_checked = 0;
  float max_abs_diff = 0.0F;
  double sumsq_ref = 0.0;
  double sumsq_ovl = 0.0;
  // Per-voxel ref values are bounded in @c [-1, 1] (Dirichlet BC), so a fixed
  // absolute tolerance is more useful than a relative one for the
  // slowly-converging interior. 5e-3 is comfortably above the worst-case
  // halo-race FP drift we observe (~5e-4 over 20 iters on 2× A100) but
  // well below the per-voxel signal magnitude (~0.5 RMS).
  constexpr float kTol = 5.0e-3F;
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t n_owned_g = block.block_count(g);
    const float* hr = host_ref[g].data();
    const float* ho = host_ovl[g].data();
    for (uint32_t b = 0; b < n_owned_g; ++b) {
      const auto& bm = block.host_bitmasks(g)[b];
      for (int z = 0; z < SB::MemZ; ++z) {
        for (int y = 0; y < SB::MemY; ++y) {
          for (int x = 0; x < SB::MemX; ++x) {
            if (!bm.is_active(x, y, z)) {
              continue;
            }
            const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                               static_cast<size_t>(x) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                   static_cast<size_t>(z);
            const float vr = hr[lin];
            const float vo = ho[lin];
            const float diff = (vr > vo) ? (vr - vo) : (vo - vr);
            if (diff > max_abs_diff) {
              max_abs_diff = diff;
            }
            if (diff > kTol) {
              if (out_of_tol < 4U) {
                CARBON_LOG_ERROR(
                    "test_block_overlapped_stencil: gpu {} brick {} ({},{},{}): ref={} "
                    "overlap={} diff={} > tol={}",
                    g, b, x, y, z, vr, vo, diff, kTol);
              }
              ++out_of_tol;
            }
            sumsq_ref += static_cast<double>(vr) * static_cast<double>(vr);
            sumsq_ovl += static_cast<double>(vo) * static_cast<double>(vo);
            ++active_checked;
          }
        }
      }
    }
  }
  const double rms_ref =
      (active_checked == 0U) ? 0.0 : std::sqrt(sumsq_ref / static_cast<double>(active_checked));
  const double rms_ovl =
      (active_checked == 0U) ? 0.0 : std::sqrt(sumsq_ovl / static_cast<double>(active_checked));
  const double rms_diff = (rms_ref > rms_ovl) ? (rms_ref - rms_ovl) : (rms_ovl - rms_ref);
  if (out_of_tol > 0U) {
    CARBON_LOG_ERROR(
        "test_block_overlapped_stencil: {} voxels exceed tolerance {} (max abs diff = {})",
        out_of_tol, kTol, max_abs_diff);
    return 1;
  }
  // RMS pin: the global scalar both paths converge towards should match much
  // tighter than the per-voxel tolerance, since FP noise in individual
  // voxels averages out into a near-zero RMS bias.
  if (rms_diff > 1.0e-4) {
    CARBON_LOG_ERROR(
        "test_block_overlapped_stencil: ref RMS {} vs overlap RMS {} differ by {} > 1e-4", rms_ref,
        rms_ovl, rms_diff);
    return 1;
  }
  CARBON_LOG_INFO(
      "test_block_overlapped_stencil: ok ({} active voxels match within abs tol {}; "
      "max abs diff = {}; ref RMS = {}, overlap RMS = {}, |Δ RMS| = {})",
      active_checked, kTol, max_abs_diff, rms_ref, rms_ovl, rms_diff);
  return 0;
}
