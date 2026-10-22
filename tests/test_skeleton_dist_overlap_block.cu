// MPI cross-rank Skeleton OverlapMode parity test for carbon::Block.
//
// Closes the Block half of P1.δ in
// docs/audit_2026_05_fourth_pass.md (sibling gap to the third-pass
// `tests/test_skeleton_dist_overlap.cu` Dense closure and the
// 2026-05-18 fourth-pass DenseDisg sibling
// `tests/test_skeleton_dist_overlap_disg.cu`). The single-rank Block
// OCC tests (`test_skeleton_block.cu`,
// `test_skeleton_block_extended.cu`) lock the apply_occ_ rewrite
// shape on a single process; this file extends the matrix to MPI so
// a regression in HaloBlock's tokens shape under {Standard, Extended,
// TwoWayExtended} — e.g. the back-walk picking up the wrong
// predecessor when the auto-halo shape differs from `Halo::tokens()`,
// or the cross-rank DistTransfer chunk being scheduled outside the
// OCC stream graph — is no longer silent.
//
// Same pipeline shape as `test_skeleton_dist_overlap_disg.cu`:
// a ping-pong Jacobi (halo_a + stencil_ab + halo_b + stencil_ba)
// with halo kernels produced by `Block::Field::make_halo_kernel` —
// the public adapter that wraps `HaloBlock<T, Components, SBlock>`
// as a `Kernel` of `Kind::Halo`. The auto-halo path in
// `Skeleton::compile` recognises `Kind::Halo` regardless of grid
// family, so apply_occ_ sees the same token-set as on Dense /
// DenseDisg and the OCC rewrite must produce bit-equal results.
//
// Bit-equality is the post-condition: under one-rank-per-GPU the
// halo path is `DistTransfer<T>` only (no intra-process
// `cudaMemcpyPeerAsync`), so the cross-stream race that motivates
// the kTol=5e-3 tolerance in the rank-local `test_skeleton_block.cu`
// does not apply here. Every kernel is deterministic and the OCC
// rewrite must not change the numerical sequence (it only
// reschedules across streams).
//
// `predicate=true` keeps the brick-active map regular and identical
// across ranks (every voxel active), so the host-side reference and
// the per-rank owned-cell slice are easy to reason about.
//
// Skipped (returns 0) when world_size < 2 or no GPUs are visible.
// The rank-local Block OCC path already has dedicated coverage in
// `test_skeleton_block.cu` and `test_skeleton_block_extended.cu`;
// running with -n 1 would just duplicate those fixtures.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/distributed.h>
#include <carbon/halo_block.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/neighbour_value.h>
#include <carbon/skeleton.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<float, 1>;

constexpr int kCycles = 4;  // each cycle = halo_a + stencil_ab + halo_b + stencil_ba

[[nodiscard]] CARBON_HD auto
encode(int64_t gx,
       int64_t gy,
       int64_t gz) -> float {
  return static_cast<float>(gx + 11 * gy + 113 * gz);
}

// Linear voxel offset inside a Block::Field per-GPU buffer. Block stores
// owned bricks first (b in [0, n_owned)), then ghost slots (b in
// [n_owned, n_storage)). Each brick is `SB::VoxelCount` cells stored in
// (z, y, x)-major order.
[[nodiscard]] auto
brick_voxel_offset(uint32_t b,
                   int ix,
                   int iy,
                   int iz) -> size_t {
  return (static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount)) +
         (static_cast<size_t>(iz) * static_cast<size_t>(SB::MemY) * static_cast<size_t>(SB::MemX)) +
         (static_cast<size_t>(iy) * static_cast<size_t>(SB::MemX)) + static_cast<size_t>(ix);
}

// Same Jacobi step lambda as `test_skeleton_dist_overlap_disg.cu`,
// retargeted to `Block::{Span, Idx}`. `loader.read_stencil(fin)` is
// what triggers the auto-halo insertion in `Skeleton::compile`; the
// lambda body matches the DenseDisg sibling so any divergence in OCC
// behaviour is purely a Block-side thing.
[[nodiscard]] auto
build_jacobi_step(BlockGrid& block,
                  FieldT& fin,
                  FieldT& fout,
                  std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      block,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HD(Span s, Idx idx) -> void {
          const carbon::NeighbourValue<float> xp =
              in.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0);
          const carbon::NeighbourValue<float> xm =
              in.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0);
          const carbon::NeighbourValue<float> yp =
              in.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0);
          const carbon::NeighbourValue<float> ym =
              in.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0);
          const carbon::NeighbourValue<float> zp =
              in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
          const carbon::NeighbourValue<float> zm =
              in.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
          const float self = in.read(s, idx, 0);
          const bool all_valid =
              xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid;
          if (!all_valid) {
            out.write(s, idx, 0, self);
            return;
          }
          const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
          out.write(s, idx, 0, sum * (1.0F / 6.0F));
        };
      },
      std::move(name));
}

// Seed the per-GPU host mirror with `encode(gx, gy, gz)` on every
// owned voxel and zero on ghost slots. With predicate=true every
// voxel inside an owned brick is active, so we walk the dense
// (ix, iy, iz) cube of every owned brick and project to global
// coords through the brick origin. Ghost slots stay at zero — the
// halo kernel overwrites them every iteration on every rank with a
// neighbour brick, so any leftover non-zero would survive a
// mis-scheduled OCC rewrite as an owned-cell mismatch on the next
// stencil step.
void
init_field(const BlockGrid& block,
           const carbon::System& sys,
           FieldT& f) {
  const size_t n_gpus = sys.gpu_group().size();
  for (size_t g = 0; g < n_gpus; ++g) {
    float* h = f.cpu_buffers()[g].data();
    const size_t storage_voxels =
        static_cast<size_t>(block.storage_block_count(g)) * static_cast<size_t>(SB::VoxelCount);
    for (size_t i = 0; i < storage_voxels; ++i) {
      h[i] = 0.0F;
    }
    const uint32_t n_owned = block.block_count(static_cast<uint32_t>(g));
    for (uint32_t b = 0; b < n_owned; ++b) {
      const auto& origin = block.host_origins(static_cast<uint32_t>(g))[b];
      for (int iz = 0; iz < SB::MemZ; ++iz) {
        for (int iy = 0; iy < SB::MemY; ++iy) {
          for (int ix = 0; ix < SB::MemX; ++ix) {
            const int64_t gx = static_cast<int64_t>(origin.x) + ix;
            const int64_t gy = static_cast<int64_t>(origin.y) + iy;
            const int64_t gz = static_cast<int64_t>(origin.z) + iz;
            h[brick_voxel_offset(b, ix, iy, iz)] = encode(gx, gy, gz);
          }
        }
      }
    }
  }
}

[[nodiscard]] auto
upload(const carbon::System& sys,
       FieldT& f,
       const carbon::Stream& stream) -> bool {
  try {
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap_block: upload failed: {}", e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
download(const carbon::System& sys,
         FieldT& f,
         const carbon::Stream& stream) -> bool {
  try {
    carbon::Transfer<float>(sys, f.gpu_buffers(), f.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap_block: download failed: {}", e.what());
    return false;
  }
  return true;
}

// Run kCycles iterations of the ping-pong Jacobi pipeline at the
// given OverlapMode. Reads u_a / u_b on input, leaves the final
// result in u_a (kCycles is even -> last write lands in u_a).
[[nodiscard]] auto
run_pipeline(const carbon::System& sys,
             BlockGrid& block,
             FieldT& u_a,
             FieldT& u_b,
             carbon::OverlapMode overlap,
             const carbon::Stream& stream,
             const char* tag) -> bool {
  carbon::Skeleton sk(sys, stream, carbon::Skeleton::Options{overlap, /*strict=*/false});
  // make_halo_kernel returns a Kernel of Kind::Halo wrapping a
  // HaloBlock<float, 1, SB> over the field's gpu_buffers. apply_occ_
  // matches on Kind::Halo so the OCC rewrite shape is the same as on
  // Dense / DenseDisg.
  sk.add(u_a.make_halo_kernel("halo_a"));
  sk.add(build_jacobi_step(block, u_a, u_b, "step_ab"));
  sk.add(u_b.make_halo_kernel("halo_b"));
  sk.add(build_jacobi_step(block, u_b, u_a, "step_ba"));

  try {
    for (int it = 0; it < kCycles; ++it) {
      sk.run();
    }
    sk.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap_block[{}]: run threw: {}", tag, e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
overlap_mode_name(carbon::OverlapMode mode) -> const char* {
  switch (mode) {
    case carbon::OverlapMode::None:
      return "None";
    case carbon::OverlapMode::Standard:
      return "Standard";
    case carbon::OverlapMode::Extended:
      return "Extended";
    case carbon::OverlapMode::TwoWayExtended:
      return "TwoWayExtended";
  }
  return "?";
}

// Snapshot owned-cell values into a flat vector. Walking order is
// (gpu, brick, iz, iy, ix) so the same walk on the variant run
// produces a directly-comparable sequence.
[[nodiscard]] auto
snapshot_owned(const BlockGrid& block,
               const carbon::System& sys,
               const FieldT& f) -> std::vector<float> {
  std::vector<float> out;
  const size_t n_gpus = sys.gpu_group().size();
  for (size_t g = 0; g < n_gpus; ++g) {
    const float* h = f.cpu_buffers()[g].data();
    const uint32_t n_owned = block.block_count(static_cast<uint32_t>(g));
    out.reserve(out.size() + static_cast<size_t>(n_owned) * static_cast<size_t>(SB::VoxelCount));
    for (uint32_t b = 0; b < n_owned; ++b) {
      for (int iz = 0; iz < SB::MemZ; ++iz) {
        for (int iy = 0; iy < SB::MemY; ++iy) {
          for (int ix = 0; ix < SB::MemX; ++ix) {
            out.push_back(h[brick_voxel_offset(b, ix, iy, iz)]);
          }
        }
      }
    }
  }
  return out;
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_skeleton_dist_overlap_block: world_size={} < 2, skipping",
                    distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_skeleton_dist_overlap_block: no GPU devices, skipping");
    return EXIT_SUCCESS;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap_block: expected gpu_group size 1, got {}",
                     sys.gpu_group().size());
    return EXIT_FAILURE;
  }

  // Two brick-rows in z per rank so each rank has interior + boundary
  // owned bricks against the cross-rank halo. Every voxel active so
  // the brick layout is regular and identical across ranks.
  const int64_t global_z = static_cast<int64_t>(world_size) * static_cast<int64_t>(SB::MemZ) * 2;
  const carbon::int64_3d global{static_cast<int64_t>(SB::MemX) * 2,
                                static_cast<int64_t>(SB::MemY) * 2, global_z};
  const auto predicate = [](const carbon::int64_3d&) -> bool {
    return true;
  };

  BlockGrid block(sys, global, predicate, "skeleton_dist_overlap_block");
  if (block.gpu_count() != 1U) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap_block[rank={}]: gpu_count == {} (expected 1)",
                     rank, block.gpu_count());
    return EXIT_FAILURE;
  }

  // ---- Reference run (OverlapMode::None) ----
  auto u_a_ref = block.template newField<float, 1>(carbon::MemorySide::HostDevice, "u_a_ref");
  auto u_b_ref = block.template newField<float, 1>(carbon::MemorySide::HostDevice, "u_b_ref");
  init_field(block, sys, u_a_ref);
  init_field(block, sys, u_b_ref);

  carbon::Stream stream_ref = sys.stream();
  if (!upload(sys, u_a_ref, stream_ref) || !upload(sys, u_b_ref, stream_ref)) {
    return EXIT_FAILURE;
  }
  if (!run_pipeline(sys, block, u_a_ref, u_b_ref, carbon::OverlapMode::None, stream_ref,
                    "None-reference")) {
    return EXIT_FAILURE;
  }
  if (!download(sys, u_a_ref, stream_ref)) {
    return EXIT_FAILURE;
  }

  // Snapshot reference owned cells (kCycles even -> result lands in u_a).
  const std::vector<float> ref_owned = snapshot_owned(block, sys, u_a_ref);

  // ---- Variant runs ----
  // Extended / TwoWayExtended on this pipeline shape (no Map
  // predecessor / successor of the stencil) reduce to Standard at
  // apply_occ_ time but the rewrite still runs and must produce
  // identical results — same matrix as the Dense / DenseDisg
  // reference tests.
  const std::array<carbon::OverlapMode, 3> variants{carbon::OverlapMode::Standard,
                                                    carbon::OverlapMode::Extended,
                                                    carbon::OverlapMode::TwoWayExtended};

  for (const auto mode : variants) {
    auto u_a = block.template newField<float, 1>(carbon::MemorySide::HostDevice, "u_a");
    auto u_b = block.template newField<float, 1>(carbon::MemorySide::HostDevice, "u_b");
    init_field(block, sys, u_a);
    init_field(block, sys, u_b);

    carbon::Stream stream = sys.stream();
    if (!upload(sys, u_a, stream) || !upload(sys, u_b, stream)) {
      return EXIT_FAILURE;
    }
    if (!run_pipeline(sys, block, u_a, u_b, mode, stream, overlap_mode_name(mode))) {
      return EXIT_FAILURE;
    }
    if (!download(sys, u_a, stream)) {
      return EXIT_FAILURE;
    }

    // Compare owned cells to the reference. Ghost slots are
    // overwritten by halo_a / halo_b on every iteration, so a
    // divergence in the OCC schedule would surface as either an
    // owned-cell mismatch (wrong Internal / Boundary split) or a
    // ghost slot still holding zero (the OCC rewrite dropped a halo
    // edge); both surface through the owned-cell sweep because the
    // stencil reads ghosts.
    const std::vector<float> got_owned = snapshot_owned(block, sys, u_a);
    if (got_owned.size() != ref_owned.size()) {
      CARBON_LOG_ERROR(
          "test_skeleton_dist_overlap_block[rank={}, mode={}]: snapshot size mismatch (got "
          "{}, want {})",
          rank, overlap_mode_name(mode), got_owned.size(), ref_owned.size());
      return EXIT_FAILURE;
    }

    size_t mismatches = 0;
    for (size_t i = 0; i < ref_owned.size(); ++i) {
      if (got_owned[i] != ref_owned[i]) {
        if (mismatches < 4U) {
          CARBON_LOG_ERROR(
              "test_skeleton_dist_overlap_block[rank={}, mode={}]: mismatch at flat index {} "
              "got={} want={}",
              rank, overlap_mode_name(mode), i, got_owned[i], ref_owned[i]);
        }
        ++mismatches;
      }
    }
    if (mismatches > 0U) {
      CARBON_LOG_ERROR("test_skeleton_dist_overlap_block[rank={}, mode={}]: {} mismatches", rank,
                       overlap_mode_name(mode), mismatches);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_skeleton_dist_overlap_block[rank={}, mode={}]: OK ({} owned cells matched)", rank,
        overlap_mode_name(mode), ref_owned.size());
  }

  CARBON_LOG_INFO("test_skeleton_dist_overlap_block[rank={}/{}]: PASS", rank, world_size);
  return EXIT_SUCCESS;
}
