// MPI cross-rank Skeleton OverlapMode parity test.
//
// Anchors gap #5 of P1.δ in docs/audit_2026_05_third_pass.md: test_skeleton's
// Sub-tests E / F / G cover OverlapMode {Standard, Extended, TwoWayExtended}
// parity vs None — but only single-process. apply_occ_'s graph rewrite cares
// about token + halo identity, so an off-by-one in the cross-rank halo path
// under OCC would be invisible to the existing matrix.
//
// This test runs a ping-pong Jacobi pipeline (halo_a + stencil_ab + halo_b +
// stencil_ba — same shape as test_skeleton's Sub-test A, plus an MPI Halo
// going through the cross-rank DistTransfer / NCCL path) under each of:
//
//   * OverlapMode::None            -- the reference pipeline.
//   * OverlapMode::Standard        -- splits stencils into Internal / Boundary,
//                                     issues the halo on a sibling stream
//                                     concurrent with Internal compute.
//   * OverlapMode::Extended        -- as Standard, plus walks the Map-pattern
//                                     predecessor closure to find a wider
//                                     Internal region; on this pipeline shape
//                                     (no Map predecessor of the stencil) it
//                                     reduces to Standard at the apply_occ_
//                                     stage but the rewrite still runs.
//   * OverlapMode::TwoWayExtended  -- same plus Map-successor closure.
//
// For each variant we run kIters cycles, download the final field on every
// rank, and compare cell-for-cell against the OverlapMode::None reference.
// Bit-equality is the post-condition because every kernel is deterministic
// and the OCC rewrite must not change the numerical sequence (it only
// reschedules across streams).
//
// Skipped (returns 0) when world_size < 2 or no GPUs are visible.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/distributed.h>
#include <carbon/halo.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/neighbour_value.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
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

constexpr int64_t kWidth = 6;
constexpr int64_t kHeight = 5;
constexpr int kCycles = 4;  // each cycle = halo_a + stencil_ab + halo_b + stencil_ba

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t gz) -> float {
  return static_cast<float>(x + y * 11 + gz * 113);
}

// Shared loader/lambda for the Jacobi step.
auto
build_jacobi_step(carbon::Dense& dense,
                  carbon::Dense::Field<float,
                                       1>& fin,
                  carbon::Dense::Field<float,
                                       1>& fout,
                  std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const carbon::NeighbourValue<float> xp = in.neighbour(s, idx, {1, 0, 0}, 0);
          const carbon::NeighbourValue<float> xm = in.neighbour(s, idx, {-1, 0, 0}, 0);
          const carbon::NeighbourValue<float> yp = in.neighbour(s, idx, {0, 1, 0}, 0);
          const carbon::NeighbourValue<float> ym = in.neighbour(s, idx, {0, -1, 0}, 0);
          const carbon::NeighbourValue<float> zp = in.neighbour(s, idx, {0, 0, 1}, 0);
          const carbon::NeighbourValue<float> zm = in.neighbour(s, idx, {0, 0, -1}, 0);
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
      std::move(name),
      /*shared_mem=*/0U);
}

void
init_field(const carbon::Dense& dense,
           carbon::Dense::Field<float,
                                1>& f) {
  const auto& span = dense.spans()[0];
  const auto haloed = span.haloed_extent();
  const int64_t pitch = span.xy_slab_elements();
  const int64_t halo_z = span.halo_z();
  const int64_t z_origin = span.z_offset();
  const int64_t lim_z = span.logic_extent().z;
  float* h = f.cpu_buffers()[0].data();
  const size_t n = f.cpu_buffers()[0].size();
  for (size_t j = 0; j < n; ++j) {
    h[j] = 0.0F;  // halo cells start at zero; the halo kernel overwrites them every iter
  }
  for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
    const int64_t global_z = z_origin + z;
    for (int64_t y = 0; y < haloed.y; ++y) {
      for (int64_t x = 0; x < haloed.x; ++x) {
        const int64_t off = z * pitch + y * haloed.x + x;
        h[off] = encode(x, y, global_z);
      }
    }
  }
}

[[nodiscard]] auto
upload(const carbon::System& sys,
       carbon::Dense::Field<float,
                            1>& f,
       const carbon::Stream& stream) -> bool {
  try {
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap: upload failed: {}", e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
download(const carbon::System& sys,
         carbon::Dense::Field<float,
                              1>& f,
         const carbon::Stream& stream) -> bool {
  try {
    carbon::Transfer<float>(sys, f.gpu_buffers(), f.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap: download failed: {}", e.what());
    return false;
  }
  return true;
}

// Run kCycles iterations of the ping-pong Jacobi pipeline at the given
// OverlapMode. Reads u_a / u_b on input, leaves the final result in u_a
// (kCycles is even -> last write lands in u_a).
[[nodiscard]] auto
run_pipeline(const carbon::System& sys,
             carbon::Dense& dense,
             carbon::Dense::Field<float,
                                  1>& u_a,
             carbon::Dense::Field<float,
                                  1>& u_b,
             carbon::OverlapMode overlap,
             const carbon::Stream& stream,
             const char* tag) -> bool {
  carbon::Halo<float, 1> halo_a(u_a);
  carbon::Halo<float, 1> halo_b(u_b);

  carbon::Skeleton sk(sys, stream, carbon::Skeleton::Options{overlap, /*strict=*/false});
  sk.add(carbon::Kernel::halo(std::move(halo_a), "halo_a"));
  sk.add(build_jacobi_step(dense, u_a, u_b, "step_ab"));
  sk.add(carbon::Kernel::halo(std::move(halo_b), "halo_b"));
  sk.add(build_jacobi_step(dense, u_b, u_a, "step_ba"));

  try {
    for (int it = 0; it < kCycles; ++it) {
      sk.run();
    }
    sk.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap[{}]: run threw: {}", tag, e.what());
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

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_skeleton_dist_overlap: world_size={} < 2, skipping",
                    distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_skeleton_dist_overlap: no GPU devices, skipping");
    return EXIT_SUCCESS;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap: expected gpu_group size 1, got {}",
                     sys.gpu_group().size());
    return EXIT_FAILURE;
  }

  // Z-only radius-1 stencil so halo_z == 1 and the cross-rank Halo path is
  // exercised on every iteration.
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);
  // global_z divisible by world_size keeps slab arithmetic compact while still
  // forcing per-rank Internal / Boundary splits.
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{kWidth, kHeight, z_per_slab * static_cast<int64_t>(world_size)};
  carbon::Dense dense(sys, global, std::move(stencil));

  // ---- Reference run (OverlapMode::None) ----
  auto u_a_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_a_ref");
  auto u_b_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_b_ref");
  init_field(dense, u_a_ref);
  init_field(dense, u_b_ref);

  carbon::Stream stream_ref = sys.stream();
  if (!upload(sys, u_a_ref, stream_ref) || !upload(sys, u_b_ref, stream_ref)) {
    return EXIT_FAILURE;
  }
  if (!run_pipeline(sys, dense, u_a_ref, u_b_ref, carbon::OverlapMode::None, stream_ref,
                    "None-reference")) {
    return EXIT_FAILURE;
  }
  if (!download(sys, u_a_ref, stream_ref)) {
    return EXIT_FAILURE;
  }

  // Snapshot the reference owned cells.
  std::vector<float> ref_owned;
  {
    const auto& span = dense.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    ref_owned.reserve(static_cast<size_t>(haloed.x) * static_cast<size_t>(haloed.y) *
                      static_cast<size_t>(lim_z));
    const float* h = u_a_ref.cpu_buffers()[0].data();
    for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          ref_owned.push_back(h[z * pitch + y * haloed.x + x]);
        }
      }
    }
  }

  // ---- Variant runs ----
  // Extended / TwoWayExtended on this pipeline shape (no Map predecessor /
  // successor of the stencil) reduce to Standard at apply_occ_ time but the
  // rewrite still runs and must produce identical results — so they belong in
  // the matrix even though they don't widen the Internal region here.
  const std::array<carbon::OverlapMode, 3> variants{carbon::OverlapMode::Standard,
                                                    carbon::OverlapMode::Extended,
                                                    carbon::OverlapMode::TwoWayExtended};

  for (const auto mode : variants) {
    auto u_a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_a");
    auto u_b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_b");
    init_field(dense, u_a);
    init_field(dense, u_b);

    carbon::Stream stream = sys.stream();
    if (!upload(sys, u_a, stream) || !upload(sys, u_b, stream)) {
      return EXIT_FAILURE;
    }
    if (!run_pipeline(sys, dense, u_a, u_b, mode, stream, overlap_mode_name(mode))) {
      return EXIT_FAILURE;
    }
    if (!download(sys, u_a, stream)) {
      return EXIT_FAILURE;
    }

    // Compare owned cells to the reference. Halo cells are recomputed by
    // halo_a / halo_b every iteration, so a divergence in the OCC schedule
    // would surface as either an owned-cell mismatch (wrong Internal /
    // Boundary split) or a halo cell still holding zero (the OCC rewrite
    // dropped a halo edge). The owned-cell sweep catches both because the
    // stencil reads halo cells.
    size_t mismatches = 0;
    {
      const auto& span = dense.spans()[0];
      const auto haloed = span.haloed_extent();
      const int64_t pitch = span.xy_slab_elements();
      const int64_t halo_z = span.halo_z();
      const int64_t lim_z = span.logic_extent().z;
      const float* h = u_a.cpu_buffers()[0].data();
      size_t i = 0;
      for (int64_t z = halo_z; z < halo_z + lim_z; ++z) {
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const float got = h[z * pitch + y * haloed.x + x];
            const float want = ref_owned[i++];
            if (got != want) {
              if (mismatches < 4U) {
                CARBON_LOG_ERROR(
                    "test_skeleton_dist_overlap[rank={}, mode={}]: mismatch (x,y,z)=({},{},{}) "
                    "got={} want={}",
                    rank, overlap_mode_name(mode), x, y, z, got, want);
              }
              ++mismatches;
            }
          }
        }
      }
    }
    if (mismatches > 0U) {
      CARBON_LOG_ERROR("test_skeleton_dist_overlap[rank={}, mode={}]: {} mismatches", rank,
                       overlap_mode_name(mode), mismatches);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_skeleton_dist_overlap[rank={}, mode={}]: OK ({} owned cells matched)",
                    rank, overlap_mode_name(mode), ref_owned.size());
  }

  CARBON_LOG_INFO("test_skeleton_dist_overlap[rank={}/{}]: PASS", rank, world_size);
  return EXIT_SUCCESS;
}
