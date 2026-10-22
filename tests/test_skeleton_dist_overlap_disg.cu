// MPI cross-rank Skeleton OverlapMode parity test for DenseDisg.
//
// Closes the DenseDisg half of P1.δ in
// docs/audit_2026_05_fourth_pass.md (sibling gap to the third-pass
// `tests/test_skeleton_dist_overlap.cu` Dense closure). The single-rank
// DenseDisg OCC tests (`test_dense_disg_overlapped.cu`,
// `test_dense_disg_skeleton_extended.cu`) lock the apply_occ_ rewrite shape
// on a single process; this file extends the matrix to MPI so a regression
// in HaloDisg's tokens shape under {Standard, Extended, TwoWayExtended} —
// e.g. the back-walk picking up the wrong predecessor when the auto-halo
// shape differs from `Halo::tokens()` — is no longer silent.
//
// Same pipeline shape as `test_skeleton_dist_overlap.cu`: a ping-pong
// Jacobi (halo_a + stencil_ab + halo_b + stencil_ba) with halo kernels
// produced by `DenseDisg::Field::make_halo_kernel` — the public adapter
// that wraps `HaloDisg<T, Components>` as a `Kernel` of `Kind::Halo`. The
// auto-halo path in `Skeleton::compile` recognises `Kind::Halo` regardless
// of grid family, so apply_occ_ sees the same token-set as on Dense and
// the OCC rewrite must produce bit-equal results.
//
// For each `OverlapMode` variant we run kCycles cycles, download the final
// field on every rank, and compare cell-for-cell against the
// `OverlapMode::None` reference. Bit-equality is the post-condition because
// every kernel is deterministic and the OCC rewrite must not change the
// numerical sequence (it only reschedules across streams).
//
// Skipped (returns 0) when world_size < 2 or no GPUs are visible. The
// rank-local DenseDisg OCC path already has dedicated coverage in
// `test_dense_disg_overlapped.cu`; running with -n 1 would just duplicate
// that fixture.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_disg.h>
#include <carbon/distributed.h>
#include <carbon/halo_disg.h>
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

constexpr int64_t kWidth = 6;
constexpr int64_t kHeight = 5;
constexpr int kCycles = 4;  // each cycle = halo_a + stencil_ab + halo_b + stencil_ba

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t gz) -> float {
  return static_cast<float>(x + y * 11 + gz * 113);
}

// Resolve the linear offset into a DenseDisg::Field's per-slab cpu/gpu buffer.
// DenseDisg uses three separate z-regions inside a single owned slab — front
// halo card (bz==0..halo_z-1), interior body (bz==halo_z..halo_z+lim_z-1, but
// stored in a packed run), and back halo card. The same helper is used by
// `tests/test_dense_disg_overlapped.cu` ; cloning it here keeps the indexing
// behaviour locked across the OCC and OCC-MPI tests.
[[nodiscard]] auto
disg_addr(const carbon::DenseDisg::Span& span,
          int64_t bz,
          int64_t x,
          int64_t y,
          int c,
          int components) -> int64_t {
  const int64_t lz = span.logic_extent().z;
  const int64_t xdim = span.haloed_extent().x;
  const int64_t xy = span.xy_slab_elements();
  int64_t region_first_z = 0;
  int64_t region_z_dim = 1;
  int64_t offset_local_no_card = x + y * xdim;
  if (bz <= 1 || bz >= lz) {
    region_first_z = bz;
    region_z_dim = 1;
  } else {
    region_first_z = 2;
    region_z_dim = lz - 2;
    offset_local_no_card += (bz - 2) * xy;
  }
  return offset_local_no_card + xy * (region_first_z * static_cast<int64_t>(components) +
                                      static_cast<int64_t>(c) * region_z_dim);
}

// Same Jacobi step lambda as test_skeleton_dist_overlap.cu, retargeted to
// DenseDisg::{Span, Idx} . `loader.read_stencil(fin)` is what triggers the
// auto-halo insertion in `Skeleton::compile` ; the lambda body is identical
// to the Dense reference so any divergence in OCC behaviour is purely a
// disg-side thing.
[[nodiscard]] auto
build_jacobi_step(carbon::DenseDisg& dense,
                  carbon::DenseDisg::Field<float,
                                           1>& fin,
                  carbon::DenseDisg::Field<float,
                                           1>& fout,
                  std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s,
                                            carbon::DenseDisg::Idx idx) -> void {
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
init_field(const carbon::DenseDisg& dense,
           carbon::DenseDisg::Field<float,
                                    1>& f) {
  // One-rank-per-GPU under MPI -> spans().size() == 1. We still walk every
  // slab so this stays correct if a future change relaxes that.
  for (size_t i = 0; i < dense.spans().size(); ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    float* h = f.cpu_buffers()[i].data();
    const size_t n = f.cpu_buffers()[i].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = 0.0F;  // halo cards start at zero; the halo kernel overwrites them every iter
    }
    for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
      const int64_t global_z = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, /*components=*/1);
          h[off] = encode(x, y, global_z);
        }
      }
    }
  }
}

[[nodiscard]] auto
upload(const carbon::System& sys,
       carbon::DenseDisg::Field<float,
                                1>& f,
       const carbon::Stream& stream) -> bool {
  try {
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap_disg: upload failed: {}", e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
download(const carbon::System& sys,
         carbon::DenseDisg::Field<float,
                                  1>& f,
         const carbon::Stream& stream) -> bool {
  try {
    carbon::Transfer<float>(sys, f.gpu_buffers(), f.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap_disg: download failed: {}", e.what());
    return false;
  }
  return true;
}

// Run kCycles iterations of the ping-pong Jacobi pipeline at the given
// OverlapMode. Reads u_a / u_b on input, leaves the final result in u_a
// (kCycles is even -> last write lands in u_a).
[[nodiscard]] auto
run_pipeline(const carbon::System& sys,
             carbon::DenseDisg& dense,
             carbon::DenseDisg::Field<float,
                                      1>& u_a,
             carbon::DenseDisg::Field<float,
                                      1>& u_b,
             carbon::OverlapMode overlap,
             const carbon::Stream& stream,
             const char* tag) -> bool {
  carbon::Skeleton sk(sys, stream, carbon::Skeleton::Options{overlap, /*strict=*/false});
  // make_halo_kernel returns a Kernel of Kind::Halo wrapping a
  // HaloDisg<float, 1> over the field's gpu_buffers. apply_occ_ matches on
  // Kind::Halo so the OCC rewrite shape is the same as on Dense.
  sk.add(u_a.make_halo_kernel("halo_a"));
  sk.add(build_jacobi_step(dense, u_a, u_b, "step_ab"));
  sk.add(u_b.make_halo_kernel("halo_b"));
  sk.add(build_jacobi_step(dense, u_b, u_a, "step_ba"));

  try {
    for (int it = 0; it < kCycles; ++it) {
      sk.run();
    }
    sk.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap_disg[{}]: run threw: {}", tag, e.what());
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
    CARBON_LOG_INFO("test_skeleton_dist_overlap_disg: world_size={} < 2, skipping",
                    distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_skeleton_dist_overlap_disg: no GPU devices, skipping");
    return EXIT_SUCCESS;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_skeleton_dist_overlap_disg: expected gpu_group size 1, got {}",
                     sys.gpu_group().size());
    return EXIT_FAILURE;
  }

  // halo_z=1 so every iteration exercises the cross-rank HaloDisg path. The
  // Jacobi step's z neighbour reads land on the halo cards on every owned
  // slab boundary.
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{kWidth, kHeight, z_per_slab * static_cast<int64_t>(world_size)};
  carbon::DenseDisg dense(sys, global, /*halo_z=*/1);

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

  // Snapshot the reference owned cells (kCycles even -> result is in u_a).
  std::vector<float> ref_owned;
  {
    const auto& span = dense.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    ref_owned.reserve(static_cast<size_t>(haloed.x) * static_cast<size_t>(haloed.y) *
                      static_cast<size_t>(lim_z));
    const float* h = u_a_ref.cpu_buffers()[0].data();
    for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          ref_owned.push_back(h[disg_addr(span, bz, x, y, 0, /*components=*/1)]);
        }
      }
    }
  }

  // ---- Variant runs ----
  // Extended / TwoWayExtended on this pipeline shape (no Map predecessor /
  // successor of the stencil) reduce to Standard at apply_occ_ time but the
  // rewrite still runs and must produce identical results — same matrix as
  // the Dense reference test.
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

    // Compare owned cells to the reference. Halo cards are recomputed by
    // halo_a / halo_b every iteration, so a divergence in the OCC schedule
    // would surface as either an owned-cell mismatch (wrong Internal /
    // Boundary split) or a halo card still holding zero (the OCC rewrite
    // dropped a halo edge). The owned-cell sweep catches both because the
    // stencil reads halo cards.
    size_t mismatches = 0;
    {
      const auto& span = dense.spans()[0];
      const auto haloed = span.haloed_extent();
      const int64_t halo_z = span.halo_z();
      const int64_t lim_z = span.logic_extent().z;
      const float* h = u_a.cpu_buffers()[0].data();
      size_t i = 0;
      for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const float got = h[disg_addr(span, bz, x, y, 0, /*components=*/1)];
            const float want = ref_owned[i++];
            if (got != want) {
              if (mismatches < 4U) {
                CARBON_LOG_ERROR(
                    "test_skeleton_dist_overlap_disg[rank={}, mode={}]: mismatch (x,y,bz)="
                    "({},{},{}) got={} want={}",
                    rank, overlap_mode_name(mode), x, y, bz, got, want);
              }
              ++mismatches;
            }
          }
        }
      }
    }
    if (mismatches > 0U) {
      CARBON_LOG_ERROR("test_skeleton_dist_overlap_disg[rank={}, mode={}]: {} mismatches", rank,
                       overlap_mode_name(mode), mismatches);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_skeleton_dist_overlap_disg[rank={}, mode={}]: OK ({} owned cells matched)", rank,
        overlap_mode_name(mode), ref_owned.size());
  }

  CARBON_LOG_INFO("test_skeleton_dist_overlap_disg[rank={}/{}]: PASS", rank, world_size);
  return EXIT_SUCCESS;
}
