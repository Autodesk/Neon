// carbon::CheckpointWriter / CheckpointReader distributed (one-shard-per-rank) round-trip.
//
// Launched under `mpiexec -n 2`. Each rank constructs a distributed Dense (one slab per
// rank), allocates a HostDevice scalar f32 + 3-component i32 vector field, fills them on
// the host with deterministic encoders that depend on the global z, and exercises the
// checkpoint write -> read -> verify cycle:
//
//   1. CheckpointWriter w(dir, dense, step, time);
//      w.add("rho", scalar).add("u", vec).commit();
//      Verifies the rank-shard layout: <dir>/rho.bin.rank<N> + <dir>/u.bin.rank<N>
//      exist on every rank, but <dir>/manifest.json is written by rank 0 only.
//   2. CheckpointReader r(dir, dense);
//      Verifies the manifest's world_size matches; field_names() lists ['rho', 'u'].
//   3. r.load("rho", scalar2); r.load("u", vec2);
//      Verifies owned-cell parity per rank; halo cells stay at sentinel.
//   4. Negative path: CheckpointReader on a separate Dense whose global_extent
//      disagrees with the manifest must throw std::invalid_argument.
//
// Cross-rank coordination of the manifest write happens through the writer's commit()
// barriers (one before so all shards land, one after so peer readers see the manifest).

#include <carbon/carbon.h>
#include <carbon/checkpoint.h>
#include <carbon/dense.h>
#include <carbon/distributed.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kSentinel = -7777;

[[nodiscard]] auto
encode_scalar(int64_t x,
              int64_t y,
              int64_t global_z) noexcept -> float {
  return 0.125F * static_cast<float>(global_z * 10000 + y * 100 + x);
}

[[nodiscard]] auto
encode_vec(int64_t x,
           int64_t y,
           int64_t global_z,
           int c) noexcept -> std::int32_t {
  return static_cast<std::int32_t>(global_z * 100000 + y * 1000 + x * 10 + c);
}

template <typename T,
          int N>
void
fill_halo_sentinel_n(carbon::Dense::Field<T,
                                          N>& field,
                     T sentinel) {
  for (size_t g = 0; g < field.cpu_buffers().size(); ++g) {
    T* h = field.cpu_buffers()[g].data();
    const size_t n = field.cpu_buffers()[g].size();
    for (size_t j = 0; j < n; ++j) {
      h[j] = sentinel;
    }
  }
}

}  // namespace

int
main() {
  carbon::init();
  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_checkpoint_dense_dist: world_size={} < 2, skipping",
                    distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_checkpoint_dense_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);

  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(world_size)};
  carbon::Dense dense(sys, global, std::move(stencil));

  auto scalar = dense.newField<float, 1>(carbon::MemorySide::Host);
  auto vec = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);

  fill_halo_sentinel_n(scalar, static_cast<float>(kSentinel));
  fill_halo_sentinel_n(vec, static_cast<std::int32_t>(kSentinel));
  scalar.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
    return encode_scalar(gc.x, gc.y, gc.z);
  });
  vec.fill_owned_host([](carbon::int64_3d gc, int c) {
    return encode_vec(gc.x, gc.y, gc.z, c);
  });

  namespace fs = std::filesystem;
  const fs::path dir = "carbon_checkpoint_dense_dist_test";
  // Rank 0 cleans the directory before the run; every other rank waits at the writer's
  // first barrier (inside the ctor's ensure_directory) before touching anything.
  if (rank == 0) {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }

  constexpr int64_t kStep = 4567;
  constexpr double kTime = 12.5;

  // 1. Write checkpoint with two fields and a step / time tag. Cross-rank atomic commit
  //    happens inside commit() (two MPI barriers fence rank-0's manifest write).
  try {
    carbon::CheckpointWriter w(dir.string(), dense, kStep, kTime);
    w.add("rho", scalar).add("u", vec).commit();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_checkpoint_dense_dist: rank {} write failed: {}", rank, e.what());
    return 1;
  }

  // Verify per-rank shard files exist locally; the manifest is written by rank 0 only,
  // so every rank should also see it (post-commit barrier guarantees visibility).
  {
    const fs::path my_rho = dir / ("rho.bin.rank" + std::to_string(rank));
    const fs::path my_u = dir / ("u.bin.rank" + std::to_string(rank));
    if (!fs::exists(my_rho) || !fs::exists(my_u)) {
      CARBON_LOG_ERROR("test_checkpoint_dense_dist: rank {} per-shard files missing (rho={}, u={})",
                       rank, fs::exists(my_rho), fs::exists(my_u));
      return 1;
    }
    if (!fs::exists(dir / "manifest.json")) {
      CARBON_LOG_ERROR("test_checkpoint_dense_dist: rank {} manifest.json not visible after commit",
                       rank);
      return 1;
    }
  }

  // 2. Construct a reader on every rank and verify metadata accessors round-trip.
  carbon::CheckpointReader r(dir.string(), dense);
  if (r.step() != kStep) {
    CARBON_LOG_ERROR("test_checkpoint_dense_dist: rank {} step round-trip mismatch: got {}", rank,
                     r.step());
    return 1;
  }
  if (r.time() != kTime) {
    CARBON_LOG_ERROR("test_checkpoint_dense_dist: rank {} time round-trip mismatch: got {}", rank,
                     r.time());
    return 1;
  }
  const auto names = r.field_names();
  if (names.size() != 2U || names[0] != "rho" || names[1] != "u") {
    CARBON_LOG_ERROR("test_checkpoint_dense_dist: rank {} field_names() unexpected (size {})", rank,
                     names.size());
    return 1;
  }

  // 3. Read back into fresh fields and verify owned-cell parity per rank.
  auto scalar2 = dense.newField<float, 1>(carbon::MemorySide::Host);
  auto vec2 = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
  fill_halo_sentinel_n(scalar2, static_cast<float>(kSentinel));
  fill_halo_sentinel_n(vec2, static_cast<std::int32_t>(kSentinel));
  try {
    r.load("rho", scalar2);
    r.load("u", vec2);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_checkpoint_dense_dist: rank {} load failed: {}", rank, e.what());
    return 1;
  }

  size_t mismatches = 0;
  {
    const auto& span = dense.spans()[0];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t per_comp = span.linear_storage_elements();
    const float* hs = scalar2.cpu_buffers()[0].data();
    const std::int32_t* hv = vec2.cpu_buffers()[0].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const float gs = hs[off];
          const float ws = is_halo ? static_cast<float>(kSentinel) : encode_scalar(x, y, global_z);
          if (gs != ws) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_checkpoint_dense_dist: rank {} scalar (x,y,z)=({},{},{}) gz={} got={} "
                  "want={} is_halo={}",
                  rank, x, y, z, global_z, gs, ws, is_halo);
            }
            ++mismatches;
          }
          for (int c = 0; c < 3; ++c) {
            const int64_t voff = static_cast<int64_t>(c) * per_comp + off;
            const std::int32_t gv = hv[voff];
            const std::int32_t wv =
                is_halo ? static_cast<std::int32_t>(kSentinel) : encode_vec(x, y, global_z, c);
            if (gv != wv) {
              if (mismatches < 8U) {
                CARBON_LOG_ERROR(
                    "test_checkpoint_dense_dist: rank {} vec (x,y,z,c)=({},{},{},{}) gz={} got={} "
                    "want={} is_halo={}",
                    rank, x, y, z, c, global_z, gv, wv, is_halo);
              }
              ++mismatches;
            }
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_checkpoint_dense_dist: rank {} {} mismatches", rank, mismatches);
    return 1;
  }

  // 4. Negative path: a Dense with a different global_extent must be rejected at reader
  //    construction. Sized so the per-rank slab count still matches (z_per_slab = 4) but
  //    the global x extent differs from the writer's, triggering validate_manifest.
  {
    carbon::Stencil stencil2(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                             /*remove_center=*/true);
    const carbon::int64_3d different_global{8, 4, z_per_slab * static_cast<int64_t>(world_size)};
    carbon::Dense dense2(sys, different_global, std::move(stencil2));
    bool threw = false;
    try {
      carbon::CheckpointReader r_bad(dir.string(), dense2);
    } catch (const std::invalid_argument&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR(
          "test_checkpoint_dense_dist: rank {} extent-mismatch reader ctor threw unexpected "
          "exception: {}",
          rank, e.what());
      return 1;
    }
    if (!threw) {
      CARBON_LOG_ERROR("test_checkpoint_dense_dist: rank {} extent-mismatch reader did not throw",
                       rank);
      return 1;
    }
  }

  // Cleanup. Rank 0 wipes the directory after every rank has finished reading.
  distributed->barrier();
  if (rank == 0) {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }

  CARBON_LOG_INFO("test_checkpoint_dense_dist: rank {} of {} ok", rank, world_size);
  return 0;
}
