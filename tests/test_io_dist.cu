// Multi-rank round-trip test for the VTK / HDF5 / nanoVDB I/O backends.
//
// Anchors gap #2 of P1.δ in docs/audit_2026_05_third_pass.md: only
// native_io_dist exercised the per-rank-file MPI shape; VTK and nanoVDB
// advertise the same shape (`.rank<N>` suffix per the io_*.inl docstrings)
// but no test confirmed their write/read pair lines up under MPI. HDF5 has
// a different shape — collective writes against a single file when
// CARBON_HAVE_PHDF5 is on, throws when HDF5 was built serial-only — and
// neither path was tested under MPI either.
//
// Each format is gated on the same CARBON_HAVE_* flag the single-process
// io tests use; absent the flag the corresponding sub-section is skipped
// at compile time. The test passes (returns 0) under world_size < 2 / no
// GPUs visible / serial-only HDF5 with a clear log line — same shape as
// every other dist test.
//
// On-disk shapes per round-trip:
//   * VTK     — per-rank legacy STRUCTURED_POINTS BINARY (.rank<N>).
//   * nanoVDB — per-rank multi-grid .nvdb (.rank<N>).
//   * HDF5    — single .h5 file, collective writes if CARBON_HAVE_PHDF5,
//               otherwise the writer throws std::runtime_error and we
//               assert that exact shape so a regression that silently
//               succeeded with the serial driver would surface.
//
// Each round-trip writes a HostDevice scalar f32 field and a 3-component
// i32 field, reads them back into fresh fields with halos pre-sentinel'd,
// and verifies (a) owned cells match the encoder bit-for-bit, (b) halo
// cells stay at the sentinel (the writers / readers must touch owned
// cells only).

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/distributed.h>
#include <carbon/io.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
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
fill_halo_sentinel(carbon::Dense::Field<T,
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

template <typename T,
          int N,
          class EncodeFn>
[[nodiscard]] auto
verify_round_trip(const carbon::Dense& dense,
                  const carbon::Dense::Field<T,
                                             N>& field,
                  EncodeFn encode,
                  int rank,
                  const char* tag) -> bool {
  const auto& span = dense.spans()[0];
  const auto haloed = span.haloed_extent();
  const int64_t pitch = span.xy_slab_elements();
  const int64_t halo_z = span.halo_z();
  const int64_t z_origin = span.z_offset();
  const int64_t lim_z = span.logic_extent().z;
  const int64_t per_comp = span.linear_storage_elements();
  const T* h = field.cpu_buffers()[0].data();
  size_t mismatches = 0;
  for (int c = 0; c < N; ++c) {
    const int64_t plane = static_cast<int64_t>(c) * per_comp;
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = plane + z * pitch + y * haloed.x + x;
          const T got = h[off];
          const T want = is_halo ? static_cast<T>(kSentinel) : encode(x, y, global_z, c);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "{}[rank={}]: mismatch (c,x,y,z)=({},{},{},{}) gz={} got={} want={} is_halo={}",
                  tag, rank, c, x, y, z, global_z, static_cast<double>(got),
                  static_cast<double>(want), is_halo);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("{}[rank={}]: {} mismatches", tag, rank, mismatches);
    return false;
  }
  return true;
}

template <typename T,
          int N,
          class EncodeFn>
void
fill_field(carbon::Dense::Field<T,
                                N>& field,
           EncodeFn encode) {
  fill_halo_sentinel(field, static_cast<T>(kSentinel));
  field.fill_owned_host([&](carbon::int64_3d gc, int c) {
    return encode(gc.x, gc.y, gc.z, c);
  });
}

}  // namespace

int
main() {
  carbon::init();
  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_io_dist: world_size={} < 2, skipping", distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_io_dist: no GPU devices, skipping");
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

  const auto enc_scalar = [](int64_t x, int64_t y, int64_t gz, int /*c*/) -> float {
    return encode_scalar(x, y, gz);
  };
  const auto enc_vec = [](int64_t x, int64_t y, int64_t gz, int c) -> std::int32_t {
    return encode_vec(x, y, gz, c);
  };

  // Allocate writer fields once; reused across formats.
  auto scalar = dense.newField<float, 1>(carbon::MemorySide::Host);
  auto vec = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
  fill_field(scalar, enc_scalar);
  fill_field(vec, enc_vec);

  // ---- VTK round-trip ----
  // VTK has no compile-time gate -- the legacy writer / reader are always
  // built into carbon, same as native binary.
  {
    const std::string path = "carbon_io_dist_test.vtk";
    const std::string my_shard = path + ".rank" + std::to_string(rank);
    std::remove(my_shard.c_str());

    try {
      carbon::io::write_vtk(path, dense, scalar, vec);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_io_dist[VTK,rank={}]: write_vtk failed: {}", rank, e.what());
      return 1;
    }

    auto scalar2 = dense.newField<float, 1>(carbon::MemorySide::Host);
    auto vec2 = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
    fill_halo_sentinel(scalar2, static_cast<float>(kSentinel));
    fill_halo_sentinel(vec2, static_cast<std::int32_t>(kSentinel));

    try {
      carbon::io::read_vtk(path, dense, scalar2, vec2);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_io_dist[VTK,rank={}]: read_vtk failed: {}", rank, e.what());
      return 1;
    }
    if (!verify_round_trip(dense, scalar2, enc_scalar, rank, "test_io_dist[VTK scalar]")) {
      return 1;
    }
    if (!verify_round_trip(dense, vec2, enc_vec, rank, "test_io_dist[VTK vec]")) {
      return 1;
    }
    std::remove(my_shard.c_str());
    CARBON_LOG_INFO("test_io_dist[VTK,rank={}]: ok", rank);
  }

  // ---- nanoVDB round-trip ----
#ifdef CARBON_HAVE_NANOVDB
  {
    const std::string path = "carbon_io_dist_test.nvdb";
    const std::string my_shard = path + ".rank" + std::to_string(rank);
    std::remove(my_shard.c_str());

    try {
      carbon::io::write_nanovdb(path, dense, scalar, vec);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_io_dist[nanoVDB,rank={}]: write_nanovdb failed: {}", rank, e.what());
      return 1;
    }

    auto scalar2 = dense.newField<float, 1>(carbon::MemorySide::Host);
    auto vec2 = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
    fill_halo_sentinel(scalar2, static_cast<float>(kSentinel));
    fill_halo_sentinel(vec2, static_cast<std::int32_t>(kSentinel));

    try {
      carbon::io::read_nanovdb(path, dense, scalar2, vec2);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_io_dist[nanoVDB,rank={}]: read_nanovdb failed: {}", rank, e.what());
      return 1;
    }
    if (!verify_round_trip(dense, scalar2, enc_scalar, rank, "test_io_dist[nanoVDB scalar]")) {
      return 1;
    }
    if (!verify_round_trip(dense, vec2, enc_vec, rank, "test_io_dist[nanoVDB vec]")) {
      return 1;
    }
    std::remove(my_shard.c_str());
    CARBON_LOG_INFO("test_io_dist[nanoVDB,rank={}]: ok", rank);
  }
#else
  CARBON_LOG_INFO("test_io_dist: nanoVDB not enabled at compile time, skipping nanoVDB section");
#endif

  // ---- HDF5 round-trip ----
  // CARBON_HAVE_PHDF5 (parallel HDF5) -> collective write to a single file.
  // CARBON_HAVE_HDF5 only (serial HDF5) -> writer must throw with a clean
  // diagnostic; we lock that contract so a future regression that silently
  // wrote rank 0's slabs to the file would surface here.
#ifdef CARBON_HAVE_HDF5
  {
    const std::string path = "carbon_io_dist_test.h5";
    if (rank == 0) {
      std::remove(path.c_str());
    }
#ifdef CARBON_HAVE_PHDF5
    try {
      carbon::io::write_hdf5(path, dense, scalar, vec);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_io_dist[HDF5,rank={}]: write_hdf5 failed: {}", rank, e.what());
      return 1;
    }

    auto scalar2 = dense.newField<float, 1>(carbon::MemorySide::Host);
    auto vec2 = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
    fill_halo_sentinel(scalar2, static_cast<float>(kSentinel));
    fill_halo_sentinel(vec2, static_cast<std::int32_t>(kSentinel));
    try {
      carbon::io::read_hdf5(path, dense, scalar2, vec2);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_io_dist[HDF5,rank={}]: read_hdf5 failed: {}", rank, e.what());
      return 1;
    }
    if (!verify_round_trip(dense, scalar2, enc_scalar, rank, "test_io_dist[HDF5 scalar]")) {
      return 1;
    }
    if (!verify_round_trip(dense, vec2, enc_vec, rank, "test_io_dist[HDF5 vec]")) {
      return 1;
    }
    if (rank == 0) {
      std::remove(path.c_str());
    }
    CARBON_LOG_INFO("test_io_dist[HDF5,rank={}]: ok (collective)", rank);
#else
    bool threw = false;
    try {
      carbon::io::write_hdf5(path, dense, scalar, vec);
    } catch (const std::runtime_error&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR(
          "test_io_dist[HDF5-serial,rank={}]: write_hdf5 threw unexpected exception type: {}", rank,
          e.what());
      return 1;
    }
    if (!threw) {
      CARBON_LOG_ERROR(
          "test_io_dist[HDF5-serial,rank={}]: write_hdf5 must throw under serial HDF5 + MPI; did "
          "not throw",
          rank);
      return 1;
    }
    CARBON_LOG_INFO("test_io_dist[HDF5-serial,rank={}]: write_hdf5 threw as expected (no PHDF5)",
                    rank);
#endif
  }
#else
  CARBON_LOG_INFO("test_io_dist: HDF5 not enabled at compile time, skipping HDF5 section");
#endif

  CARBON_LOG_INFO("test_io_dist: rank {} of {} PASS", rank, world_size);
  return 0;
}
