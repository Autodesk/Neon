// carbon::CheckpointWriter / CheckpointReader round-trip test (single-process).
//
// Builds a multi-GPU Dense, allocates two HostDevice fields with different (T, Components)
// (a scalar float and a 3-component int32 vector), seeds them with deterministic per-cell
// patterns, and exercises the checkpoint write -> read -> verify cycle:
//
//   1. CheckpointWriter w(dir, dense, step, time);
//      w.add("rho", scalar).add("u", vec).commit();
//   2. Verify the on-disk layout: <dir>/manifest.json + <dir>/rho.bin + <dir>/u.bin all exist.
//   3. CheckpointReader r(dir, dense);
//      Verify r.step() / r.time() / r.field_names() reflect the writer's input.
//   4. r.load("rho", scalar2); r.load("u", vec2);
//      Verify owned cells round-trip byte-for-byte; halo cells stay at sentinel.
//   5. Negative paths:
//      - load() on a missing field name throws std::out_of_range.
//      - load() on a dtype-mismatched bound Field throws std::invalid_argument.
//      - reader ctor on a directory without manifest.json throws std::runtime_error.

#include <carbon/carbon.h>
#include <carbon/checkpoint.h>
#include <carbon/dense.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
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

[[nodiscard]] auto
verify_scalar(const carbon::Dense& dense,
              const carbon::Dense::Field<float,
                                         1>& field) -> size_t {
  const size_t n_gpus = dense.spans().size();
  size_t mismatches = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const float* h = field.cpu_buffers()[g].data();
    for (int64_t z = 0; z < haloed.z; ++z) {
      const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
      const int64_t global_z = z_origin + z;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = z * pitch + y * haloed.x + x;
          const float got = h[off];
          const float want =
              is_halo ? static_cast<float>(kSentinel) : encode_scalar(x, y, global_z);
          if (got != want) {
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_checkpoint_dense: scalar mismatch slab {} (x,y,z)=({},{},{}) gz={} "
                  "got={} want={} is_halo={}",
                  g, x, y, z, global_z, got, want, is_halo);
            }
            ++mismatches;
          }
        }
      }
    }
  }
  return mismatches;
}

[[nodiscard]] auto
verify_vec(const carbon::Dense& dense,
           const carbon::Dense::Field<std::int32_t,
                                      3>& field) -> size_t {
  const size_t n_gpus = dense.spans().size();
  size_t mismatches = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t per_comp = span.linear_storage_elements();
    const std::int32_t* h = field.cpu_buffers()[g].data();
    for (int c = 0; c < 3; ++c) {
      const int64_t plane = static_cast<int64_t>(c) * per_comp;
      for (int64_t z = 0; z < haloed.z; ++z) {
        const bool is_halo = (z < halo_z) || (z >= halo_z + lim_z);
        const int64_t global_z = z_origin + z;
        for (int64_t y = 0; y < haloed.y; ++y) {
          for (int64_t x = 0; x < haloed.x; ++x) {
            const int64_t off = plane + z * pitch + y * haloed.x + x;
            const std::int32_t got = h[off];
            const std::int32_t want =
                is_halo ? static_cast<std::int32_t>(kSentinel) : encode_vec(x, y, global_z, c);
            if (got != want) {
              if (mismatches < 8U) {
                CARBON_LOG_ERROR(
                    "test_checkpoint_dense: vec mismatch slab {} c={} (x,y,z)=({},{},{}) gz={} "
                    "got={} want={} is_halo={}",
                    g, c, x, y, z, global_z, got, want, is_halo);
              }
              ++mismatches;
            }
          }
        }
      }
    }
  }
  return mismatches;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_WARN("test_checkpoint_dense: no CUDA devices, skipping");
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{4, 4, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil));

  auto scalar = dense.newField<float, 1>(carbon::MemorySide::Host);
  auto vec = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);

  fill_halo_sentinel_n(scalar, static_cast<float>(kSentinel));
  fill_halo_sentinel_n(vec, static_cast<std::int32_t>(kSentinel));
  scalar.fill_owned_host([](carbon::int64_3d gc, int /*c*/) -> float {
    return encode_scalar(gc.x, gc.y, gc.z);
  });
  vec.fill_owned_host([](carbon::int64_3d gc, int c) -> std::int32_t {
    return encode_vec(gc.x, gc.y, gc.z, c);
  });

  namespace fs = std::filesystem;
  const fs::path dir = "carbon_checkpoint_dense_test";
  fs::remove_all(dir);

  constexpr int64_t kStep = 1234;
  constexpr double kTime = 5.25;

  // 1. Write checkpoint with two fields and a step / time tag.
  try {
    carbon::CheckpointWriter w(dir.string(), dense, kStep, kTime);
    w.add("rho", scalar).add("u", vec).commit();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_checkpoint_dense: write failed: {}", e.what());
    return 1;
  }

  // 2. Verify on-disk artefacts: manifest.json + the two field shards.
  if (!fs::exists(dir / "manifest.json")) {
    CARBON_LOG_ERROR("test_checkpoint_dense: manifest.json missing after commit");
    return 1;
  }
  if (!fs::exists(dir / "rho.bin")) {
    CARBON_LOG_ERROR("test_checkpoint_dense: rho.bin missing after commit");
    return 1;
  }
  if (!fs::exists(dir / "u.bin")) {
    CARBON_LOG_ERROR("test_checkpoint_dense: u.bin missing after commit");
    return 1;
  }

  // 3. Reader ctor parses manifest; verify metadata accessors.
  carbon::CheckpointReader r(dir.string(), dense);
  if (r.step() != kStep) {
    CARBON_LOG_ERROR("test_checkpoint_dense: step round-trip mismatch: got {}, want {}", r.step(),
                     kStep);
    return 1;
  }
  if (r.time() != kTime) {
    CARBON_LOG_ERROR("test_checkpoint_dense: time round-trip mismatch: got {}, want {}", r.time(),
                     kTime);
    return 1;
  }
  const auto names = r.field_names();
  if (names.size() != 2U || names[0] != "rho" || names[1] != "u") {
    CARBON_LOG_ERROR("test_checkpoint_dense: field_names() = [{}] (size {}), expected ['rho', 'u']",
                     names.empty() ? std::string("") : names[0], names.size());
    return 1;
  }

  // 4. Read back into fresh fields and verify owned-cell parity + halo untouched.
  auto scalar2 = dense.newField<float, 1>(carbon::MemorySide::Host);
  auto vec2 = dense.newField<std::int32_t, 3>(carbon::MemorySide::Host);
  fill_halo_sentinel_n(scalar2, static_cast<float>(kSentinel));
  fill_halo_sentinel_n(vec2, static_cast<std::int32_t>(kSentinel));
  try {
    r.load("rho", scalar2);
    r.load("u", vec2);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_checkpoint_dense: load failed: {}", e.what());
    return 1;
  }
  if (size_t mismatches = verify_scalar(dense, scalar2); mismatches > 0U) {
    CARBON_LOG_ERROR("test_checkpoint_dense: {} scalar mismatches after round-trip", mismatches);
    return 1;
  }
  if (size_t mismatches = verify_vec(dense, vec2); mismatches > 0U) {
    CARBON_LOG_ERROR("test_checkpoint_dense: {} vec mismatches after round-trip", mismatches);
    return 1;
  }

  // 5a. Missing field name -> std::out_of_range.
  {
    auto missing = dense.newField<float, 1>(carbon::MemorySide::Host);
    bool threw = false;
    try {
      r.load("does_not_exist", missing);
    } catch (const std::out_of_range&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_checkpoint_dense: missing-field load threw unexpected exception: {}",
                       e.what());
      return 1;
    }
    if (!threw) {
      CARBON_LOG_ERROR("test_checkpoint_dense: missing-field load did not throw");
      return 1;
    }
  }

  // 5b. dtype mismatch -> std::invalid_argument (bind a double to a float slot).
  {
    auto wrong_dtype = dense.newField<double, 1>(carbon::MemorySide::Host);
    bool threw = false;
    try {
      r.load("rho", wrong_dtype);
    } catch (const std::invalid_argument&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_checkpoint_dense: dtype-mismatch load threw unexpected exception: {}",
                       e.what());
      return 1;
    }
    if (!threw) {
      CARBON_LOG_ERROR("test_checkpoint_dense: dtype-mismatch load did not throw");
      return 1;
    }
  }

  // 5c. Reader ctor on a non-existent dir throws std::runtime_error (no manifest).
  {
    bool threw = false;
    try {
      carbon::CheckpointReader r2("carbon_checkpoint_dense_does_not_exist", dense);
    } catch (const std::runtime_error&) {
      threw = true;
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR(
          "test_checkpoint_dense: missing-dir reader ctor threw unexpected exception: {}",
          e.what());
      return 1;
    }
    if (!threw) {
      CARBON_LOG_ERROR("test_checkpoint_dense: missing-dir reader ctor did not throw");
      return 1;
    }
  }

  // 5d. A manifest whose "file" entry escapes the checkpoint directory must be rejected at parse
  // time. field_shard_path joins the entry onto the directory with fs::path::operator/, which
  // discards the directory entirely for an absolute right-hand side, so an edited manifest could
  // otherwise point the reader at any file on the host and load it into a field buffer. The
  // writer never emits such an entry — only a hand-edited or corrupt manifest reaches this.
  {
    const std::string escape_dir = "carbon_checkpoint_dense_escape";
    fs::remove_all(escape_dir);
    fs::create_directories(escape_dir);
    const auto ge = dense.global_extent();

    for (const char* hostile : {"../../etc/passwd", "/etc/passwd", "sub/dir/rho.bin"}) {
      {
        std::ofstream mf(fs::path(escape_dir) / "manifest.json");
        mf << "{\"schema_version\":1,\"step\":0,\"time\":0.0,\"global_extent\":[" << ge.x << ","
           << ge.y << "," << ge.z << "],\"world_size\":1,"
           << "\"fields\":[{\"name\":\"rho\",\"dtype\":\"f32\",\"components\":1,\"file\":\""
           << hostile << "\"}]}";
      }

      bool threw = false;
      try {
        carbon::CheckpointReader bad(escape_dir, dense);
      } catch (const std::runtime_error&) {
        threw = true;
      } catch (const std::exception& e) {
        CARBON_LOG_ERROR("test_checkpoint_dense: manifest file entry '{}' threw unexpected: {}",
                         hostile, e.what());
        fs::remove_all(escape_dir);
        return 1;
      }
      if (!threw) {
        CARBON_LOG_ERROR(
            "test_checkpoint_dense: manifest file entry '{}' was accepted; the reader would "
            "resolve a path outside the checkpoint directory",
            hostile);
        fs::remove_all(escape_dir);
        return 1;
      }
    }
    fs::remove_all(escape_dir);
  }

  // The manifest is committed by rename from a temp sibling, so a completed commit must leave no
  // manifest.json.tmp behind for the next reader (or a directory listing) to trip over.
  if (fs::exists(fs::path(dir) / "manifest.json.tmp")) {
    CARBON_LOG_ERROR("test_checkpoint_dense: manifest.json.tmp survived commit in '{}'",
                     dir.string());
    return 1;
  }

  fs::remove_all(dir);
  CARBON_LOG_INFO("test_checkpoint_dense: ok ({} GPUs; rho f32 + u i32x3 round-trip)", n_gpus);
  return 0;
}
