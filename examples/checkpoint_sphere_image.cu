// Tutorial: Dense grid, sphere signed-distance field (SDF), checkpoint / restart,
// downsampled slice, and grayscale PGM export.
//
//   1. Build a @c Dense layout + scalar @c float field.
//   2. Fill owned cells with the analytic SDF of a sphere (negative inside, positive
//      outside, zero on the surface) using @c fill_owned_host .
//   3. @c CheckpointWriter saves @c "sdf" under a directory, then @c commit() .
//   4. Simulate restart: fresh field, @c CheckpointReader @c load("sdf", ...) and verify
//      against the same SDF formula.
//   5. Downsample an XY slice at mid- @c z (stride 4 in x/y) into a small host buffer.
//   6. Stretch values to 8-bit and write Netpbm @c P5 (binary PGM). Under MPI with
//      @c world_size > 1 the slice may span ranks — this tutorial skips the image
//      step and logs a hint (single-rank @c mpiexec -n 1 reproduces the full path).
//
// Build & run
// -----------
//   cmake --build build --target carbon_example_checkpoint_sphere_image
//   ./build/examples/carbon_example_checkpoint_sphere_image
//   ./build/examples/carbon_example_checkpoint_sphere_image /tmp/out.pgm
//
// With MPI (checkpoint still works; image only when a single rank owns the full grid
// or you run @c mpiexec -n 1 ):
//   cmake --build build-mpi --target carbon_example_checkpoint_sphere_image
//   mpiexec -n 1 ./build-mpi/examples/carbon_example_checkpoint_sphere_image

#include <carbon/carbon.h>
#include <carbon/checkpoint.h>
#include <carbon/dense.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

#ifdef CARBON_HAVE_MPI_NCCL
[[nodiscard]] auto
world_size(const carbon::System& sys) noexcept -> int {
  if (const auto* d = sys.distributed(); d != nullptr) {
    return d->world_size();
  }
  return 1;
}

[[nodiscard]] auto
world_rank(const carbon::System& sys) noexcept -> int {
  if (const auto* d = sys.distributed(); d != nullptr) {
    return d->world_rank();
  }
  return 0;
}

void
barrier(const carbon::System& sys) {
  if (const auto* d = sys.distributed(); d != nullptr) {
    d->barrier();
  }
}
#else
[[nodiscard]] auto
world_size(const carbon::System& /*sys*/) noexcept -> int {
  return 1;
}

[[nodiscard]] auto
world_rank(const carbon::System& /*sys*/) noexcept -> int {
  return 0;
}

void
barrier(const carbon::System& /*sys*/) {}
#endif

[[nodiscard]] auto
sphere_sdf(float px,
           float py,
           float pz,
           float cx,
           float cy,
           float cz,
           float radius) noexcept -> float {
  const float dx = px - cx;
  const float dy = py - cy;
  const float dz = pz - cz;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
}

[[nodiscard]] auto
try_sample_owned_sdf(const carbon::Dense& dense,
                     const carbon::Dense::Field<float,
                                                1>& f,
                     int64_t gx,
                     int64_t gy,
                     int64_t gz,
                     float* out) -> bool {
  const auto ge = dense.global_extent();
  if (gx < 0 || gx >= ge.x || gy < 0 || gy >= ge.y || gz < 0 || gz >= ge.z) {
    return false;
  }
  const size_t n_gpus = dense.spans().size();
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t hz = span.halo_z();
    const int64_t lz = span.logic_extent().z;
    const int64_t z0 = span.z_offset();
    const int64_t bz = gz - z0;
    if (bz < hz || bz >= hz + lz) {
      continue;
    }
    const float* h = f.cpu_buffers()[g].data();
    const int64_t off = bz * pitch + gy * haloed.x + gx;
    *out = h[off];
    return true;
  }
  return false;
}

auto
downsample_xy_slice(const carbon::Dense& dense,
                    const carbon::Dense::Field<float,
                                               1>& f,
                    int64_t gz_slice,
                    int64_t stride,
                    int64_t* out_w,
                    int64_t* out_h) -> std::vector<float> {
  const auto ge = dense.global_extent();
  const int64_t w = (ge.x + stride - 1) / stride;
  const int64_t h = (ge.y + stride - 1) / stride;
  *out_w = w;
  *out_h = h;
  std::vector<float> img(static_cast<size_t>(w * h), 0.0F);
  for (int64_t j = 0; j < h; ++j) {
    for (int64_t i = 0; i < w; ++i) {
      const int64_t gx = std::min(i * stride + stride / 2, ge.x - 1);
      const int64_t gy = std::min(j * stride + stride / 2, ge.y - 1);
      float v = 0.0F;
      if (!try_sample_owned_sdf(dense, f, gx, gy, gz_slice, &v)) {
        v = std::numeric_limits<float>::quiet_NaN();
      }
      img[static_cast<size_t>(j * w + i)] = v;
    }
  }
  return img;
}

auto
write_pgm_gray(std::string_view path,
               const std::vector<float>& img,
               int64_t width,
               int64_t height) -> bool {
  float mn = std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();
  for (float v : img) {
    if (!std::isfinite(v)) {
      continue;
    }
    mn = std::min(mn, v);
    mx = std::max(mx, v);
  }
  if (!(mn < mx)) {
    mn = -1.0F;
    mx = 1.0F;
  }
  const float inv = 1.0F / (mx - mn);
  std::ofstream out(std::string(path), std::ios::binary);
  if (!out) {
    return false;
  }
  out << "P5\n" << width << ' ' << height << "\n255\n";
  for (float v : img) {
    unsigned char b = 0;
    if (std::isfinite(v)) {
      const float t = (v - mn) * inv;
      const float c = std::clamp(t, 0.0F, 1.0F);
      b = static_cast<unsigned char>(std::lround(c * 255.0F));
    }
    out.put(static_cast<char>(b));
  }
  return static_cast<bool>(out.good());
}

}  // namespace

int
main(int argc,
     char** argv) {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_ERROR("checkpoint_sphere_image: no CUDA devices — aborting");
    return EXIT_FAILURE;
  }

#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  carbon::System sys = (distributed->world_size() > 1) ? carbon::System(distributed)
                                                       : carbon::System(carbon::get_all_devices());
#else
  carbon::System sys(carbon::get_all_devices());
#endif
  const size_t n_gpus = sys.gpu_group().size();
  const int rank = world_rank(sys);
  const int ranks = world_size(sys);

  carbon::Stencil stencil(std::vector<carbon::int32_3d>{
      {0, 0, 0}, {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}});

  constexpr int64_t kStride = 4;
  const int64_t z_per_slab = 8;
  const carbon::int64_3d global{96, 96, z_per_slab * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil));

  const float cx = 0.5F * static_cast<float>(global.x - 1);
  const float cy = 0.5F * static_cast<float>(global.y - 1);
  const float cz = 0.5F * static_cast<float>(global.z - 1);
  const float radius = 0.32F * static_cast<float>(std::min({global.x, global.y, global.z}));

  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Host).set_name("sdf");
  sdf.fill_owned_host([&](carbon::int64_3d gc, int /*c*/) -> float {
    const float px = static_cast<float>(gc.x);
    const float py = static_cast<float>(gc.y);
    const float pz = static_cast<float>(gc.z);
    return sphere_sdf(px, py, pz, cx, cy, cz, radius);
  });

  namespace fs = std::filesystem;
  const fs::path dir = "ckpt_sphere_image_demo";
  if (rank == 0) {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  barrier(sys);

  constexpr int64_t kStep = 100;
  constexpr double kTime = 0.0;
  {
    carbon::CheckpointWriter w(dir.string(), dense, kStep, kTime);
    w.add("sdf", sdf);
    w.commit();
  }
  CARBON_LOG_INFO("checkpoint_sphere_image: rank {} wrote checkpoint '{}'", rank, dir.string());

  auto sdf_loaded = dense.newField<float, 1>(carbon::MemorySide::Host).set_name("sdf_loaded");
  {
    carbon::CheckpointReader r(dir.string(), dense);
    r.load("sdf", sdf_loaded);
  }

  size_t mismatches = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = dense.spans()[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t hz = span.halo_z();
    const int64_t lz = span.logic_extent().z;
    const int64_t z0 = span.z_offset();
    const float* h = sdf_loaded.cpu_buffers()[g].data();
    for (int64_t bz = hz; bz < hz + lz; ++bz) {
      const int64_t gz = z0 + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = bz * pitch + y * haloed.x + x;
          const float got = h[off];
          const float want = sphere_sdf(static_cast<float>(x), static_cast<float>(y),
                                        static_cast<float>(gz), cx, cy, cz, radius);
          if (got != want) {
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    CARBON_LOG_ERROR("checkpoint_sphere_image: restart verification failed ({} mismatches)",
                     mismatches);
    return EXIT_FAILURE;
  }

  const int64_t gz_slice = global.z / 2;
  std::string out_path = "sphere_sdf_slice.pgm";
  if (argc >= 2 && argv[1] != nullptr) {
    out_path = argv[1];
  }

  if (ranks == 1 && rank == 0) {
    int64_t iw = 0;
    int64_t ih = 0;
    const std::vector<float> img =
        downsample_xy_slice(dense, sdf_loaded, gz_slice, kStride, &iw, &ih);
    if (!write_pgm_gray(out_path, img, iw, ih)) {
      CARBON_LOG_ERROR("checkpoint_sphere_image: failed to write '{}'", out_path);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("checkpoint_sphere_image: wrote downsampled slice (z={}, stride={}) to '{}'",
                    gz_slice, kStride, out_path);
  } else if (rank == 0) {
    CARBON_LOG_INFO(
        "checkpoint_sphere_image: skipping PGM export under MPI world_size={} (each rank holds a "
        "z-shard only). Run with mpiexec -n 1 to produce '{}'.",
        ranks, out_path);
  }

  barrier(sys);
  if (rank == 0) {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
  CARBON_LOG_INFO("checkpoint_sphere_image: rank {} of {} OK ({} GPU(s))", rank, ranks, n_gpus);
  return EXIT_SUCCESS;
}
