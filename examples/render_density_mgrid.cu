// Parallel density rendering on an adaptive carbon::MultiResBlock tower: analytic
// SDF seeded per level, Gaussian screen splats from every retained level into one
// shared DensityImage (4K Netpbm export). Finest bricks hug the surface; coarser
// levels fill the deep interior so a single image shows the resolution transition.
//
// The scene is a sphere unioned with an x-aligned capped cylinder that intersects
// it, composed with the defining property of a signed distance function: the SDF
// of a union is the pointwise min of the operand SDFs. Because both operands are
// exact distances the union volume has a closed form, so the 3D mass integral
// stays a real self-check (see k_* constants and scene_volume below).
//
// Build & run
// -----------
//   cmake --build build --target carbon_example_render_density_mgrid
//   ./build/examples/carbon_example_render_density_mgrid
//   ./build/examples/carbon_example_render_density_mgrid out.ppm   # green on white
//   ./build/examples/carbon_example_render_density_mgrid out.pgm   # grayscale P5
//   ./build/examples/carbon_example_render_density_mgrid --level 0 out.ppm
//
// With MPI (combine folds cross-rank contributions; only rank 0 writes the image):
//   cmake --build build-mpi --target carbon_example_render_density_mgrid
//   mpiexec -n 1 ./build-mpi/examples/carbon_example_render_density_mgrid

#include <carbon/carbon.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/mgrid.h>
#include <carbon/render.h>
#include <carbon/system.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef CARBON_HAVE_VIEWER
#include "viewer/carbon_viewer.h"
#include "viewer/render_rgb_utils.h"
#endif

namespace {

using SBlock = carbon::DefaultMultiResBlock;
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;

constexpr int k_depth = 3;
constexpr int k_samples_per_axis = 2;
constexpr float k_sdf_edge_width = 3.0F;
constexpr float k_splat_sigma_px = 1.75F;
constexpr float k_band_fine = 8.0F;
constexpr float k_band_outer = 48.0F;
constexpr float k_mass_rel_tol = 0.15F;

// Cylinder shape as a fraction of the sphere radius. The closed-form union volume
// below needs cyl_radius < radius and cyl_half_len >= radius (the cylinder must
// span the sphere along its axis); both fractions honour that. cyl_radius also
// exceeds k_band_outer so the cylinder core is deep enough to reach the coarsest
// level, which keeps all three levels exercised inside the cylinder too.
constexpr float k_cyl_radius_frac = 0.40F;
constexpr float k_cyl_half_len_frac = 1.30F;

// Green ink on a white page: empty pixels stay white and the densest reach k_ink.
// Colour output needs P6, so the default export is .ppm; a .pgm path still writes
// the grayscale P5 (white-on-black) form.
constexpr const char* k_default_out = "render_density_mgrid.ppm";
constexpr carbon::float32_3d k_ink{0.0F, 0.42F, 0.12F};
constexpr carbon::float32_3d k_page{1.0F, 1.0F, 1.0F};

#ifdef CARBON_HAVE_MPI_NCCL
[[nodiscard]] auto
world_rank(const carbon::System& sys) noexcept -> int {
  if (const auto* d = sys.distributed(); d != nullptr) {
    return d->world_rank();
  }
  return 0;
}
#else
[[nodiscard]] auto
world_rank(const carbon::System& /*sys*/) noexcept -> int {
  return 0;
}
#endif

// Sphere unioned with an x-aligned capped cylinder, both centred on the same point.
struct Scene {
  float cx{};
  float cy{};
  float cz{};
  float radius{};
  float cyl_radius{};
  float cyl_half_len{};
};

[[nodiscard]] CARBON_HD auto
sphere_sdf(float px,
           float py,
           float pz,
           const Scene& s) noexcept -> float {
  const float dx = px - s.cx;
  const float dy = py - s.cy;
  const float dz = pz - s.cz;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - s.radius;
}

// Exact distance to a capped cylinder whose axis runs along x. In the (radial,
// axial) half-plane the solid is a rectangle, so the usual 2D box distance applies:
// negative inside, and outside it is the length of the componentwise-positive part.
[[nodiscard]] CARBON_HD auto
cylinder_x_sdf(float px,
               float py,
               float pz,
               const Scene& s) noexcept -> float {
  const float dy = py - s.cy;
  const float dz = pz - s.cz;
  const float radial = std::sqrt(dy * dy + dz * dz) - s.cyl_radius;
  const float axial = std::fabs(px - s.cx) - s.cyl_half_len;
  const float out_r = std::fmax(radial, 0.0F);
  const float out_a = std::fmax(axial, 0.0F);
  return std::fmin(std::fmax(radial, axial), 0.0F) + std::sqrt(out_r * out_r + out_a * out_a);
}

// Defining property of an SDF used here: the union of two solids is the pointwise
// min of their signed distances. Outside both operands the nearer surface is the
// nearest surface of the union; inside either operand the point is inside the union.
[[nodiscard]] CARBON_HD auto
scene_sdf(float px,
          float py,
          float pz,
          const Scene& s) noexcept -> float {
  return std::fmin(sphere_sdf(px, py, pz, s), cylinder_x_sdf(px, py, pz, s));
}

// Exact volume of sphere ∪ cylinder when the axis passes through the sphere centre,
// a = cyl_radius < radius, and L = cyl_half_len >= radius. The cylinder then spans
// the sphere, so their intersection is the spherical core within distance a of the
// axis, of volume (4pi/3)(R^3 - (R^2 - a^2)^{3/2}). Adding the sphere and cylinder
// volumes and subtracting that leaves 2pi a^2 L + (4pi/3)(R^2 - a^2)^{3/2}.
[[nodiscard]] auto
scene_volume(const Scene& s) -> double {
  const double r = static_cast<double>(s.radius);
  const double a = static_cast<double>(s.cyl_radius);
  const double l = static_cast<double>(s.cyl_half_len);
  const double core = std::pow(r * r - a * a, 1.5);
  return 2.0 * M_PI * a * a * l + (4.0 / 3.0) * M_PI * core;
}

[[nodiscard]] CARBON_HD auto
level_scale(int level,
            int ref_factor) noexcept -> int {
  int scale = 1;
  for (int i = 0; i < level; ++i) {
    scale *= ref_factor;
  }
  return scale;
}

// Refinement level a point wants, from its distance to the nearest scene surface.
// Returns -1 for the far exterior, which no level claims.
[[nodiscard]] auto
desired_level(float sdf) -> int {
  if (sdf >= k_band_outer) {
    return -1;
  }
  const float adf = std::abs(sdf);
  if (adf < k_band_fine) {
    return 0;
  }
  if (adf < k_band_outer) {
    return 1;
  }
  return k_depth - 1;
}

// The level bands must be evaluated at a level-INDEPENDENT position. Testing each
// level at its own voxel centre does not partition space: near a band boundary the
// coarse centre and its fine children can disagree, leaving a region claimed by two
// levels (double-splatted) or by none (a hole). Both show up as concentric rings.
// Snapping every voxel to the coarsest cell containing it makes the assignment an
// exact partition at coarsest-cell granularity.
[[nodiscard]] auto
coarse_anchor(const carbon::int64_3d& voxel,
              int level,
              int ref_factor) -> carbon::float32_3d {
  const int64_t scale = static_cast<int64_t>(level_scale(level, ref_factor));
  const int64_t coarsest = static_cast<int64_t>(level_scale(k_depth - 1, ref_factor));
  const int64_t fx = (voxel.x * scale) / coarsest;
  const int64_t fy = (voxel.y * scale) / coarsest;
  const int64_t fz = (voxel.z * scale) / coarsest;
  const float c = static_cast<float>(coarsest);
  return {(static_cast<float>(fx) + 0.5F) * c, (static_cast<float>(fy) + 0.5F) * c,
          (static_cast<float>(fz) + 0.5F) * c};
}

[[nodiscard]] auto
is_active_at_level(int level,
                   const carbon::int64_3d& voxel,
                   int ref_factor,
                   const Scene& scene) -> bool {
  const carbon::float32_3d a = coarse_anchor(voxel, level, ref_factor);
  return desired_level(scene_sdf(a.x, a.y, a.z, scene)) == level;
}

[[nodiscard]] auto
frame_sum(const carbon::DensityImage& image) -> double {
  double acc = 0.0;
  for (float v : image.host_frame()) {
    acc += static_cast<double>(v);
  }
  return acc;
}

// Guards the invariant the level bands have to satisfy: every point of the rendered
// shell is claimed by exactly one level. A band evaluated per-level instead of at a
// shared anchor breaks it, double-splatting some shells and leaving others empty.
//
// Coverage is accumulated at coarsest-cell granularity, which is where the anchored
// assignment is exact: each active level-L voxel contributes the scale^3 finest
// cells it occupies, so a correctly claimed coarse cell totals exactly coarsest^3
// and a culled one totals 0. Anything else is a gap, a partial claim, or a
// double-claim. Unlike the projected image this says nothing about the scene shape,
// so it holds for any SDF. Returns the number of offending coarse cells.
[[nodiscard]] auto
level_partition_error(const carbon::MultiResBlock<SBlock>& mblock,
                      int ref_factor,
                      const Scene& scene) -> uint64_t {
  const int64_t coarsest = static_cast<int64_t>(level_scale(k_depth - 1, ref_factor));
  const carbon::int64_3d finest = mblock.finest_extent();
  const carbon::int64_3d nc{(finest.x + coarsest - 1) / coarsest,
                            (finest.y + coarsest - 1) / coarsest,
                            (finest.z + coarsest - 1) / coarsest};
  const size_t cells =
      static_cast<size_t>(nc.x) * static_cast<size_t>(nc.y) * static_cast<size_t>(nc.z);
  std::vector<int32_t> cover(cells, 0);

  for (int level = 0; level < mblock.depth(); ++level) {
    const int64_t scale = static_cast<int64_t>(level_scale(level, ref_factor));
    const int32_t contribution = static_cast<int32_t>(scale * scale * scale);
    const auto& grid = mblock.grid(level);
    for (size_t g = 0; g < mblock.system().gpu_group().size(); ++g) {
      const uint32_t nb = grid.block_count(g);
      for (uint32_t b = 0; b < nb; ++b) {
        const auto& bm = grid.host_bitmasks(g)[b];
        const auto origin = grid.host_origins(g)[b];
        for (int z = 0; z < SBlock::MemZ; ++z) {
          for (int y = 0; y < SBlock::MemY; ++y) {
            for (int x = 0; x < SBlock::MemX; ++x) {
              if (!bm.is_active(x, y, z)) {
                continue;
              }
              const int64_t ix = (origin.x + x) * scale / coarsest;
              const int64_t iy = (origin.y + y) * scale / coarsest;
              const int64_t iz = (origin.z + z) * scale / coarsest;
              if (ix < 0 || ix >= nc.x || iy < 0 || iy >= nc.y || iz < 0 || iz >= nc.z) {
                continue;
              }
              cover[static_cast<size_t>(ix) +
                    static_cast<size_t>(nc.x) *
                        (static_cast<size_t>(iy) +
                         static_cast<size_t>(nc.y) * static_cast<size_t>(iz))] += contribution;
            }
          }
        }
      }
    }
  }

  const int32_t full = static_cast<int32_t>(coarsest * coarsest * coarsest);
  const float anchor_step = static_cast<float>(coarsest);
  uint64_t bad = 0;
  for (int64_t iz = 0; iz < nc.z; ++iz) {
    for (int64_t iy = 0; iy < nc.y; ++iy) {
      for (int64_t ix = 0; ix < nc.x; ++ix) {
        // Same anchor coarse_anchor() derives, so the expectation is exact.
        const float ax = (static_cast<float>(ix) + 0.5F) * anchor_step;
        const float ay = (static_cast<float>(iy) + 0.5F) * anchor_step;
        const float az = (static_cast<float>(iz) + 0.5F) * anchor_step;
        const int32_t want = (desired_level(scene_sdf(ax, ay, az, scene)) >= 0) ? full : 0;
        const size_t lin =
            static_cast<size_t>(ix) +
            static_cast<size_t>(nc.x) *
                (static_cast<size_t>(iy) + static_cast<size_t>(nc.y) * static_cast<size_t>(iz));
        if (cover[lin] != want) {
          ++bad;
        }
      }
    }
  }
  return bad;
}

[[nodiscard]] auto
host_mass_3d(const carbon::MultiResBlock<SBlock>& mblock,
             typename carbon::MultiResBlock<SBlock>::template Field<float,
                                                                    1>& sdf,
             const carbon::Stream& stream,
             int ref_factor) -> double {
  const size_t n_gpus = mblock.system().gpu_group().size();
  double mass = 0.0;
  for (int level = 0; level < mblock.depth(); ++level) {
    const int scale = level_scale(level, ref_factor);
    const double voxel_volume =
        static_cast<double>(scale) * static_cast<double>(scale) * static_cast<double>(scale);
    const auto& grid = mblock.grid(level);
    const auto host = grid.download_to_host<float, 1>(sdf.level_field(level), stream);
    stream.wait();
    for (size_t g = 0; g < n_gpus; ++g) {
      const uint32_t nb = grid.block_count(g);
      for (uint32_t b = 0; b < nb; ++b) {
        const auto& bm = grid.host_bitmasks(g)[b];
        const auto origin = grid.host_origins(g)[b];
        for (int z = 0; z < SBlock::MemZ; ++z) {
          for (int y = 0; y < SBlock::MemY; ++y) {
            for (int x = 0; x < SBlock::MemX; ++x) {
              if (!bm.is_active(x, y, z)) {
                continue;
              }
              const size_t mem_x = static_cast<size_t>(SBlock::MemX);
              const size_t mem_y = static_cast<size_t>(SBlock::MemY);
              const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SBlock::VoxelCount) +
                                 static_cast<size_t>(x) + mem_x * static_cast<size_t>(y) +
                                 mem_x * mem_y * static_cast<size_t>(z);
              const float stored = host[g].data()[lin];
              const float rho = carbon::render::sdf_smooth_density(stored, k_sdf_edge_width);
              mass += static_cast<double>(rho) * voxel_volume;
            }
          }
        }
      }
    }
  }
  return mass;
}

[[nodiscard]] auto
total_brick_count(const carbon::MultiResBlock<SBlock>& mblock) -> uint64_t {
  uint64_t total = 0;
  const size_t n_gpus = mblock.system().gpu_group().size();
  for (int level = 0; level < mblock.depth(); ++level) {
    const auto& grid = mblock.grid(level);
    for (size_t g = 0; g < n_gpus; ++g) {
      total += static_cast<uint64_t>(grid.block_count(g));
    }
  }
  return total;
}

}  // namespace

auto
main(int argc,
     char** argv) -> int {
  carbon::init();

#ifdef CARBON_HAVE_MPI_NCCL
  auto distributed = std::make_shared<const carbon::Distributed>();
  const int ranks = distributed->world_size();
  carbon::System sys =
      (ranks > 1) ? carbon::System(distributed) : carbon::System(carbon::get_all_devices());
#else
  carbon::System sys(carbon::get_all_devices());
#endif

  if (sys.gpu_group().size() == 0U) {
    CARBON_LOG_ERROR("render_density_mgrid: no devices visible");
    return EXIT_FAILURE;
  }

  std::string out_path = k_default_out;
  bool interactive = false;
  int splat_level = -1;
  int64_t domain_n = 512;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--interactive") {
      interactive = true;
    } else if (arg == "--level" && i + 1 < argc) {
      splat_level = std::atoi(argv[++i]);
    } else if (arg.size() > 0 && arg[0] != '-') {
      const char* end = nullptr;
      const long parsed = std::strtol(arg.c_str(), const_cast<char**>(&end), 10);
      const bool is_count = end != arg.c_str() && *end == '\0' && parsed > 0;
      if (is_count && out_path == k_default_out) {
        domain_n = parsed;
      } else if (out_path == k_default_out || arg[0] != '-') {
        out_path = arg;
      }
    }
  }

#ifndef CARBON_HAVE_VIEWER
  (void)interactive;
#endif

  const int ref_factor = SBlock::UserX;
  const int z_align = SBlock::MemZ * static_cast<int>(std::pow(ref_factor, k_depth - 1));
  if (sys.gpu_group().size() > 1U && domain_n % z_align != 0) {
    CARBON_LOG_ERROR(
        "render_density_mgrid: multi-GPU run needs N ({}) a multiple of MemZ*R^(depth-1) ({})",
        domain_n, z_align);
    return EXIT_FAILURE;
  }
  if (splat_level >= k_depth || splat_level < -1) {
    CARBON_LOG_ERROR("render_density_mgrid: --level must be in [-1, {}), got {}", k_depth - 1,
                     splat_level);
    return EXIT_FAILURE;
  }

  const carbon::int64_3d finest{domain_n, domain_n, domain_n};
  const float cx = 0.5F * static_cast<float>(finest.x - 1);
  const float cy = 0.5F * static_cast<float>(finest.y - 1);
  const float cz = 0.5F * static_cast<float>(finest.z - 1);
  const float radius = 0.35F * static_cast<float>(finest.x);
  const Scene scene{cx, cy, cz, radius, k_cyl_radius_frac * radius, k_cyl_half_len_frac * radius};

  // The closed-form union volume assumes the whole scene is inside the domain; a
  // cylinder cap clipped by the boundary would silently break the mass check.
  if (scene.cx - scene.cyl_half_len < 0.0F ||
      scene.cx + scene.cyl_half_len > static_cast<float>(finest.x)) {
    CARBON_LOG_ERROR("render_density_mgrid: cylinder half-length {:.1f} does not fit in N={}",
                     scene.cyl_half_len, finest.x);
    return EXIT_FAILURE;
  }

  const auto is_active = [ref_factor, scene](int level, const carbon::int64_3d& voxel) -> bool {
    return is_active_at_level(level, voxel, ref_factor, scene);
  };

  carbon::MultiResBlock<SBlock> mblock(sys, finest, k_depth, is_active, ref_factor,
                                       "render_density_mgrid");

  auto sdf = mblock.newField<float, 1>(carbon::MemorySide::Device, "sdf");
  const carbon::Stream stream = sys.stream();

  for (int level = 0; level < mblock.depth(); ++level) {
    const int scale = level_scale(level, ref_factor);
    const float spacing = static_cast<float>(scale);
    const auto seed =
        mblock.make_kernel(level, [&sdf, level, spacing, scene](carbon::KernelLoader& loader) {
          auto out = loader.write(sdf.level_view(level));
          return [out, spacing, scene] CARBON_HD(Span span, Idx idx) -> void {
            const carbon::int64_3d gc = span.global_location(idx);
            const float fx = (static_cast<float>(gc.x) + 0.5F) * spacing;
            const float fy = (static_cast<float>(gc.y) + 0.5F) * spacing;
            const float fz = (static_cast<float>(gc.z) + 0.5F) * spacing;
            out.write(span, idx, 0, scene_sdf(fx, fy, fz, scene));
          };
        });
    seed.launch(stream);
  }
  stream.wait();

  if (world_rank(sys) == 0) {
    for (int level = 0; level < mblock.depth(); ++level) {
      uint64_t bricks = 0;
      for (size_t g = 0; g < sys.gpu_group().size(); ++g) {
        bricks += static_cast<uint64_t>(mblock.grid(level).block_count(g));
      }
      CARBON_LOG_INFO("render_density_mgrid: level {} bricks={}", level, bricks);
    }
    CARBON_LOG_INFO("render_density_mgrid: total bricks={}", total_brick_count(mblock));
  }

  constexpr int k_frame_w = 3840;
  constexpr int k_frame_h = 2160;
  // Frame the whole union: the cylinder caps reach past the sphere along x.
  const float half_x = std::fmax(radius, scene.cyl_half_len);
  const carbon::float32_3d bounds_min{cx - half_x, cy - radius, cz - radius};
  const carbon::float32_3d bounds_max{cx + half_x, cy + radius, cz + radius};

  carbon::DensityImage image(sys, k_frame_w, k_frame_h);

  const auto render_for_camera = [&](const carbon::Camera& camera,
                                     carbon::DensityImage& out_image) {
    out_image.clear(stream);
    const int level_begin = (splat_level < 0) ? 0 : splat_level;
    const int level_end = (splat_level < 0) ? mblock.depth() : splat_level + 1;
    for (int level = level_begin; level < level_end; ++level) {
      const int scale = level_scale(level, ref_factor);
      const float block_spacing = static_cast<float>(scale);
      const float vol = block_spacing * block_spacing * block_spacing;
      const int samples = k_samples_per_axis * scale;
      const auto splat = carbon::make_density_kernel(
          mblock.grid(level), out_image, camera,
          [&sdf, level, vol](carbon::KernelLoader& loader) {
            const auto part = loader.read(sdf.level_view(level));
            return [part, vol] CARBON_HD(Span span, Idx idx) -> float {
              const float stored = part.read(span, idx, 0);
              return carbon::render::sdf_smooth_density(stored, k_sdf_edge_width) * vol;
            };
          },
          {}, block_spacing, samples, k_splat_sigma_px);
      splat.launch(stream);
    }
    out_image.combine(stream);
    stream.wait();
  };

#ifdef CARBON_HAVE_VIEWER
  bool use_interactive = interactive;
  if (use_interactive && world_rank(sys) == 0 && !carbon::viewer::Viewer::display_available()) {
    CARBON_LOG_WARN("render_density_mgrid: no display; falling back to file output");
    use_interactive = false;
  }
  if (use_interactive) {
    const auto orbit = carbon::viewer::OrbitCamera::from_frame_bounds(
        bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}, 0.785398F);
    carbon::viewer::Viewer viewer(k_frame_w, k_frame_h, "render_density_mgrid");
    viewer.run(sys, orbit,
               [&](const carbon::Camera& camera, int /*w*/, int /*h*/, int mode,
                   std::vector<float>& rgb_out) {
                 render_for_camera(camera, image);
                 carbon::viewer::rgb_from_density(image, rgb_out, mode);
               });
    return EXIT_SUCCESS;
  }
#endif

  render_for_camera(
      carbon::Camera::frame_bounds(bounds_min, bounds_max, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F},
                                   0.785398F, k_frame_w, k_frame_h),
      image);

  if (world_rank(sys) == 0) {
    const double frame_mass = frame_sum(image);
    if (frame_mass <= 0.0) {
      CARBON_LOG_ERROR("render_density_mgrid: empty frame (sum={})", frame_mass);
      return EXIT_FAILURE;
    }

    const double mass_3d = host_mass_3d(mblock, sdf, stream, ref_factor);
    const double expected_volume = scene_volume(scene);
    const double rel = std::abs(mass_3d - expected_volume) / expected_volume;
    CARBON_LOG_INFO(
        "render_density_mgrid: frame sum={:.6g} mass_3d={:.6g} expected_vol={:.6g} "
        "rel_err={:.4g}",
        frame_mass, mass_3d, expected_volume, rel);
    if (rel > static_cast<double>(k_mass_rel_tol)) {
      CARBON_LOG_ERROR("render_density_mgrid: 3D mass rel err {:.4g} exceeds tol {:.4g}", rel,
                       k_mass_rel_tol);
      return EXIT_FAILURE;
    }

    const uint64_t bad_cells = level_partition_error(mblock, ref_factor, scene);
    CARBON_LOG_INFO("render_density_mgrid: level-partition mismatched coarse cells={}", bad_cells);
    if (bad_cells != 0) {
      CARBON_LOG_ERROR("render_density_mgrid: {} coarse cells are not claimed by exactly one level",
                       bad_cells);
      return EXIT_FAILURE;
    }

    const bool wrote = out_path.ends_with(".pgm") ? image.write_pgm(out_path)
                                                  : image.write_ppm(out_path, k_ink, k_page);
    if (!wrote) {
      CARBON_LOG_ERROR("render_density_mgrid: failed to write {}", out_path);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("render_density_mgrid: wrote {}", out_path);
  }

  return EXIT_SUCCESS;
}
