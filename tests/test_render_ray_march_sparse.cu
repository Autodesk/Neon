// Ray marching on the sparse grid families (Block, MultiResBlock).
//
// Until the brick-index locator landed, make_ray_march_kernel static_asserted on the dense
// z-slab iteration model, so a sparse grid could only be drawn by forward splatting. The
// march is now driven entirely through a grid-generic sampler (detail/render_sample.h), and
// what needs pinning is that the sparse world-to-cell map really is the same map the dense
// one implements.
//
// So the main test is a parity test rather than a "does it draw something" test: the same
// analytic sphere SDF, the same camera, and the same shading options are rendered through
// Dense and through a sparse Block that retains only the bricks near the surface, and the
// two silhouettes have to agree pixel for pixel. A wrong origin convention, a half-cell
// offset, a mis-scaled spacing or a dropped GPU slab all move the silhouette and would show
// up here; none of them would show up in a visibility check.
//
// Run this at more than one GPU count. Both grids gather across their slab seams through a
// halo, so the agreement is not just high but identical and GPU-count-independent; a seam
// defect shows up as agreement that decays as the slabs get thinner, which a single-count
// run reads as ordinary discretisation noise.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/devices.h>
#include <carbon/halo.h>
#include <carbon/halo_block.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/mgrid.h>
#include <carbon/render.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using MSB = carbon::DefaultMultiResBlock;
using MGrid = carbon::MultiResBlock<MSB>;

constexpr int64_t kN = 64;
constexpr int k_w = 160;
constexpr int k_h = 160;

constexpr float kCentre = 0.5F * static_cast<float>(kN - 1);
constexpr float kRadius = 0.28F * static_cast<float>(kN);

// Block / mGrid voxels are cell-centred (voxel v sits at world_origin + (v + 0.5) * spacing)
// while dense cells are node-centred (cell g sits at origin + g * spacing). Shifting the
// sparse world origin back by half a cell makes voxel v and node v the same point, which is
// what lets the two renders be compared pixel for pixel.
constexpr carbon::float32_3d kBlockOrigin{-0.5F, -0.5F, -0.5F};

/** Signed distance to the test sphere at an arbitrary world point. */
[[nodiscard]] CARBON_HD auto
sphere_sdf(float x,
           float y,
           float z) -> float {
  const float dx = x - kCentre;
  const float dy = y - kCentre;
  const float dz = z - kCentre;
  return std::sqrt(dx * dx + dy * dy + dz * dz) - kRadius;
}

// The SurfaceBlinn band is exp(-(sdf / width)^2) and the march drops samples beyond
// 4 * width, so retaining |sdf| <= 4 * width keeps every voxel that can contribute. Below
// that the sparse render would legitimately differ from the dense one and the parity check
// would be measuring the shell radius rather than the coordinate map.
constexpr float kIsoWidth = 3.0F;
constexpr float kShellHalfWidth = 4.0F * kIsoWidth;

[[nodiscard]] auto
shell_active(const carbon::int64_3d& v) -> bool {
  const float x = kBlockOrigin.x + static_cast<float>(v.x) + 0.5F;
  const float y = kBlockOrigin.y + static_cast<float>(v.y) + 0.5F;
  const float z = kBlockOrigin.z + static_cast<float>(v.z) + 0.5F;
  return std::fabs(sphere_sdf(x, y, z)) <= kShellHalfWidth;
}

[[nodiscard]] auto
sphere_camera() -> carbon::Camera {
  const carbon::float32_3d bmin{kCentre - kRadius, kCentre - kRadius, kCentre - kRadius};
  const carbon::float32_3d bmax{kCentre + kRadius, kCentre + kRadius, kCentre + kRadius};
  return carbon::Camera::frame_bounds(bmin, bmax, {0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F},
                                      0.785398F, k_w, k_h);
}

[[nodiscard]] auto
surface_opts() -> carbon::RayMarchOptions {
  carbon::RayMarchOptions opts{};
  opts.shading = carbon::RayMarchShading::SurfaceBlinn;
  opts.absorption_scale = 2.0F;
  opts.isosurface_width = kIsoWidth;
  opts.light_dir = {0.3F, -0.85F, 0.45F};
  opts.ambient = 0.12F;
  opts.diffuse = 0.62F;
  opts.specular = 0.45F;
  opts.shininess = 64.0F;
  opts.material_color = {0.92F, 0.94F, 1.0F};
  opts.step_size = 0.25F;
  return opts;
}

/** Per-pixel "the surface was hit here", the silhouette the parity check compares. */
[[nodiscard]] auto
silhouette(const carbon::RayMarchImage& image) -> std::vector<char> {
  const auto& trans = image.host_transmittance();
  std::vector<char> hit(trans.size(), 0);
  for (size_t i = 0; i < trans.size(); ++i) {
    hit[i] = (trans[i] < 0.5F) ? 1 : 0;
  }
  return hit;
}

[[nodiscard]] auto
count_hits(const std::vector<char>& hit) -> size_t {
  size_t n = 0;
  for (char h : hit) {
    n += static_cast<size_t>(h);
  }
  return n;
}

// --------------------------------------------------------------------------------------
// Dense reference
// --------------------------------------------------------------------------------------

[[nodiscard]] auto
render_dense(carbon::System& sys,
             const carbon::Stream& stream) -> std::vector<char> {
  using Span = carbon::Dense::Span;
  using Idx = carbon::Dense::Idx;

  const carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const carbon::Dense dense(sys, {kN, kN, kN}, stencil, carbon::BoundaryMode::Open, {}, 1.0F);
  auto sdf = dense.newField<float, 1>(carbon::MemorySide::Device, "sdf");

  const carbon::Kernel fill = dense.make_kernel([&sdf](carbon::KernelLoader& loader) {
    auto out = loader.write(sdf);
    return [out] CARBON_HD(Span s, Idx i) -> void {
      const carbon::int64_3d g = s.global_location(i);
      const carbon::float32_3d o = s.origin();
      const float sp = s.spacing();
      out.write(s, i, 0,
                sphere_sdf(o.x + static_cast<float>(g.x) * sp, o.y + static_cast<float>(g.y) * sp,
                           o.z + static_cast<float>(g.z) * sp));
    };
  });
  fill.launch(stream);
  const carbon::Halo<float, 1> halo(sdf);
  halo(stream);
  stream.wait();

  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);
  const carbon::Kernel march = carbon::make_ray_march_kernel(
      dense, image, sphere_camera(), surface_opts(), [&sdf](carbon::KernelLoader& loader) {
        const auto field = loader.read(sdf);
        return [field] CARBON_HD(Span s, Idx i) -> float {
          return field.read(s, i, 0);
        };
      });
  march.launch(stream);
  image.combine(stream);
  image.dist_combine();
  return silhouette(image);
}

// --------------------------------------------------------------------------------------
// Block
// --------------------------------------------------------------------------------------

[[nodiscard]] auto
render_block(carbon::System& sys,
             const carbon::Stream& stream,
             size_t& retained_bricks) -> std::vector<char> {
  using Span = BlockGrid::Span;
  using Idx = BlockGrid::Idx;

  const BlockGrid block(sys, {kN, kN, kN}, shell_active, "ray_march_block");
  retained_bricks = block.block_count();
  auto sdf = block.newField<float, 1>(carbon::MemorySide::Device, "sdf");

  const carbon::Kernel fill = block.make_kernel([&sdf](carbon::KernelLoader& loader) {
    auto out = loader.write(sdf);
    return [out] CARBON_HD(Span s, Idx i) -> void {
      const carbon::int64_3d v = s.global_location(i);
      out.write(s, i, 0,
                sphere_sdf(kBlockOrigin.x + static_cast<float>(v.x) + 0.5F,
                           kBlockOrigin.y + static_cast<float>(v.y) + 0.5F,
                           kBlockOrigin.z + static_cast<float>(v.z) + 0.5F));
    };
  });
  fill.launch(stream);
  // Same reason the Dense case above exchanges halos: a trilinear gather anchored at an
  // owned cell next to a slab seam reaches one voxel into the ghost ring, and the locator
  // now resolves there. Without the exchange those corners would hold stale data and the
  // seam would show up as a silhouette mismatch that grows with the GPU count.
  carbon::HaloBlock<float, 1, SB> halo(block, sdf);
  halo(stream);
  stream.wait();

  carbon::RayMarchOptions opts = surface_opts();
  opts.world_origin = kBlockOrigin;
  opts.world_spacing = 1.0F;

  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);
  const carbon::Kernel march = carbon::make_ray_march_kernel(
      block, image, sphere_camera(), opts, [&sdf](carbon::KernelLoader& loader) {
        const auto field = loader.read(sdf);
        return [field] CARBON_HD(Span s, Idx i) -> float {
          return field.read(s, i, 0);
        };
      });
  march.launch(stream);
  image.combine(stream);
  image.dist_combine();
  return silhouette(image);
}

// --------------------------------------------------------------------------------------
// MultiResBlock
// --------------------------------------------------------------------------------------

constexpr int kDepth = 2;
constexpr int64_t kMFinestZ = 128;  // Keeps the coarsest brick z-extent >= the slab count.

// The finest level is refined over only *half* the sphere; the coarse level covers the whole
// box. So a ray crossing the sphere reads level 0 on one side of the split and level 1 on the
// other, and find_finest has to make that choice per sample. Refining the whole shell instead
// would leave the coarse level unreachable — the tower would render byte-identically to the
// flat Block above and the level-selection path would be untested.
constexpr int64_t kRefinedXEnd = kN / 2;

[[nodiscard]] auto
mgrid_active(int level,
             const carbon::int64_3d& v) -> bool {
  if (level > 0) {
    return true;
  }
  return v.x < kRefinedXEnd && shell_active(v);
}

template <typename Partition, int Levels>
struct LevelReader {
  Partition part[Levels]{};

  [[nodiscard]] CARBON_HD auto
  operator()(int level,
             const MGrid::Span& span,
             const MGrid::Idx& idx) const -> float {
    return part[level].read(span, idx, 0);
  }
};

[[nodiscard]] auto
render_mgrid(carbon::System& sys,
             const carbon::Stream& stream) -> std::vector<char> {
  using Span = MGrid::Span;
  using Idx = MGrid::Idx;

  const MGrid mgrid(sys, {kN, kN, kMFinestZ}, kDepth, mgrid_active, MSB::UserX, "ray_march_mgrid");
  auto sdf = mgrid.newField<float, 1>(carbon::MemorySide::Device, "sdf");

  for (int l = 0; l < kDepth; ++l) {
    const float sp = static_cast<float>(mgrid.descriptor().voxel_spacing(l));
    const carbon::Kernel fill = mgrid.make_kernel(l, [&sdf, l, sp](carbon::KernelLoader& loader) {
      auto out = loader.write(sdf.level_view(l));
      return [out, sp] CARBON_HD(Span s, Idx i) -> void {
        const carbon::int64_3d v = s.global_location(i);
        out.write(s, i, 0,
                  sphere_sdf(kBlockOrigin.x + (static_cast<float>(v.x) + 0.5F) * sp,
                             kBlockOrigin.y + (static_cast<float>(v.y) + 0.5F) * sp,
                             kBlockOrigin.z + (static_cast<float>(v.z) + 0.5F) * sp));
      };
    });
    fill.launch(stream);
    sdf.make_halo_kernel(l).launch(stream);
  }
  stream.wait();

  carbon::RayMarchOptions opts = surface_opts();
  opts.world_origin = kBlockOrigin;
  opts.world_spacing = 1.0F;

  carbon::RayMarchImage image(sys, k_w, k_h);
  image.clear(stream);
  const carbon::Kernel march = carbon::make_ray_march_kernel(
      mgrid, image, sphere_camera(), opts, [&sdf](carbon::KernelLoader& loader) {
        using PartitionT = decltype(loader.read(sdf.level_view(0)));
        LevelReader<PartitionT, kDepth> reader{};
        for (int l = 0; l < kDepth; ++l) {
          reader.part[l] = loader.read(sdf.level_view(l));
        }
        return reader;
      });
  march.launch(stream);
  image.combine(stream);
  image.dist_combine();
  return silhouette(image);
}

/** Fraction of pixels on which two silhouettes agree. */
[[nodiscard]] auto
agreement(const std::vector<char>& a,
          const std::vector<char>& b) -> double {
  size_t same = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    same += static_cast<size_t>(a[i] == b[i]);
  }
  return static_cast<double>(same) / static_cast<double>(a.size());
}

}  // namespace

auto
main() -> int {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_render_ray_march_sparse: skipping (no CUDA devices visible)");
    return EXIT_SUCCESS;
  }

  auto devs = carbon::get_all_devices();
  carbon::System sys(carbon::detail::GpuGroup{std::move(devs)});
  const carbon::Stream stream = sys.stream();

  const std::vector<char> dense_hit = render_dense(sys, stream);
  size_t retained = 0;
  const std::vector<char> block_hit = render_block(sys, stream, retained);
  const std::vector<char> mgrid_hit = render_mgrid(sys, stream);

  const size_t dense_n = count_hits(dense_hit);
  const size_t block_n = count_hits(block_hit);
  const size_t mgrid_n = count_hits(mgrid_hit);
  const size_t pixels = dense_hit.size();

  // A silhouette that covers a plausible slice of the frame. Without this the agreement
  // check below would pass trivially on two blank images.
  const size_t min_hits = pixels / 20;
  if (dense_n < min_hits) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march_sparse dense: only {} of {} pixels hit the sphere", dense_n, pixels);
  }
  if (block_n < min_hits) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march_sparse block: only {} of {} pixels hit the sphere ({} bricks "
        "retained); the march found no matter in the sparse grid",
        block_n, pixels, retained);
  }
  if (mgrid_n < min_hits) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march_sparse mgrid: only {} of {} pixels hit the sphere", mgrid_n, pixels);
  }

  // Both grids hold the same analytic field on the same lattice, and both gather across
  // slab seams through their halo, so the silhouettes are not merely close: they are equal,
  // at every GPU count. The threshold is set just below exact rather than at a comfortable
  // 99% because the interesting regression is small and count-dependent — before the
  // locator carried a ghost ring, seam corners fell back to the centre value and agreement
  // decayed from 0.9914 on two GPUs to 0.9878 on eight, which a 1% budget would have
  // absorbed silently.
  const double block_vs_dense = agreement(dense_hit, block_hit);
  if (block_vs_dense < 0.999) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march_sparse block_matches_dense: silhouettes agree on {:.4f} of pixels "
        "(dense {} hits, block {} hits over {} bricks)",
        block_vs_dense, dense_n, block_n, retained);
  }

  // The tower reads level 0 over the refined half and level 1 over the rest, so its
  // silhouette is the fine one on one side and a coarse-cell staircase on the other: looser
  // than the Block parity above, but still the same sphere.
  const double mgrid_vs_dense = agreement(dense_hit, mgrid_hit);
  if (mgrid_vs_dense < 0.99) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march_sparse mgrid_matches_dense: silhouettes agree on {:.4f} of pixels "
        "(dense {} hits, mgrid {} hits)",
        mgrid_vs_dense, dense_n, mgrid_n);
  }

  // Half the sphere comes off the coarse level, so the tower must *not* reproduce the flat
  // Block render. If it does, find_finest resolved everything to one level and the
  // multi-resolution path above is untested however green it looks.
  if (mgrid_hit == block_hit) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_render_ray_march_sparse mgrid_uses_coarse_level: the tower rendered exactly the "
        "flat Block silhouette, so the coarse level was never sampled");
  }

  CARBON_LOG_INFO(
      "test_render_ray_march_sparse: OK (dense {} hits, block {} hits / {} bricks @ {:.4f}, "
      "mgrid {} hits @ {:.4f})",
      dense_n, block_n, retained, block_vs_dense, mgrid_n, mgrid_vs_dense);
  return EXIT_SUCCESS;
}
