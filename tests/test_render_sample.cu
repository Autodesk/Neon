// Cross-family parity test for the grid-generic point sampler (detail/render_sample.h).
//
// The oracle is linear-field exactness: trilinear interpolation reproduces an affine field
// f(p) = a*p.x + b*p.y + c*p.z + d exactly, and central differences on it recover (a, b, c)
// exactly. So seeding every family with the same analytic field and querying at off-lattice
// positions checks the whole world-to-cell map at once — origin convention, spacing, slab
// z-offset, sparse brick lookup and multi-resolution level selection all have to be right or
// the interpolated value drifts off the plane.
//
// Every family is queried at (cell position + 0.25 cell) in all three axes, skipping cells
// whose eight-corner stencil would leave the slab's owned band or the domain rim — the two
// places where the samplers deliberately clamp or fall back, and where exactness therefore
// does not apply by design.
//
// The multi-resolution case is queried on the *coarse* level, where cells are R times wider.
// That is the part of the tower a single-level test cannot reach: it only comes out right if
// the per-level world frame (spacing = base * R^level) is wired correctly.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/dense_disg.h>
#include <carbon/detail/render_sample.h>
#include <carbon/devices.h>
#include <carbon/halo.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/mgrid.h>
#include <carbon/reduce.h>
#include <carbon/system.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace {

constexpr int64_t kNx = 32;
constexpr int64_t kNy = 32;
constexpr int64_t kNz = 64;

constexpr float kSpacing = 0.25F;
constexpr carbon::float32_3d kOrigin{-2.0F, 1.0F, 0.5F};

// Affine coefficients, deliberately unequal per axis so an axis mix-up cannot cancel out.
constexpr float kA = 0.7F;
constexpr float kB = -1.3F;
constexpr float kC = 0.45F;
constexpr float kD = 2.25F;

// The interpolation is exact in exact arithmetic; these bound float round-off over a field
// whose magnitude reaches ~10 across the domain, with the gradient divided by a 0.125 step.
constexpr float kValueTol = 2e-3F;
constexpr float kGradTol = 8e-3F;

[[nodiscard]] CARBON_HD auto
analytic(carbon::float32_3d p) -> float {
  return kA * p.x + kB * p.y + kC * p.z + kD;
}

/** World position of Cartesian cell @p g (node-centred, matching @c dense::Span ). */
[[nodiscard]] CARBON_HD auto
node_world(carbon::int64_3d g) -> carbon::float32_3d {
  return {
      kOrigin.x + static_cast<float>(g.x) * kSpacing,
      kOrigin.y + static_cast<float>(g.y) * kSpacing,
      kOrigin.z + static_cast<float>(g.z) * kSpacing,
  };
}

/** World position of sparse voxel @p v at spacing @p sp (cell-centred, matching @c Locator ). */
[[nodiscard]] CARBON_HD auto
cell_world(carbon::int64_3d v,
           float sp) -> carbon::float32_3d {
  return {
      kOrigin.x + (static_cast<float>(v.x) + 0.5F) * sp,
      kOrigin.y + (static_cast<float>(v.y) + 0.5F) * sp,
      kOrigin.z + (static_cast<float>(v.z) + 0.5F) * sp,
  };
}

/**
 * Cell-value reader handed to a Cartesian / block sampler. A named functor rather than a
 * lambda because nvcc forbids defining an extended device lambda inside another one, and the
 * reader has to be constructed in the loading lambda but invoked from the per-cell body.
 */
template <typename Partition, typename Span, typename Idx>
struct ScalarReader {
  Partition part{};

  [[nodiscard]] CARBON_HD auto
  operator()(const Span& span,
             const Idx& idx) const -> float {
    return static_cast<float>(part.read(span, idx, 0));
  }
};

// Each family reduces one int64_t carrying both counters: performed queries in the low bits,
// violations above kQueryScale. Counting the queries too is what keeps the test honest — the
// skip filters below are broad enough that a wrong world frame could otherwise silently push
// every cell into the skipped set and leave a green "0 violations".
constexpr int64_t kQueryScale = 1LL << 32;

/** Encoded verdict for one performed query: one query, plus 0-2 violations. */
[[nodiscard]] CARBON_HD auto
check_query(const carbon::render::FieldSample& value,
            const carbon::render::FieldGradientSample& grad,
            carbon::float32_3d p) -> int64_t {
  int64_t bad = 0;
  if (!value.valid || std::fabs(value.value - analytic(p)) > kValueTol) {
    bad += 1;
  }
  if (!grad.valid || std::fabs(grad.value.x - kA) > kGradTol ||
      std::fabs(grad.value.y - kB) > kGradTol || std::fabs(grad.value.z - kC) > kGradTol) {
    bad += 1;
  }
  return 1 + bad * kQueryScale;
}

// --------------------------------------------------------------------------------------
// Dense
// --------------------------------------------------------------------------------------

[[nodiscard]] auto
check_dense(carbon::System& sys,
            const carbon::Stream& stream) -> int64_t {
  using Span = carbon::Dense::Span;
  using Idx = carbon::Dense::Idx;

  // A real stencil, so halo_z > 0 and the sampler's addressable band covers the halo rows
  // rather than the owned rows alone. Without it the multi-GPU seam is never sampled: the
  // interesting failure is a band that is off by halo_z, and with halo_z == 0 every wrong
  // formula agrees with every right one.
  const carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const carbon::Dense grid(sys, {kNx, kNy, kNz}, stencil, carbon::BoundaryMode::Open, kOrigin,
                           kSpacing);
  auto field = grid.newField<float, 1>(carbon::MemorySide::Device, "f");

  const carbon::Kernel fill = grid.make_kernel([field](carbon::KernelLoader& loader) {
    auto p = loader.write(field);
    return [p] CARBON_HD(Span s, Idx i) -> void {
      p.write(s, i, 0, analytic(node_world(s.global_location(i))));
    };
  });
  fill.launch(stream);
  const carbon::Halo<float, 1> halo(field);
  halo(stream);
  stream.wait();

  carbon::ReductionResult<int64_t> result(sys);
  const carbon::Kernel check = grid.make_reduce_kernel<int64_t>(
      [field](carbon::KernelLoader& loader) {
        const ScalarReader<decltype(loader.read(field)), Span, Idx> reader{loader.read(field)};
        return [reader] CARBON_HD(Span s, Idx i) -> int64_t {
          const carbon::int64_3d g = s.global_location(i);
          const carbon::int64_3d ge = s.global_extent();
          if (g.x < 1 || g.y < 1 || g.x + 2 >= ge.x || g.y + 2 >= ge.y) {
            return 0;
          }
          // Skipped at the global domain edges only, not at slab seams: the halo exchange
          // above filled the seam rows, so a first/last owned cell must still interpolate
          // exactly. Outside the domain there is nothing to fill and the sampler clamps,
          // which a linear field would report as a gradient error.
          if (g.z < 1 || g.z + 2 >= ge.z) {
            return 0;
          }

          const auto sampler = carbon::render::make_sampler(s, reader);
          const carbon::float32_3d b = node_world(g);
          const carbon::float32_3d p{b.x + 0.25F * kSpacing, b.y + 0.25F * kSpacing,
                                     b.z + 0.25F * kSpacing};
          return check_query(sampler.sample(p), sampler.gradient(p), p);
        };
      },
      result, carbon::reduce::Sum<int64_t>{});

  check.launch(stream);
  result.sync(stream);
  stream.wait();

  return result.host_value();
}

// --------------------------------------------------------------------------------------
// DenseDisg
// --------------------------------------------------------------------------------------

[[nodiscard]] auto
check_dense_disg(carbon::System& sys,
                 const carbon::Stream& stream) -> int64_t {
  using Span = carbon::DenseDisg::Span;
  using Idx = carbon::DenseDisg::Idx;

  const carbon::DenseDisg grid(sys, {kNx, kNy, kNz}, 1, carbon::BoundaryMode::Open, kOrigin,
                               kSpacing);
  auto field = grid.newField<float, 1>(carbon::MemorySide::Device, "f");

  const carbon::Kernel fill = grid.make_kernel([field](carbon::KernelLoader& loader) {
    auto p = loader.write(field);
    return [p] CARBON_HD(Span s, Idx i) -> void {
      p.write(s, i, 0, analytic(node_world(s.global_location(i))));
    };
  });
  fill.launch(stream);

  carbon::ReductionResult<int64_t> result(sys);
  const carbon::Kernel check = grid.make_reduce_kernel<int64_t>(
      [field](carbon::KernelLoader& loader) {
        const ScalarReader<decltype(loader.read(field)), Span, Idx> reader{loader.read(field)};
        return [reader] CARBON_HD(Span s, Idx i) -> int64_t {
          const auto sampler = carbon::render::make_sampler(s, reader);

          const carbon::int64_3d g = s.global_location(i);
          const carbon::int64_3d ge = s.global_extent();
          if (g.x < 1 || g.y < 1 || g.x + 2 >= ge.x || g.y + 2 >= ge.y) {
            return 0;
          }
          if (g.z < sampler.addressable_z_lo() + 1 || g.z + 2 >= sampler.addressable_z_hi()) {
            return 0;
          }

          const carbon::float32_3d b = node_world(g);
          const carbon::float32_3d p{b.x + 0.25F * kSpacing, b.y + 0.25F * kSpacing,
                                     b.z + 0.25F * kSpacing};
          return check_query(sampler.sample(p), sampler.gradient(p), p);
        };
      },
      result, carbon::reduce::Sum<int64_t>{});

  check.launch(stream);
  result.sync(stream);
  stream.wait();
  return result.host_value();
}

// --------------------------------------------------------------------------------------
// Block
// --------------------------------------------------------------------------------------

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;

/**
 * Fully active. Sparse-hole fallback is a separate concern from the world-to-cell map, and a
 * dense predicate keeps the eight-corner stencil complete wherever the query filter allows a
 * query at all.
 */
[[nodiscard]] auto
block_active(const carbon::int64_3d&) -> bool {
  return true;
}

[[nodiscard]] auto
check_block(carbon::System& sys,
            const carbon::Stream& stream) -> int64_t {
  using Span = BlockGrid::Span;
  using Idx = BlockGrid::Idx;

  const BlockGrid grid(sys, {kNx, kNy, kNz}, block_active, "sample_block");
  auto field = grid.newField<float, 1>(carbon::MemorySide::Device, "f");

  const carbon::Kernel fill = grid.make_kernel([field](carbon::KernelLoader& loader) {
    auto p = loader.write(field);
    return [p] CARBON_HD(Span s, Idx i) -> void {
      p.write(s, i, 0, analytic(cell_world(s.global_location(i), kSpacing)));
    };
  });
  fill.launch(stream);

  carbon::ReductionResult<int64_t> result(sys);
  const carbon::Kernel check = grid.make_reduce_kernel<int64_t>(
      [&grid, field](carbon::KernelLoader& loader) {
        const ScalarReader<decltype(loader.read(field)), Span, Idx> reader{loader.read(field)};
        const auto locator = grid.locator(loader.relative_gpu_id(), kOrigin, kSpacing);
        return [reader, locator] CARBON_HD(Span s, Idx i) -> int64_t {
          const auto sampler = carbon::render::make_sampler(locator, s, reader);

          const carbon::int64_3d g = s.global_location(i);
          const carbon::int64_3d ext = locator.voxel_extent;
          if (g.x < 1 || g.y < 1 || g.z < 1 || g.x + 2 >= ext.x || g.y + 2 >= ext.y ||
              g.z + 2 >= ext.z) {
            return 0;
          }

          // Stay clear of the slab seam: corners there live in the neighbouring GPU's bricks,
          // which this slab's table does not index (the locator's ownership contract).
          const carbon::float32_3d centre = locator.voxel_centre(g);
          if (centre.z - 2.0F * kSpacing < locator.owned_z_lo ||
              centre.z + 2.0F * kSpacing > locator.owned_z_hi) {
            return 0;
          }

          const carbon::float32_3d p{centre.x + 0.25F * kSpacing, centre.y + 0.25F * kSpacing,
                                     centre.z + 0.25F * kSpacing};
          return check_query(sampler.sample(p), sampler.gradient(p), p);
        };
      },
      result, carbon::reduce::Sum<int64_t>{});

  check.launch(stream);
  result.sync(stream);
  stream.wait();
  return result.host_value();
}

// --------------------------------------------------------------------------------------
// MultiResBlock
// --------------------------------------------------------------------------------------

using MSB = carbon::DefaultMultiResBlock;
using MGrid = carbon::MultiResBlock<MSB>;

constexpr int kDepth = 2;
constexpr int64_t kMFinestZ = 128;  // Coarsest brick z-extent stays >= 8 slabs.

[[nodiscard]] auto
mgrid_active(int,
             const carbon::int64_3d&) -> bool {
  return true;
}

/**
 * Per-level value reader for @c MultiResSampler . A functor rather than a lambda because it
 * has to hold one partition per level, and the sampler calls it with the level the tower
 * resolved to.
 */
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
check_mgrid(carbon::System& sys,
            const carbon::Stream& stream) -> int64_t {
  using Span = MGrid::Span;
  using Idx = MGrid::Idx;

  const MGrid grid(sys, {kNx, kNy, kMFinestZ}, kDepth, mgrid_active, MSB::UserX, "sample_mgrid");
  auto field = grid.newField<float, 1>(carbon::MemorySide::Device, "f");

  for (int l = 0; l < kDepth; ++l) {
    const float sp = kSpacing * static_cast<float>(grid.descriptor().voxel_spacing(l));
    const carbon::Kernel fill = grid.make_kernel(l, [field, l, sp](carbon::KernelLoader& loader) {
      auto p = loader.write(field.level_view(l));
      return [p, sp] CARBON_HD(Span s, Idx i) -> void {
        p.write(s, i, 0, analytic(cell_world(s.global_location(i), sp)));
      };
    });
    fill.launch(stream);
  }

  using PartitionT = decltype(std::declval<carbon::KernelLoader&>().read(
      std::declval<MGrid::Field<float, 1>&>().level_view(0)));

  const int coarse = kDepth - 1;
  carbon::ReductionResult<int64_t> result(sys);
  const carbon::Kernel check = grid.make_reduce_kernel<int64_t>(
      coarse,
      [&grid, field, coarse](carbon::KernelLoader& loader) {
        LevelReader<PartitionT, kDepth> reader{};
        for (int l = 0; l < kDepth; ++l) {
          reader.part[l] = loader.read(field.level_view(l));
        }
        const auto tower = grid.locator(loader.relative_gpu_id(), kOrigin, kSpacing);
        return [tower, reader, coarse] CARBON_HD(Span s, Idx i) -> int64_t {
          const auto sampler = carbon::render::make_sampler(tower, reader);

          const auto& loc = tower.level_locator[coarse];
          const float sp = loc.world_spacing;
          const carbon::int64_3d g = s.global_location(i);
          const carbon::int64_3d ext = loc.voxel_extent;
          if (g.x < 1 || g.y < 1 || g.z < 1 || g.x + 2 >= ext.x || g.y + 2 >= ext.y ||
              g.z + 2 >= ext.z) {
            return 0;
          }
          const carbon::float32_3d centre = loc.voxel_centre(g);
          if (centre.z - 2.0F * sp < tower.owned_z_lo() ||
              centre.z + 2.0F * sp > tower.owned_z_hi()) {
            return 0;
          }

          // Offset by a quarter of a *coarse* cell so the query is off-lattice at every level.
          const carbon::float32_3d p{centre.x + 0.25F * sp, centre.y + 0.25F * sp,
                                     centre.z + 0.25F * sp};
          return check_query(sampler.sample(p), sampler.gradient(p), p);
        };
      },
      result, carbon::reduce::Sum<int64_t>{});

  check.launch(stream);
  result.sync(stream);
  stream.wait();
  return result.host_value();
}

}  // namespace

auto
main() -> int {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_render_sample: skipping (no CUDA devices visible)");
    return EXIT_SUCCESS;
  }

  auto devs = carbon::get_all_devices();
  carbon::System sys(carbon::detail::GpuGroup{std::move(devs)});
  const carbon::Stream stream = sys.stream();

  // Floor on the query count per family. Set well below what any supported GPU count
  // produces (the tightest case is the coarse mGrid level split across 8 slabs), but high
  // enough that a world frame broken badly enough to skip everything cannot pass.
  constexpr int64_t kMinQueries = 512;

  const struct {
    const char* name;
    int64_t (*run)(carbon::System&,
                   const carbon::Stream&);
  } cases[] = {
      {"dense", &check_dense},
      {"dense_disg", &check_dense_disg},
      {"block", &check_block},
      {"mgrid", &check_mgrid},
  };

  for (const auto& c : cases) {
    const int64_t tally = c.run(sys, stream);
    const int64_t queries = tally % kQueryScale;
    const int64_t violations = tally / kQueryScale;
    if (violations != 0) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_render_sample {}: {} violation(s) over {} quer(ies)", c.name, violations, queries);
    }
    if (queries < kMinQueries) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_render_sample {}: only {} quer(ies) ran (expected >= {}); the skip filters "
          "swallowed the domain, so a clean result here would not mean anything",
          c.name, queries, kMinQueries);
    }
    CARBON_LOG_INFO("test_render_sample {}: OK ({} queries)", c.name, queries);
  }

  CARBON_LOG_INFO("test_render_sample: OK");
  return EXIT_SUCCESS;
}
