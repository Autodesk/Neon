// Unit tests for the mGrid brick-index tower (detail/mgrid/locator.h).
//
// The tower stacks one Block locator per refinement level over a shared world frame, so
// find_finest can resolve an arbitrary world point to the finest active cell covering it
// without any traversal structure. The tests pin three properties:
//
//  1. Per-level round-trip. For every active cell of every level, the world centre of that
//     cell resolves back to the same cell through that level's own locator. This is the
//     level-l analogue of test_block_locator, run through the tower's shared frame so the
//     R^l spacing is exercised.
//
//  2. find_finest resolves to the finest level that actually holds the point. The domain is
//     built so refinement is spatially selective (only a slab of the domain is active on
//     level 0), which makes "finest available" differ from "level 0" across the domain.
//
//  3. Owned world-z bands agree across levels. mGrid projects the coarsest z-cut to every
//     level precisely so a parent brick and its descendants stay co-resident; if that ever
//     stopped holding, a per-slab render segment on one level would not line up with the
//     same segment on another. Checked on the host across all GPUs.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/devices.h>
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

using SBlock = carbon::DefaultMultiResBlock;  // 8^3 memory brick / 2^3 user block (octree).
using MGrid = carbon::MultiResBlock<SBlock>;
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;
using Tower = carbon::detail::mgrid::MultiResLocator<SBlock>;

constexpr int kDepth = 2;
constexpr int64_t kFinestX = 32;
constexpr int64_t kFinestY = 32;
// Multiple of MemZ * R^(depth-1) = 8 * 2 = 16, and deep enough that the coarsest level still
// has one brick row per slab on an 8-GPU box (coarsest brick z-extent = kFinestZ / 2 / 8).
constexpr int64_t kFinestZ = 128;

constexpr carbon::float32_3d kWorldOrigin{-1.5F, 0.75F, 2.0F};
constexpr float kBaseSpacing = 0.125F;

/**
 * Spatially selective refinement: the coarse level covers everything, the fine level covers
 * only the lower-x half. So find_finest must answer level 0 there and level 1 elsewhere,
 * which is what makes property (2) a real check rather than a tautology.
 */
[[nodiscard]] auto
active_at(int level,
          const carbon::int64_3d& v) -> bool {
  if (level != 0) {
    return true;
  }
  return v.x < kFinestX / 2;
}

[[nodiscard]] CARBON_HD auto
same_cell(const Idx& a,
          const Idx& b) -> bool {
  return a.block_idx() == b.block_idx() && a.intra().x == b.intra().x &&
         a.intra().y == b.intra().y && a.intra().z == b.intra().z;
}

// Both reductions carry two counters: cells visited in the low bits, violations above
// kVisitScale. Counting the visits matters as much as counting the failures — a reduction
// that silently never ran also reports zero violations.
constexpr int64_t kVisitScale = 1LL << 32;

/**
 * Per-level violations under the tower's shared world frame, packed with the visit count.
 * Zero violations means every active cell of @p level round-trips through its own locator and
 * resolves through @c find_finest to a level no coarser than itself.
 */
[[nodiscard]] auto
count_violations(const MGrid& mgrid,
                 carbon::System& sys,
                 const carbon::Stream& stream,
                 int level) -> int64_t {
  carbon::ReductionResult<int64_t> result(sys);
  const auto kernel = mgrid.template make_reduce_kernel<int64_t>(
      level,
      [&mgrid, level](carbon::KernelLoader& loader) {
        const Tower tower = mgrid.locator(loader.relative_gpu_id(), kWorldOrigin, kBaseSpacing);
        return [tower, level] CARBON_HD(Span span, Idx idx) -> int64_t {
          int64_t bad = 0;
          const auto& loc = tower.level_locator[level];
          const carbon::int64_3d gc = span.global_location(idx);
          const carbon::float32_3d centre = loc.voxel_centre(gc);

          const auto by_voxel = loc.locate(span, gc);
          if (!by_voxel.valid || !same_cell(by_voxel.idx, idx)) {
            bad += 1;
          }

          const auto by_world = loc.locate_world(span, centre);
          if (!by_world.valid || !same_cell(by_world.idx, idx)) {
            bad += 1;
          }

          // This cell exists at @c level , so the finest level holding @c centre is at most
          // @c level . (It can be strictly finer: a coarse cell overlapping a refined region.)
          int hit_level = -1;
          Idx hit_idx{};
          if (!tower.find_finest(centre, hit_level, hit_idx)) {
            bad += 1;
          } else if (hit_level > level) {
            bad += 1;
          }

          // Whatever level find_finest picked must itself round-trip at that level.
          if (hit_level >= 0) {
            const auto recheck =
                tower.level_locator[hit_level].locate_world(tower.level_span[hit_level], centre);
            if (!recheck.valid || !same_cell(recheck.idx, hit_idx)) {
              bad += 1;
            }
          }

          return 1 + bad * kVisitScale;
        };
      },
      result, carbon::reduce::Sum<int64_t>{});

  kernel.launch(stream);
  result.sync(stream);
  return result.host_value();
}

/**
 * find_finest must pick level 0 inside the refined half and level 1 outside it. Counted on
 * the coarse level so every cell visited has a well-defined "is my region refined" answer.
 */
[[nodiscard]] auto
count_level_selection_errors(const MGrid& mgrid,
                             carbon::System& sys,
                             const carbon::Stream& stream) -> int64_t {
  const int coarse = kDepth - 1;
  const int64_t refined_x_end = kFinestX / 2;
  const int64_t spacing = mgrid.descriptor().voxel_spacing(coarse);

  carbon::ReductionResult<int64_t> result(sys);
  const auto kernel = mgrid.template make_reduce_kernel<int64_t>(
      coarse,
      [&mgrid, coarse, refined_x_end, spacing](carbon::KernelLoader& loader) {
        const Tower tower = mgrid.locator(loader.relative_gpu_id(), kWorldOrigin, kBaseSpacing);
        return [tower, coarse, refined_x_end, spacing] CARBON_HD(Span span, Idx idx) -> int64_t {
          const carbon::int64_3d gc = span.global_location(idx);
          const carbon::float32_3d centre = tower.level_locator[coarse].voxel_centre(gc);

          // Finest-level voxel column this coarse cell's centre falls in.
          const int64_t finest_x = gc.x * spacing + spacing / 2;
          const bool expect_refined = finest_x < refined_x_end;

          int hit_level = -1;
          Idx hit_idx{};
          if (!tower.find_finest(centre, hit_level, hit_idx)) {
            return 1 + kVisitScale;
          }
          return 1 + ((hit_level == (expect_refined ? 0 : coarse)) ? 0 : kVisitScale);
        };
      },
      result, carbon::reduce::Sum<int64_t>{});

  kernel.launch(stream);
  result.sync(stream);
  return result.host_value();
}

/** Every level must report the same owned world-z band on each GPU (co-residency). */
void
check_bands_agree(const MGrid& mgrid) {
  for (size_t g = 0; g < mgrid.system().gpu_group().size(); ++g) {
    const Tower tower = mgrid.locator(g, kWorldOrigin, kBaseSpacing);
    const float lo = tower.level_locator[0].owned_z_lo;
    const float hi = tower.level_locator[0].owned_z_hi;
    for (int l = 1; l < kDepth; ++l) {
      const float l_lo = tower.level_locator[l].owned_z_lo;
      const float l_hi = tower.level_locator[l].owned_z_hi;
      if (std::fabs(l_lo - lo) > 1e-4F || std::fabs(l_hi - hi) > 1e-4F) {
        carbon::log::throw_with_log<std::runtime_error>(
            "test_mgrid_locator bands: gpu {} level {} owns [{}, {}) but level 0 owns [{}, {})", g,
            l, l_lo, l_hi, lo, hi);
      }
    }
  }
}

}  // namespace

auto
main() -> int {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_mgrid_locator: skipping (no CUDA devices visible)");
    return EXIT_SUCCESS;
  }

  auto devs = carbon::get_all_devices();
  carbon::System sys(carbon::detail::GpuGroup{std::move(devs)});
  const carbon::Stream stream = sys.stream();

  const carbon::int64_3d finest{kFinestX, kFinestY, kFinestZ};
  const MGrid mgrid(sys, finest, kDepth, active_at, SBlock::UserX, "mgrid_locator");
  CARBON_LOG_INFO("test_mgrid_locator: finest={}x{}x{}, depth={}, gpus={}", kFinestX, kFinestY,
                  kFinestZ, kDepth, sys.gpu_group().size());

  check_bands_agree(mgrid);

  /** Unpack a packed tally, failing both on violations and on an empty visit set. */
  const auto verdict = [](const char* label, int64_t tally, int64_t expected_visits) {
    const int64_t visited = tally % kVisitScale;
    const int64_t bad = tally / kVisitScale;
    if (bad != 0) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_mgrid_locator {}: {} violation(s) over {} cell(s)", label, bad, visited);
    }
    if (visited != expected_visits) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_mgrid_locator {}: visited {} cell(s), expected {}; a reduction that skipped the "
          "domain would report zero violations either way",
          label, visited, expected_visits);
    }
    CARBON_LOG_INFO("test_mgrid_locator {}: OK ({} cells)", label, visited);
  };

  for (int l = 0; l < kDepth; ++l) {
    verdict("round_trip", count_violations(mgrid, sys, stream, l),
            mgrid.grid(l).active_voxel_count());
  }

  verdict("find_finest", count_level_selection_errors(mgrid, sys, stream),
          mgrid.grid(kDepth - 1).active_voxel_count());

  CARBON_LOG_INFO("test_mgrid_locator: OK");
  return EXIT_SUCCESS;
}
