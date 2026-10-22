// Unit tests for the Block brick-index volume (detail/block/locator.h).
//
// The locator answers the inverse of Span::global_location — "which cell contains this
// point" — so the tests are round-trips: every active voxel the kernel visits must be
// recovered from its own global voxel coordinate and from the world position of its
// centre, under both the identity world frame and a shifted / scaled one.
//
// Also pins the miss cases (outside the domain, brick outside this GPU's table) and the
// multi-GPU ownership contract: the owned world-z bands are disjoint across GPUs and
// cover the whole domain, which is what keeps per-slab render segments from double
// counting.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using LocatorT = carbon::detail::block::Locator<SB>;

constexpr int64_t kDomainN = 64;

/** Hollow sphere: leaves whole bricks unretained, so the table has real holes. */
[[nodiscard]] auto
shell_predicate(const carbon::int64_3d& gc) -> bool {
  const double c = static_cast<double>(kDomainN - 1) * 0.5;
  const double dx = static_cast<double>(gc.x) - c;
  const double dy = static_cast<double>(gc.y) - c;
  const double dz = static_cast<double>(gc.z) - c;
  const double r2 = dx * dx + dy * dy + dz * dz;
  const double outer = static_cast<double>(kDomainN) * 0.45;
  const double inner = static_cast<double>(kDomainN) * 0.22;
  return r2 <= outer * outer && r2 >= inner * inner;
}

[[nodiscard]] CARBON_HD auto
same_cell(const Idx& a,
          const Idx& b) -> bool {
  return a.block_idx() == b.block_idx() && a.intra().x == b.intra().x &&
         a.intra().y == b.intra().y && a.intra().z == b.intra().z;
}

// The reduction carries two counters: voxels visited in the low bits, violations above
// kVisitScale. Counting the visits matters as much as counting the failures — a reduction
// that silently never ran also reports zero violations, which would make a green result here
// mean nothing.
constexpr int64_t kVisitScale = 1LL << 32;

/**
 * Per-voxel locator violations under the world frame @p world_origin / @p world_spacing ,
 * packed with the visit count. Zero violations means every active voxel round-trips and
 * every miss case misses.
 */
[[nodiscard]] auto
count_violations(const BlockGrid& block,
                 carbon::System& sys,
                 const carbon::Stream& stream,
                 carbon::float32_3d world_origin,
                 float world_spacing) -> int64_t {
  carbon::ReductionResult<int64_t> result(sys);
  const auto kernel = block.template make_reduce_kernel<int64_t>(
      [&block, world_origin, world_spacing](carbon::KernelLoader& loader) {
        const LocatorT loc = block.locator(loader.relative_gpu_id(), world_origin, world_spacing);
        return [loc] CARBON_HD(Span span, Idx idx) -> int64_t {
          int64_t bad = 0;
          const carbon::int64_3d gc = span.global_location(idx);

          const auto by_voxel = loc.locate(span, gc);
          if (!by_voxel.valid || !same_cell(by_voxel.idx, idx)) {
            bad += 1;
          }

          const auto by_world = loc.locate_world(span, loc.voxel_centre(gc));
          if (!by_world.valid || !same_cell(by_world.idx, idx)) {
            bad += 1;
          }

          if (loc.locate(span, carbon::int64_3d{loc.voxel_extent.x + 4, gc.y, gc.z}).valid) {
            bad += 1;
          }
          if (loc.locate(span, carbon::int64_3d{gc.x, -1, gc.z}).valid) {
            bad += 1;
          }

          const carbon::int64_3d brick = loc.voxel_to_brick(gc);
          if (loc.brick_at(brick.x, brick.y, brick.z) != idx.block_idx()) {
            bad += 1;
          }
          if (loc.brick_at(brick.x - loc.lattice.x - 1, brick.y, brick.z) != LocatorT::k_no_block) {
            bad += 1;
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
 * Owned world-z bands must tile the domain: contiguous, ascending, and spanning the whole
 * z extent once the (possibly ragged) last brick row is included.
 */
void
check_owned_bands(const BlockGrid& block,
                  float world_spacing) {
  const carbon::float32_3d origin{0.0F, 0.0F, 0.0F};
  float prev_hi = 0.0F;
  for (size_t g = 0; g < block.gpu_count(); ++g) {
    const LocatorT loc = block.locator(g, origin, world_spacing);
    if (loc.owned_z_lo != prev_hi) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_block_locator owned_bands: gpu {} starts at {} but the previous band ended at {}",
          g, loc.owned_z_lo, prev_hi);
    }
    if (loc.owned_z_hi <= loc.owned_z_lo) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_block_locator owned_bands: gpu {} has an empty band [{}, {})", g, loc.owned_z_lo,
          loc.owned_z_hi);
    }
    prev_hi = loc.owned_z_hi;
  }

  const float domain_hi = static_cast<float>(kDomainN) * world_spacing;
  if (prev_hi < domain_hi) {
    carbon::log::throw_with_log<std::runtime_error>(
        "test_block_locator owned_bands: the union ends at {} but the domain ends at {}", prev_hi,
        domain_hi);
  }
}

/** A GPU with retained bricks must produce a table at least as large as its brick count. */
void
check_host_table(const BlockGrid& block) {
  for (size_t g = 0; g < block.gpu_count(); ++g) {
    const LocatorT loc = block.locator(g);
    const auto& origins = block.host_origins(g);
    if (origins.empty()) {
      if (loc.valid()) {
        carbon::log::throw_with_log<std::runtime_error>(
            "test_block_locator host_table: gpu {} has no brick but a valid locator", g);
      }
      continue;
    }
    if (!loc.valid()) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_block_locator host_table: gpu {} retained {} brick(s) but the locator is invalid",
          g, origins.size());
    }
    const int64_t entries = loc.lattice.x * loc.lattice.y * loc.lattice.z;
    if (entries < static_cast<int64_t>(origins.size())) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_block_locator host_table: gpu {} table holds {} entries for {} brick(s)", g,
          entries, origins.size());
    }
  }
}

}  // namespace

auto
main() -> int {
  carbon::init();
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_block_locator: skipping (no CUDA devices visible)");
    return EXIT_SUCCESS;
  }

  auto devs = carbon::get_all_devices();
  carbon::System sys(carbon::detail::GpuGroup{std::move(devs)});
  const carbon::Stream stream = sys.stream();

  const carbon::int64_3d global{kDomainN, kDomainN, kDomainN};
  const auto predicate = [](const carbon::int64_3d& gc) -> bool {
    return shell_predicate(gc);
  };
  const BlockGrid block(sys, global, predicate, "block_locator");
  CARBON_LOG_INFO("test_block_locator: domain={}^3, gpus={}, retained {} bricks ({} active voxels)",
                  kDomainN, sys.gpu_group().size(), block.block_count(),
                  block.active_voxel_count());

  check_host_table(block);
  check_owned_bands(block, 1.0F);
  check_owned_bands(block, 0.25F);

  const auto run_frame = [&](const char* label, carbon::float32_3d origin, float spacing) {
    const int64_t tally = count_violations(block, sys, stream, origin, spacing);
    const int64_t visited = tally % kVisitScale;
    const int64_t bad = tally / kVisitScale;
    if (bad != 0) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_block_locator {}: {} violation(s) over {} voxel(s)", label, bad, visited);
    }
    if (visited != block.active_voxel_count()) {
      carbon::log::throw_with_log<std::runtime_error>(
          "test_block_locator {}: visited {} voxel(s) but the grid has {} active; a reduction "
          "that skipped the domain would report zero violations either way",
          label, visited, block.active_voxel_count());
    }
    CARBON_LOG_INFO("test_block_locator {}: OK ({} voxels)", label, visited);
  };

  run_frame("identity_frame", {0.0F, 0.0F, 0.0F}, 1.0F);
  run_frame("shifted_frame", {-10.0F, 5.0F, 3.0F}, 0.25F);

  CARBON_LOG_INFO("test_block_locator: OK");
  return EXIT_SUCCESS;
}
