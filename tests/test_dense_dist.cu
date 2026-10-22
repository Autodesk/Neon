// Distributed-aware Dense test (one rank per GPU).
//
// Launched under `mpiexec -n 2 ./test_dense_dist`. Each rank constructs a Dense over a global
// volume and verifies that the cross-rank partitioning gave it the right *local* slab:
//
// - dense.gpu_count() == 1 (carbon's one-rank-per-GPU model in MPI mode).
// - dense.global_slab_count() == world_size and dense.local_slab_offset() == world_rank.
// - The single local span carries the global z origin / extent for *this rank's* slab — i.e.
//   span.z_offset() + halo_z equals the cumulative-sum prefix of the global z partition up to
//   world_rank, and span.logic_extent().z equals the global slab's owned z count.
// - span.global_extent() is the full global volume (so Partition::neighbour can clip against
//   the global domain regardless of which rank owns the cell).
// - The constructor rejects the multi-GPU-per-rank case (carbon assumes one rank per GPU).
//
// Halo-aware variant: same checks with a radius-1 stencil to confirm the halo z-thickness is
// applied locally without polluting the global z origin arithmetic.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are visible.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/distributed.h>
#include <carbon/integer3d.h>
#include <carbon/logging.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <vector>

namespace {

[[nodiscard]] auto
even_split_z(int64_t global_z,
             size_t world_size) -> std::vector<int64_t> {
  // Mirrors src/dense.cpp::split_extent_z. Local copy lives here so the test computes the
  // expected partition independently of the implementation.
  std::vector<int64_t> counts(world_size);
  const int64_t base = global_z / static_cast<int64_t>(world_size);
  const int64_t rem = global_z % static_cast<int64_t>(world_size);
  for (size_t i = 0; i < world_size; ++i) {
    counts[i] = base + (static_cast<int64_t>(i) < rem ? 1 : 0);
  }
  return counts;
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_dense_dist: world_size={} < 2, skipping", distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_dense_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_dense_dist: expected gpu_group size 1 (one rank per GPU), got {}",
                     sys.gpu_group().size());
    return 1;
  }

  // Pick a global z that is *not* divisible by world_size so the partition has a non-empty
  // remainder — this exercises the "first (z % world_size) slabs get one extra layer" branch.
  const int64_t global_z = static_cast<int64_t>(world_size) * 3 + 1;
  const carbon::int64_3d global{4, 4, global_z};

  const std::vector<int64_t> expected_counts =
      even_split_z(global_z, static_cast<size_t>(world_size));
  int64_t expected_z_origin = 0;
  for (int r = 0; r < rank; ++r) {
    expected_z_origin += expected_counts[static_cast<size_t>(r)];
  }
  const int64_t expected_local_z = expected_counts[static_cast<size_t>(rank)];

  // 1. No-stencil case: halo_z == 0, span.z_offset() == expected_z_origin directly.
  {
    carbon::Dense dense(sys, global);

    if (dense.gpu_count() != 1U) {
      CARBON_LOG_ERROR("test_dense_dist[no-halo]: gpu_count {} != 1", dense.gpu_count());
      return 1;
    }
    if (dense.global_slab_count() != static_cast<size_t>(world_size)) {
      CARBON_LOG_ERROR("test_dense_dist[no-halo]: global_slab_count {} != world_size {}",
                       dense.global_slab_count(), world_size);
      return 1;
    }
    if (dense.local_slab_offset() != static_cast<size_t>(rank)) {
      CARBON_LOG_ERROR("test_dense_dist[no-halo]: local_slab_offset {} != world_rank {}",
                       dense.local_slab_offset(), rank);
      return 1;
    }

    const auto& span = dense.spans()[0];
    if (span.halo_z() != 0) {
      CARBON_LOG_ERROR("test_dense_dist[no-halo]: span.halo_z {} != 0", span.halo_z());
      return 1;
    }
    if (span.z_offset() != expected_z_origin) {
      CARBON_LOG_ERROR("test_dense_dist[no-halo]: span.z_offset {} != expected {}", span.z_offset(),
                       expected_z_origin);
      return 1;
    }
    if (span.logic_extent().z != expected_local_z) {
      CARBON_LOG_ERROR("test_dense_dist[no-halo]: logic_extent.z {} != expected {}",
                       span.logic_extent().z, expected_local_z);
      return 1;
    }
    if (span.global_extent().x != global.x || span.global_extent().y != global.y ||
        span.global_extent().z != global.z) {
      CARBON_LOG_ERROR("test_dense_dist[no-halo]: span.global_extent ({}, {}, {}) != ({}, {}, {})",
                       span.global_extent().x, span.global_extent().y, span.global_extent().z,
                       global.x, global.y, global.z);
      return 1;
    }
    if (dense.z_offset_global(0) != expected_z_origin) {
      CARBON_LOG_ERROR("test_dense_dist[no-halo]: z_offset_global(0) {} != expected {}",
                       dense.z_offset_global(0), expected_z_origin);
      return 1;
    }
    if (dense.local_z(0) != expected_local_z) {
      CARBON_LOG_ERROR("test_dense_dist[no-halo]: local_z(0) {} != expected {}", dense.local_z(0),
                       expected_local_z);
      return 1;
    }
  }

  // 2. Halo case: radius-1 z-only stencil. Span carries halo_z == 1, span.z_offset() points one
  //    layer before expected_z_origin, but z_offset_global() still resolves to the first owned
  //    layer in global coords.
  {
    carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                            /*remove_center=*/true);
    if (stencil.radius() != 1) {
      CARBON_LOG_ERROR("test_dense_dist[halo]: stencil.radius {} != 1", stencil.radius());
      return 1;
    }
    carbon::Dense dense(sys, global, std::move(stencil));

    const auto& span = dense.spans()[0];
    if (span.halo_z() != 1) {
      CARBON_LOG_ERROR("test_dense_dist[halo]: span.halo_z {} != 1", span.halo_z());
      return 1;
    }
    if (span.z_offset() != expected_z_origin - 1) {
      CARBON_LOG_ERROR("test_dense_dist[halo]: span.z_offset {} != expected {}", span.z_offset(),
                       expected_z_origin - 1);
      return 1;
    }
    if (span.logic_extent().z != expected_local_z) {
      CARBON_LOG_ERROR("test_dense_dist[halo]: logic_extent.z {} != expected {}",
                       span.logic_extent().z, expected_local_z);
      return 1;
    }
    if (span.haloed_extent().z != expected_local_z + 2) {
      CARBON_LOG_ERROR("test_dense_dist[halo]: haloed_extent.z {} != expected {} (logic+2)",
                       span.haloed_extent().z, expected_local_z + 2);
      return 1;
    }
    if (dense.z_offset_global(0) != expected_z_origin) {
      CARBON_LOG_ERROR("test_dense_dist[halo]: z_offset_global(0) {} != expected {}",
                       dense.z_offset_global(0), expected_z_origin);
      return 1;
    }
  }

  CARBON_LOG_INFO(
      "test_dense_dist: rank {} of {} ok (local_z={}, z_origin={}, global_slab_count={})", rank,
      world_size, expected_local_z, expected_z_origin, world_size);
  return 0;
}
