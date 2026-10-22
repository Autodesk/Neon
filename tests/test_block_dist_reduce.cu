// Cross-rank Sum reduction test for carbon::Block (one rank per GPU).
//
// Launched under `mpiexec -n 2 ./test_block_dist_reduce`. Each rank owns a Z-partitioned slab
// of a global block-sparse grid (every voxel active, so the brick-active map is regular and
// every rank carries the same brick layout shape on its slabs), fills the owned voxels with a
// deterministic per-voxel value v(gx,gy,gz), and verifies that the three-stage Block reduce
// pipeline
//
//   stage 1 (per local GPU)         block_grid_reduce_kernel + cub::DeviceReduce
//   stage 2 (intra-process combine) — single-GPU per rank under MPI, so a no-op fast path
//   stage 3 (cross-rank, NCCL/RCCL) ncclAllReduce in-place via dist_allreduce_in_place
//
// gives the same value on every rank as a host-side reference over the entire global volume.
//
// Cases covered:
//   1. Standard view across ranks  -> equals global host reference.
//   2. Internal + Boundary view across ranks -> sum equals global host reference, and the
//      ReductionResult::host_value_combined helper reproduces it independently.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are visible — the
// rank-local pipeline already has dedicated coverage in test_block_reduce under the
// non-MPI configuration.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/distributed.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/reduce_dist.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<int64_t, 1>;

[[nodiscard]] CARBON_HD auto
voxel_value(int64_t gx,
            int64_t gy,
            int64_t gz) -> int64_t {
  // Same deterministic encoder shape as test_block_reduce so the math is easy to
  // reason about and the values are identical across ranks regardless of partition.
  return gx + 7 * gy + 53 * gz + 1;
}

[[nodiscard]] auto
host_reference_sum(int64_t W,
                   int64_t H,
                   int64_t D) -> int64_t {
  int64_t s = 0;
  for (int64_t z = 0; z < D; ++z) {
    for (int64_t y = 0; y < H; ++y) {
      for (int64_t x = 0; x < W; ++x) {
        s += voxel_value(x, y, z);
      }
    }
  }
  return s;
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_block_dist_reduce: world_size={} < 2, skipping",
                    distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_block_dist_reduce: no CUDA devices, skipping");
    return EXIT_SUCCESS;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_block_dist_reduce: expected gpu_group size 1 (one rank per GPU), got {}",
                     sys.gpu_group().size());
    return EXIT_FAILURE;
  }

  // Pick a global z that gives at least one full brick row per rank — Block's
  // global z extent must be >= global_slab_count * MemZ so partitioning works.
  const int64_t global_z = static_cast<int64_t>(world_size) * SB::MemZ * 2;
  const carbon::int64_3d global{static_cast<int64_t>(SB::MemX) * 2,
                                static_cast<int64_t>(SB::MemY) * 2, global_z};

  // Every voxel active so the host reference reduces to a closed-form sum
  // over the entire global volume.
  const auto predicate = [](const carbon::int64_3d&) -> bool {
    return true;
  };

  BlockGrid block(sys, global, predicate, "dist_block_reduce");
  if (block.gpu_count() != 1U) {
    CARBON_LOG_ERROR("test_block_dist_reduce[rank={}]: gpu_count == {} (expected 1)", rank,
                     block.gpu_count());
    return EXIT_FAILURE;
  }

  // Stage owned-brick data on device by initialising on host and async_copy-ing
  // per (single) local GPU. With predicate=true every voxel is active.
  FieldT u = block.template newField<int64_t, 1>(carbon::MemorySide::Device, "u");
  carbon::Stream stream = sys.stream();
  const uint32_t n_owned = block.block_count(0);
  const uint32_t n_storage = block.storage_block_count(0);
  const size_t total_slots = static_cast<size_t>(n_storage) * static_cast<size_t>(SB::VoxelCount);
  if (total_slots > 0U) {
    auto host_buf = carbon::Cpu::buffer<int64_t>(total_slots);
    std::vector<int64_t> stage(total_slots, 0);
    for (uint32_t b = 0; b < n_owned; ++b) {
      const auto& origin = block.host_origins(0)[b];
      const auto& bm = block.host_bitmasks(0)[b];
      for (int iz = 0; iz < SB::MemZ; ++iz) {
        for (int iy = 0; iy < SB::MemY; ++iy) {
          for (int ix = 0; ix < SB::MemX; ++ix) {
            if (!bm.is_active(ix, iy, iz)) {
              continue;
            }
            const int64_t gx = static_cast<int64_t>(origin.x) + ix;
            const int64_t gy = static_cast<int64_t>(origin.y) + iy;
            const int64_t gz = static_cast<int64_t>(origin.z) + iz;
            const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                               static_cast<size_t>(ix) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(iy) +
                               static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                   static_cast<size_t>(iz);
            stage[lin] = voxel_value(gx, gy, gz);
          }
        }
      }
    }
    std::memcpy(host_buf.data(), stage.data(), total_slots * sizeof(int64_t));
    sys.async_copy(host_buf, u.gpu_buffers()[0], total_slots * sizeof(int64_t), stream);
  }
  stream.wait();

  const int64_t want = host_reference_sum(global.x, global.y, global.z);

  auto make_sum_kernel = [&](carbon::ReductionResult<int64_t>& dst) {
    return block.template make_reduce_kernel<int64_t>(
        [&u](carbon::KernelLoader& loader) {
          const auto in = loader.read(u);
          return [in] CARBON_HD(Span s, Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        dst, carbon::reduce::Sum<int64_t>{});
  };

  // ---- Case 1: Standard view across ranks ----
  {
    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = make_sum_kernel(result);
    try {
      reducer.launch(stream, carbon::DataView::Standard);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_block_dist_reduce[rank={}]: Standard launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);
    stream.wait();

    const int64_t got = result.host_value(carbon::DataView::Standard);
    if (got != want) {
      CARBON_LOG_ERROR(
          "test_block_dist_reduce[rank={}]: Standard mismatch: got {}, expected {} (host ref "
          "over global {} x {} x {})",
          rank, got, want, global.x, global.y, global.z);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_block_dist_reduce[rank={}]: Standard sum = {} matches global ref over {} ranks", rank,
        got, world_size);
  }

  // ---- Case 2: Internal + Boundary cross-rank recombine ----
  {
    carbon::ReductionResult<int64_t> split(sys);
    auto reducer = make_sum_kernel(split);
    try {
      reducer.launch(stream, carbon::DataView::Internal);
      reducer.launch(stream, carbon::DataView::Boundary);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_block_dist_reduce[rank={}]: Internal/Boundary launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    split.sync(stream);
    stream.wait();

    const int64_t internal = split.host_value(carbon::DataView::Internal);
    const int64_t boundary = split.host_value(carbon::DataView::Boundary);
    if (internal + boundary != want) {
      CARBON_LOG_ERROR(
          "test_block_dist_reduce[rank={}]: Internal({}) + Boundary({}) = {} != global ref {}",
          rank, internal, boundary, internal + boundary, want);
      return EXIT_FAILURE;
    }

    const int64_t combined = split.host_value_combined(carbon::reduce::Sum<int64_t>{});
    if (combined != want) {
      CARBON_LOG_ERROR("test_block_dist_reduce[rank={}]: host_value_combined({}) != global ref {}",
                       rank, combined, want);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_block_dist_reduce[rank={}]: split Internal({}) + Boundary({}) = combined {} "
        "matches global ref {}",
        rank, internal, boundary, combined, want);
  }

  CARBON_LOG_INFO("test_block_dist_reduce[rank={}]: PASS", rank);
  return EXIT_SUCCESS;
}
