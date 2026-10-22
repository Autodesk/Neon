// Cross-rank (MPI / one-rank-per-GPU) correctness test for carbon::MultiResBlock (mGrid)
// co-resident cross-level access. Launched under `mpiexec -n 2 ./test_mgrid_dist`.
//
// The co-resident partitioning rule (docs/design_mgrid.md §4) projects the coarsest level's
// z-cut down to every finer level, so a parent brick and all of its descendants live on the
// SAME rank. Cross-level get_parent / get_child / parent / child are therefore pure local
// dereferences with NO cross-rank traffic — this test proves exactly that: each rank fills
// its own slab's levels and verifies every owned cross-level read resolves locally and
// matches encode(level, global_voxel), with zero halo / DistTransfer / NCCL on the path.
//
// Domain: fully-active 16×16×32 finest, depth-2 octree. With world_size==2 rank 0 owns the
// lower coarsest brick row (and its finer descendants) and rank 1 the upper — so each rank
// owns one L1 brick and eight L0 bricks and verifies 4096 child + 4096 parent reads.
//
// Skipped (returns 0) when world_size < 2 or no CUDA device is visible. CUDA-only (raw
// <<<>>> kernels): PR1/PR2 defer KernelLoader integration for mGrid.

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/distributed.h>
#include <carbon/mgrid.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <memory>

namespace {

using SBlock = carbon::DefaultMultiResBlock;  // 8³ memory brick / 2³ user block (octree).
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;
template <typename T, int C>
using MPartition = carbon::detail::mgrid::Partition<T, C, SBlock>;

constexpr int Mem = SBlock::MemX;

[[nodiscard]] CARBON_HD constexpr auto
encode(int level,
       int x,
       int y,
       int z) noexcept -> int32_t {
  return level * 1000000 + (x + 100 * y + 10000 * z);
}

__global__ auto
fill_level_kernel(int32_t* mem,
                  Span span,
                  int level) -> void {
  Idx idx{};
  if (!span.set_and_validate(blockIdx.x, static_cast<int>(threadIdx.x),
                             static_cast<int>(threadIdx.y), static_cast<int>(threadIdx.z), idx)) {
    return;
  }
  const auto g = span.global_location(idx);
  mem[span.linear_offset(idx, 1, 0)] =
      encode(level, static_cast<int>(g.x), static_cast<int>(g.y), static_cast<int>(g.z));
}

__global__ auto
check_child_kernel(MPartition<int32_t,
                              1> p,
                   Span own_span,
                   Span child_span,
                   int ref_factor,
                   int* err,
                   int* ok) -> void {
  Idx idx{};
  if (!own_span.set_and_validate(blockIdx.x, static_cast<int>(threadIdx.x),
                                 static_cast<int>(threadIdx.y), static_cast<int>(threadIdx.z),
                                 idx)) {
    return;
  }
  if (!p.has_children(own_span, idx)) {
    atomicAdd(err, 1);
    return;
  }
  for (int8_t oz = 0; oz < ref_factor; ++oz) {
    for (int8_t oy = 0; oy < ref_factor; ++oy) {
      for (int8_t ox = 0; ox < ref_factor; ++ox) {
        const carbon::int8_3d off{ox, oy, oz};
        const Idx c = p.get_child(own_span, idx, off);
        const auto cv = p.child(own_span, idx, off, 0);
        if (!c.valid_block() || !cv.valid) {
          atomicAdd(err, 1);
          continue;
        }
        const auto g = child_span.global_location(c);
        const int32_t expected =
            encode(0, static_cast<int>(g.x), static_cast<int>(g.y), static_cast<int>(g.z));
        if (cv.value != expected) {
          atomicAdd(err, 1);
        } else {
          atomicAdd(ok, 1);
        }
      }
    }
  }
}

__global__ auto
check_parent_kernel(MPartition<int32_t,
                               1> p,
                    Span own_span,
                    Span parent_span,
                    int* err,
                    int* ok) -> void {
  Idx idx{};
  if (!own_span.set_and_validate(blockIdx.x, static_cast<int>(threadIdx.x),
                                 static_cast<int>(threadIdx.y), static_cast<int>(threadIdx.z),
                                 idx)) {
    return;
  }
  if (!p.has_parent(own_span, idx)) {
    atomicAdd(err, 1);
    return;
  }
  const Idx parent = p.get_parent(own_span, idx);
  const auto pv = p.parent(own_span, idx, 0);
  if (!parent.valid_block() || !pv.valid) {
    atomicAdd(err, 1);
    return;
  }
  const auto g = parent_span.global_location(parent);
  const int32_t expected =
      encode(1, static_cast<int>(g.x), static_cast<int>(g.y), static_cast<int>(g.z));
  if (pv.value != expected) {
    atomicAdd(err, 1);
  } else {
    atomicAdd(ok, 1);
  }
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_mgrid_dist: world_size={} < 2, skipping", distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_mgrid_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System system(distributed);
  if (system.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_mgrid_dist: expected gpu_group size 1 (one rank per GPU), got {}",
                     system.gpu_group().size());
    return 1;
  }

  // finest.z (32) is a multiple of MemZ*R^(depth-1) = 8*2 = 16 (z-alignment precondition).
  const carbon::int64_3d finest{16, 16, 32};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_dist");
  const int ref_factor = mblock.ref_factor();

  // This rank owns one coarsest (L1) brick row's worth of the tower: 1 L1 brick + its 8 L0
  // descendant bricks, all co-resident here. (Local slab == gpu 0 under one-rank-per-GPU.)
  const uint32_t nb0 = mblock.grid(0).block_count(0);
  const uint32_t nb1 = mblock.grid(1).block_count(0);
  if (nb0 != 8U || nb1 != 1U) {
    std::fprintf(stderr, "test_mgrid_dist: rank %d unexpected local block counts L0=%u L1=%u\n",
                 rank, nb0, nb1);
    return 1;
  }

  auto field = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "f");
  cudaSetDevice(system.gpu_group().devices()[0].device_id());
  const dim3 thread_block(Mem, Mem, Mem);

  for (int l = 0; l < mblock.depth(); ++l) {
    const uint32_t nb = mblock.grid(l).block_count(0);
    if (nb == 0U) {
      continue;
    }
    int32_t* mem = field.level_field(l).partitions()[0].ptr();
    fill_level_kernel<<<dim3(nb, 1, 1), thread_block>>>(mem, mblock.grid(l).span(0), l);
  }
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
    std::fprintf(stderr, "test_mgrid_dist: rank %d fill sync -> %s\n", rank, cudaGetErrorString(e));
    return 1;
  }

  int* d_counters = nullptr;
  if (cudaMalloc(&d_counters, sizeof(int) * 2U) != cudaSuccess) {
    std::fprintf(stderr, "test_mgrid_dist: rank %d cudaMalloc counters failed\n", rank);
    return 1;
  }
  cudaMemset(d_counters, 0, sizeof(int) * 2U);
  int* err = d_counters;
  int* ok = d_counters + 1;

  check_child_kernel<<<dim3(nb1, 1, 1), thread_block>>>(
      field.make_partition(1, 0), mblock.grid(1).span(0), mblock.grid(0).span(0), ref_factor, err,
      ok);
  check_parent_kernel<<<dim3(nb0, 1, 1), thread_block>>>(
      field.make_partition(0, 0), mblock.grid(0).span(0), mblock.grid(1).span(0), err, ok);
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
    std::fprintf(stderr, "test_mgrid_dist: rank %d check sync -> %s\n", rank,
                 cudaGetErrorString(e));
    cudaFree(d_counters);
    return 1;
  }

  int host_counters[2] = {0, 0};
  cudaMemcpy(host_counters, d_counters, sizeof(int) * 2U, cudaMemcpyDeviceToHost);
  cudaFree(d_counters);

  // L1 children: 1 brick × 512 voxels × 8 octants = 4096. L0 parents: 8 × 512 = 4096.
  const int expected_ok = 4096 + 4096;
  if (host_counters[0] != 0 || host_counters[1] != expected_ok) {
    std::fprintf(stderr, "test_mgrid_dist: rank %d err=%d ok=%d (expected ok=%d)\n", rank,
                 host_counters[0], host_counters[1], expected_ok);
    return 1;
  }
  CARBON_LOG_INFO("test_mgrid_dist: rank {} of {} ok (local cross-level reads verified: {})", rank,
                  world_size, host_counters[1]);
  return 0;
}
