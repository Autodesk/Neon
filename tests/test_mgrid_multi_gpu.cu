// Device multi-GPU correctness test for carbon::MultiResBlock (mGrid) co-resident cross-level
// access. Builds a 2-level octree over a fully-active 16×16×32 finest domain across every
// visible GPU and checks that the co-resident partitioning rule (docs/design_mgrid.md §4)
// keeps a parent brick and all of its children on the SAME GPU — so cross-level reads are
// pure local dereferences (no peer copy / Transfer / NCCL).
//
// Per GPU g it (1) fills every level's storage with value = encode(level, global_voxel) via a
// raw kernel over that GPU's owned bricks, then (2) reads each owned level-1 voxel's eight
// octree children and each owned level-0 voxel's parent through MultiResField::make_partition
// (level, g) — verifying every cross-level value matches its encode() and that the resolved
// child/parent Idx is valid on the same GPU. The summed ok-count across GPUs must equal the
// total cross-level reads of the whole tower.
//
// CUDA-only (raw <<<>>> kernels + per-device cudaMalloc/cudaMemcpy for the counters), same
// rationale as test_mgrid_cross_level / test_block_cuda: PR1/PR2 defer KernelLoader integration
// for mGrid. With a single visible GPU the partition collapses to one slab (still correct, less
// coverage); the co-resident multi-slab path is exercised whenever >= 2 GPUs are present.

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/devices.h>
#include <carbon/mgrid.h>

#include <cuda_runtime.h>

#include "test_domain.h"

#include <cstdint>
#include <cstdio>
#include <vector>

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

[[nodiscard]] auto
sync_or_fail(const char* what) -> bool {
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
    std::fprintf(stderr, "test_mgrid_multi_gpu: %s sync -> %s\n", what, cudaGetErrorString(e));
    return false;
  }
  return true;
}

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("test_mgrid_multi_gpu: no CUDA devices visible — skipping\n");
    return 0;
  }
  const size_t n_gpus = devices.size();
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  // The coarsest level's brick spans MemZ*R^(depth-1) = 8*2 = 16 finest cells. finest.z is a
  // multiple of that, satisfying the multi-slab z-alignment precondition so the coarsest z-cut
  // projects exactly to the finer level, and is grown to keep one coarsest brick row per GPU.
  constexpr int64_t kCoarsestBrickZ = Mem * 2;
  const carbon::int64_3d finest{16, 16, carbon::test::block_slab_z(32, n_gpus, kCoarsestBrickZ)};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_multi_gpu");
  const int ref_factor = mblock.ref_factor();

  // Fully-active tower, so every candidate brick is retained on both levels and the totals are
  // just the brick counts of each level's extent.
  const auto brick_count = [](const carbon::int64_3d& extent) -> uint32_t {
    const auto rows = [](int64_t cells) -> int64_t {
      return (cells + Mem - 1) / Mem;
    };
    return static_cast<uint32_t>(rows(extent.x) * rows(extent.y) * rows(extent.z));
  };
  const uint32_t want_l0 = brick_count(finest);
  const uint32_t want_l1 =
      brick_count({finest.x / ref_factor, finest.y / ref_factor, finest.z / ref_factor});
  uint32_t total_l0 = 0;
  uint32_t total_l1 = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    total_l0 += mblock.grid(0).block_count(g);
    total_l1 += mblock.grid(1).block_count(g);
  }
  if (total_l0 != want_l0 || total_l1 != want_l1) {
    std::fprintf(stderr,
                 "test_mgrid_multi_gpu: unexpected total block counts L0=%u (want %u) L1=%u "
                 "(want %u)\n",
                 total_l0, want_l0, total_l1, want_l1);
    return 1;
  }

  auto field = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "f");
  const dim3 thread_block(Mem, Mem, Mem);
  const auto& gpu_devices = system.gpu_group().devices();

  // Phase 1: fill each level's per-GPU storage.
  for (size_t g = 0; g < n_gpus; ++g) {
    cudaSetDevice(gpu_devices[g].device_id());
    for (int l = 0; l < mblock.depth(); ++l) {
      const uint32_t nb = mblock.grid(l).block_count(g);
      if (nb == 0U) {
        continue;
      }
      int32_t* mem = field.level_field(l).partitions()[g].ptr();
      fill_level_kernel<<<dim3(nb, 1, 1), thread_block>>>(mem, mblock.grid(l).span(g), l);
    }
  }
  for (size_t g = 0; g < n_gpus; ++g) {
    cudaSetDevice(gpu_devices[g].device_id());
    if (!sync_or_fail("fill")) {
      return 1;
    }
  }

  // Phase 2: per-GPU cross-level checks. Counters live on each device (CUDA-only test).
  int total_err = 0;
  int total_ok = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    cudaSetDevice(gpu_devices[g].device_id());
    int* d_counters = nullptr;
    if (cudaMalloc(&d_counters, sizeof(int) * 2U) != cudaSuccess) {
      std::fprintf(stderr, "test_mgrid_multi_gpu: cudaMalloc counters failed on gpu %zu\n", g);
      return 1;
    }
    cudaMemset(d_counters, 0, sizeof(int) * 2U);
    int* err = d_counters;
    int* ok = d_counters + 1;

    const uint32_t nb1 = mblock.grid(1).block_count(g);
    if (nb1 != 0U) {
      check_child_kernel<<<dim3(nb1, 1, 1), thread_block>>>(
          field.make_partition(1, g), mblock.grid(1).span(g), mblock.grid(0).span(g), ref_factor,
          err, ok);
    }
    const uint32_t nb0 = mblock.grid(0).block_count(g);
    if (nb0 != 0U) {
      check_parent_kernel<<<dim3(nb0, 1, 1), thread_block>>>(
          field.make_partition(0, g), mblock.grid(0).span(g), mblock.grid(1).span(g), err, ok);
    }
    if (!sync_or_fail("check")) {
      cudaFree(d_counters);
      return 1;
    }
    int host_counters[2] = {0, 0};
    cudaMemcpy(host_counters, d_counters, sizeof(int) * 2U, cudaMemcpyDeviceToHost);
    cudaFree(d_counters);
    total_err += host_counters[0];
    total_ok += host_counters[1];
  }

  // Each L0 voxel checks its one parent and each L1 voxel its R³ children, so both directions
  // total the finest voxel count (the tower is fully active, so every brick is retained).
  const int expected_ok = static_cast<int>(2 * finest.x * finest.y * finest.z);
  if (total_err != 0 || total_ok != expected_ok) {
    std::fprintf(stderr, "test_mgrid_multi_gpu: err=%d ok=%d (expected ok=%d) across %zu GPU(s)\n",
                 total_err, total_ok, expected_ok, n_gpus);
    return 1;
  }
  std::printf("test_mgrid_multi_gpu: OK (%zu GPU(s), cross-level reads checked: %d)\n", n_gpus,
              total_ok);
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
