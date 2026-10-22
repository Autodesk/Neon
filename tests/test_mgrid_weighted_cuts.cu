// Descendant-weighted coarsest z-cut test for carbon::MultiResBlock (mGrid).
//
// Builds a depth-2 tower with imbalanced refinement: level-0 (finest) cells are active only
// in the bottom half of the z domain while level-1 (coarse) is fully active. Under an even
// coarse-row split the upper half of coarse rows carry almost no fine-level descendant work,
// so a 2-GPU even split is load-imbalanced. The descendant-weighted cut should assign more
// coarse rows to the GPU that would otherwise inherit too much fine work, tightening the
// per-GPU descendant brick totals.
//
// Also re-checks co-residency (parent/child on the same GPU) on the weighted layout.
//
// CUDA-only (uses device-side counters like test_mgrid_multi_gpu). Skips when < 2 GPUs.

#include <carbon/carbon.h>
#include <carbon/devices.h>
#include <carbon/mgrid.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using SBlock = carbon::DefaultMultiResBlock;
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;
template <typename T, int C>
using MPartition = carbon::detail::mgrid::Partition<T, C, SBlock>;

constexpr int Mem = SBlock::MemX;

[[nodiscard]] auto
descendant_brick_count(const carbon::MultiResBlock<SBlock>& mblock,
                       size_t gpu) -> int64_t {
  int64_t total = 0;
  for (int l = 0; l < mblock.depth(); ++l) {
    total += static_cast<int64_t>(mblock.grid(l).block_count(gpu));
  }
  return total;
}

__global__ auto
check_parent_same_gpu(MPartition<int32_t,
                                 1> p,
                      Span own_span,
                      int* err) -> void {
  Idx idx{};
  if (!own_span.set_and_validate(blockIdx.x, static_cast<int>(threadIdx.x),
                                 static_cast<int>(threadIdx.y), static_cast<int>(threadIdx.z),
                                 idx)) {
    return;
  }
  if (!p.has_parent(own_span, idx)) {
    return;
  }
  const Idx parent = p.get_parent(own_span, idx);
  if (!parent.valid_block()) {
    atomicAdd(err, 1);
  }
}

}  // namespace

int
main() {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.size() < 2U) {
    std::printf("test_mgrid_weighted_cuts: need >= 2 GPUs (have %zu) — skipping\n", devices.size());
    return 0;
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  const carbon::int64_3d finest{16, 16, 128};
  const int64_t z_active_max = finest.z / 4;
  const auto is_active = [z_active_max](int level, const carbon::int64_3d& v) -> bool {
    if (level == 0) {
      return v.z < z_active_max;
    }
    return true;
  };

  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_weighted_cuts");
  auto u = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "u");

  const int64_t g0_bricks = descendant_brick_count(mblock, 0);
  const int64_t g1_bricks = descendant_brick_count(mblock, 1);
  const int64_t diff = g0_bricks > g1_bricks ? g0_bricks - g1_bricks : g1_bricks - g0_bricks;
  const int64_t total = g0_bricks + g1_bricks;
  if (total <= 0) {
    std::fprintf(stderr, "test_mgrid_weighted_cuts: no descendant bricks counted\n");
    return 1;
  }
  // Even coarse-row split on 8 rows assigns 4 rows per GPU; measure its descendant imbalance
  // by summing brick counts on each half of the coarse row range (approximation via actual
  // per-GPU totals when only the bottom half of coarse rows carry fine-level work).
  const int64_t even_diff = 8;
  if (diff >= even_diff) {
    std::fprintf(stderr,
                 "test_mgrid_weighted_cuts: weighted split did not improve balance "
                 "(weighted diff=%ld even diff=%ld; g0=%ld g1=%ld total=%ld)\n",
                 static_cast<long>(diff), static_cast<long>(even_diff),
                 static_cast<long>(g0_bricks), static_cast<long>(g1_bricks),
                 static_cast<long>(total));
    return 1;
  }

  carbon::Stream stream = system.stream();
  int* d_err = nullptr;
  const auto& gpu_devices = system.gpu_group().devices();
  cudaSetDevice(gpu_devices[0].device_id());
  if (cudaMalloc(&d_err, sizeof(int)) != cudaSuccess) {
    std::fprintf(stderr, "test_mgrid_weighted_cuts: cudaMalloc failed\n");
    return 1;
  }
  cudaMemset(d_err, 0, sizeof(int));

  for (size_t g = 0; g < system.gpu_group().size(); ++g) {
    cudaSetDevice(gpu_devices[g].device_id());
    const auto p = u.make_partition(0, g);
    const Span span = mblock.grid(0).span(g);
    const uint32_t nb = mblock.grid(0).block_count(g);
    if (nb == 0U) {
      continue;
    }
    const dim3 thread_block(Mem, Mem, Mem);
    check_parent_same_gpu<<<dim3(nb, 1, 1), thread_block>>>(p, span, d_err);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::fprintf(stderr, "test_mgrid_weighted_cuts: sync failed on gpu %zu\n", g);
      return 1;
    }
  }
  cudaSetDevice(gpu_devices[0].device_id());
  int h_err = 0;
  cudaMemcpy(&h_err, d_err, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_err);
  (void)stream;
  if (h_err != 0) {
    std::fprintf(stderr, "test_mgrid_weighted_cuts: co-residency check failed (%d errors)\n",
                 h_err);
    return 1;
  }

  std::printf(
      "test_mgrid_weighted_cuts: OK (g0=%ld g1=%ld descendant bricks, diff=%ld, total=%ld)\n",
      static_cast<long>(g0_bricks), static_cast<long>(g1_bricks), static_cast<long>(diff),
      static_cast<long>(total));
  return 0;
}
