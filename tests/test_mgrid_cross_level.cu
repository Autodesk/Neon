// Device single-GPU correctness test for the carbon::MultiResBlock (mGrid) cross-level data
// plane. Builds a 2-level octree over a fully-active 16³ finest domain, allocates a
// MultiResField<int,1>, and:
//
//   1. fills every level's storage with value = encode(level, global_voxel) via a raw kernel
//      (PR1 deliberately defers KernelLoader integration, so this uses a hand-rolled kernel —
//      hence the CUDA-only registration, mirroring test_block_cuda);
//   2. for each active level-1 voxel and each octree octant, reads its child through the
//      MultiResPartition built by MultiResField::make_partition(1) and checks the value equals
//      encode(0, child_global);
//   3. for each active level-0 voxel, reads its parent through make_partition(0) and checks the
//      value equals encode(1, parent_global).
//
// All cross-level reads go through device pointers + the parent/child block-id tables uploaded
// at construction, so this exercises the real device path (not the host round-trip in
// test_mgrid_static).

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/devices.h>
#include <carbon/mgrid.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

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
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("test_mgrid_cross_level: no CUDA devices visible — skipping\n");
    return 0;
  }
  if (devices.size() > 1U) {
    devices.erase(devices.begin() + 1, devices.end());
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  const carbon::int64_3d finest{16, 16, 16};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_cross_level");

  const int ref_factor = mblock.ref_factor();
  if (mblock.grid(0).block_count() != 8U || mblock.grid(1).block_count() != 1U) {
    std::fprintf(stderr, "test_mgrid_cross_level: unexpected block counts L0=%u L1=%u\n",
                 mblock.grid(0).block_count(), mblock.grid(1).block_count());
    return 1;
  }

  auto field = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "f");

  cudaSetDevice(system.gpu_group().devices()[0].device_id());
  const dim3 thread_block(Mem, Mem, Mem);

  for (int l = 0; l < mblock.depth(); ++l) {
    int32_t* mem = field.level_field(l).device_partition().ptr();
    fill_level_kernel<<<dim3(mblock.grid(l).block_count(), 1, 1), thread_block>>>(
        mem, mblock.grid(l).span(0), l);
  }
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
    std::fprintf(stderr, "test_mgrid_cross_level: fill sync -> %s\n", cudaGetErrorString(e));
    return 1;
  }

  const std::vector<size_t> counts{2U};
  carbon::Buffer<int> counters = system.gpu_group().buffer<int>(counts);
  carbon::Stream stream = system.stream();
  system.async_memset(counters[0], 0, sizeof(int) * 2U, stream);
  stream.wait();
  int* err = counters[0].data();
  int* ok = err + 1;

  // Children: level-1 voxels read their level-0 children.
  check_child_kernel<<<dim3(mblock.grid(1).block_count(), 1, 1), thread_block>>>(
      field.make_partition(1), mblock.grid(1).span(0), mblock.grid(0).span(0), ref_factor, err, ok);
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
    std::fprintf(stderr, "test_mgrid_cross_level: child sync -> %s\n", cudaGetErrorString(e));
    return 1;
  }

  // Parents: level-0 voxels read their level-1 parent.
  check_parent_kernel<<<dim3(mblock.grid(0).block_count(), 1, 1), thread_block>>>(
      field.make_partition(0), mblock.grid(0).span(0), mblock.grid(1).span(0), err, ok);
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) {
    std::fprintf(stderr, "test_mgrid_cross_level: parent sync -> %s\n", cudaGetErrorString(e));
    return 1;
  }

  auto host_counters = carbon::Cpu::buffer<int>(2U);
  system.async_copy(counters[0], host_counters, sizeof(int) * 2U, stream);
  stream.wait();
  const int err_count = host_counters.data()[0];
  const int ok_count = host_counters.data()[1];

  // 8 level-1 octant-children per voxel × 512 voxels = 4096; 4096 level-0 voxels × 1 parent.
  const int expected_ok = 4096 + 4096;
  if (err_count != 0 || ok_count != expected_ok) {
    std::fprintf(stderr, "test_mgrid_cross_level: err=%d ok=%d (expected ok=%d)\n", err_count,
                 ok_count, expected_ok);
    return 1;
  }
  std::printf("test_mgrid_cross_level: OK (cross-level reads checked: %d)\n", ok_count);
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
