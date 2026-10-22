// Device correctness test for the carbon::MultiResBlock (mGrid) *kernel layer* —
// MultiResBlock::make_kernel + MultiResField::level_view bound through the ordinary
// KernelLoader, with NO hand-rolled <<<>>> kernel (unlike test_mgrid_cross_level /
// test_mgrid_multi_gpu, which predate the kernel layer). This is the deliverable of the mGrid
// runtime-port kernel PR: cross-level reads written exactly like every other grid family's
// compute kernel.
//
// Builds a fully-active depth-2 octree over a 16×16×N finest domain across every visible GPU
// (N is a multiple of MemZ·R and at least one coarsest brick row per GPU, so the multi-slab
// z-alignment precondition holds), then:
//
//   1. fill: per level, a make_kernel(level) writes value = encode(level, global_voxel) into
//      field `f` via loader.write(f.level_view(level)).
//   2. parent check: a make_kernel(0) reads each level-0 voxel's parent through
//      loader.read(f.level_view(0)) (a cross-level mgrid::Partition) and writes 1/0 into `out`
//      level-0 according to whether the parent value matches encode(1, parent_global).
//   3. child check: a make_kernel(1) reads each level-1 voxel's 8 octree children through
//      loader.read(f.level_view(1)) and writes the count of matching children into `out`
//      level-1.
//   4. reduce: the same cross-level parent/child reads driven through
//      MultiResBlock::make_reduce_kernel — the per-cell contribution returns the match indicator
//      and the 3-stage reduce folds it across every brick and GPU (== 8192 each).
//
// `out` is then read back with Block::download_to_host (owned bricks only) and summed on the
// host. Because the tower is fully active every owned voxel is written, so the sum is the
// total number of correct cross-level reads — independent of how many GPUs the slabs split
// across. CUDA-only registration: mGrid SYCL parity is a tracked follow-up (matches the rest
// of the mGrid test suite); nothing here is intrinsically CUDA-specific.

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/devices.h>
#include <carbon/kernel.h>
#include <carbon/mgrid.h>

#include "test_domain.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using SBlock = carbon::DefaultMultiResBlock;  // 8³ memory brick / 2³ user block (octree).
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;

constexpr int VoxelCount = SBlock::VoxelCount;

[[nodiscard]] CARBON_HD constexpr auto
encode(int level,
       int x,
       int y,
       int z) noexcept -> int32_t {
  return level * 1000000 + (x + 100 * y + 10000 * z);
}

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("test_mgrid_kernel: no CUDA devices visible — skipping\n");
    return 0;
  }
  const size_t n_gpus = devices.size();
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  // The coarsest level's brick spans MemZ·R^(depth-1) = 8·2 = 16 finest cells; grow z so every
  // GPU still gets a coarsest brick row, keeping the z-alignment precondition satisfied.
  const carbon::int64_3d finest{16, 16, carbon::test::block_slab_z(32, n_gpus, SBlock::MemZ * 2)};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_kernel");
  const int ref_factor = mblock.ref_factor();

  auto f = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "f");
  auto out = mblock.newField<int32_t, 1>(carbon::MemorySide::Device, "out");

  carbon::Stream stream = system.stream();

  // 1. Fill every level's own storage with encode(level, global) — own-level write through the
  //    cross-level partition the level view hands back.
  for (int l = 0; l < mblock.depth(); ++l) {
    carbon::Kernel fill = mblock.make_kernel(l, [f, l](carbon::KernelLoader& loader) {
      auto p = loader.write(f.level_view(l));
      return [p, l] CARBON_HD(Span s, Idx idx) -> void {
        const auto g = s.global_location(idx);
        p.write(s, idx, 0,
                encode(l, static_cast<int>(g.x), static_cast<int>(g.y), static_cast<int>(g.z)));
      };
    });
    fill.launch(stream);
  }

  // 2. Parent check (level 0 reads level 1). Co-resident, so this is a pure local dereference.
  carbon::Kernel parent_check =
      mblock.make_kernel(0, [f, out, ref_factor](carbon::KernelLoader& loader) {
        auto pf = loader.read(f.level_view(0));
        auto po = loader.write(out.level_view(0));
        return [pf, po, ref_factor] CARBON_HD(Span s, Idx idx) -> void {
          const auto g = s.global_location(idx);
          const int32_t expected =
              encode(1, static_cast<int>(g.x / ref_factor), static_cast<int>(g.y / ref_factor),
                     static_cast<int>(g.z / ref_factor));
          const auto pv = pf.parent(s, idx, 0);
          po.write(s, idx, 0, (pv.valid && pv.value == expected) ? 1 : 0);
        };
      });
  parent_check.launch(stream);

  // 3. Child check (level 1 reads level 0): count matching octree children per level-1 voxel.
  carbon::Kernel child_check =
      mblock.make_kernel(1, [f, out, ref_factor](carbon::KernelLoader& loader) {
        auto pf = loader.read(f.level_view(1));
        auto po = loader.write(out.level_view(1));
        return [pf, po, ref_factor] CARBON_HD(Span s, Idx idx) -> void {
          const auto g = s.global_location(idx);
          int matches = 0;
          for (int8_t oz = 0; oz < ref_factor; ++oz) {
            for (int8_t oy = 0; oy < ref_factor; ++oy) {
              for (int8_t ox = 0; ox < ref_factor; ++ox) {
                const carbon::int8_3d off{ox, oy, oz};
                const int32_t expected = encode(0, static_cast<int>(ref_factor * g.x + ox),
                                                static_cast<int>(ref_factor * g.y + oy),
                                                static_cast<int>(ref_factor * g.z + oz));
                const auto cv = pf.child(s, idx, off, 0);
                if (cv.valid && cv.value == expected) {
                  ++matches;
                }
              }
            }
          }
          po.write(s, idx, 0, matches);
        };
      });
  child_check.launch(stream);

  // Read `out` back (owned bricks only) and sum every written entry.
  auto host_l0 = mblock.grid(0).download_to_host<int32_t, 1>(out.level_field(0), stream);
  auto host_l1 = mblock.grid(1).download_to_host<int32_t, 1>(out.level_field(1), stream);
  stream.wait();

  int64_t total = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const size_t n0 =
        static_cast<size_t>(mblock.grid(0).block_count(g)) * static_cast<size_t>(VoxelCount);
    for (size_t i = 0; i < n0; ++i) {
      total += host_l0[g].data()[i];
    }
    const size_t n1 =
        static_cast<size_t>(mblock.grid(1).block_count(g)) * static_cast<size_t>(VoxelCount);
    for (size_t i = 0; i < n1; ++i) {
      total += host_l1[g].data()[i];
    }
  }

  // Every L0 voxel has one matching parent, and every L1 voxel has R³ matching children, so both
  // directions total the finest voxel count. GPU-count-independent (sum over all slabs).
  const int64_t finest_voxels = finest.x * finest.y * finest.z;
  const int64_t expected_total = finest_voxels + finest_voxels;
  if (total != expected_total) {
    std::fprintf(stderr,
                 "test_mgrid_kernel: cross-level read total %ld != expected %ld (%zu GPU(s))\n",
                 static_cast<long>(total), static_cast<long>(expected_total), n_gpus);
    return 1;
  }

  // 4. Same cross-level reads, but through make_reduce_kernel — the per-cell contribution reads a
  //    parent / child and returns the match indicator; the 3-stage reduce folds it across every
  //    brick and GPU. Sum over L0 parents == 8192, sum over L1 child matches == 8192.
  carbon::ReductionResult<int64_t> parent_matches(system);
  carbon::Kernel parent_reduce = mblock.make_reduce_kernel<int64_t>(
      0,
      [f, ref_factor](carbon::KernelLoader& loader) {
        auto pf = loader.read(f.level_view(0));
        return [pf, ref_factor] CARBON_HD(Span s, Idx idx) -> int64_t {
          const auto g = s.global_location(idx);
          const int32_t expected =
              encode(1, static_cast<int>(g.x / ref_factor), static_cast<int>(g.y / ref_factor),
                     static_cast<int>(g.z / ref_factor));
          const auto pv = pf.parent(s, idx, 0);
          return (pv.valid && pv.value == expected) ? 1 : 0;
        };
      },
      parent_matches);
  parent_reduce.launch(stream, carbon::DataView::Standard);
  parent_matches.sync(stream);
  stream.wait();

  carbon::ReductionResult<int64_t> child_matches(system);
  carbon::Kernel child_reduce = mblock.make_reduce_kernel<int64_t>(
      1,
      [f, ref_factor](carbon::KernelLoader& loader) {
        auto pf = loader.read(f.level_view(1));
        return [pf, ref_factor] CARBON_HD(Span s, Idx idx) -> int64_t {
          const auto g = s.global_location(idx);
          int64_t matches = 0;
          for (int8_t oz = 0; oz < ref_factor; ++oz) {
            for (int8_t oy = 0; oy < ref_factor; ++oy) {
              for (int8_t ox = 0; ox < ref_factor; ++ox) {
                const carbon::int8_3d off{ox, oy, oz};
                const int32_t expected = encode(0, static_cast<int>(ref_factor * g.x + ox),
                                                static_cast<int>(ref_factor * g.y + oy),
                                                static_cast<int>(ref_factor * g.z + oz));
                const auto cv = pf.child(s, idx, off, 0);
                if (cv.valid && cv.value == expected) {
                  ++matches;
                }
              }
            }
          }
          return matches;
        };
      },
      child_matches);
  child_reduce.launch(stream, carbon::DataView::Standard);
  child_matches.sync(stream);
  stream.wait();

  const int64_t reduced_parents = parent_matches.host_value(carbon::DataView::Standard);
  const int64_t reduced_children = child_matches.host_value(carbon::DataView::Standard);
  if (reduced_parents != finest_voxels || reduced_children != finest_voxels) {
    std::fprintf(stderr,
                 "test_mgrid_kernel: cross-level reduce mismatch (parents=%ld children=%ld, "
                 "expected %ld/%ld)\n",
                 static_cast<long>(reduced_parents), static_cast<long>(reduced_children),
                 static_cast<long>(finest_voxels), static_cast<long>(finest_voxels));
    return 1;
  }

  std::printf(
      "test_mgrid_kernel: OK (%zu GPU(s), cross-level reads via KernelLoader: %ld; reduce: "
      "%ld+%ld)\n",
      n_gpus, static_cast<long>(total), static_cast<long>(reduced_parents),
      static_cast<long>(reduced_children));
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
