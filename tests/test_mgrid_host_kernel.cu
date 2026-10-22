// OpenMP host-kernel parity test for the mGrid cross-level kernel layer —
// MultiResBlock::make_host_kernel / make_host_reduce_kernel + MultiResField::level_view
// bound through KernelLoader on the CPU mirror. Mirrors test_mgrid_kernel.cu (device path)
// without hand-rolled kernels or D2H readback: every check reads cpu_buffers() directly.
//
// Single-GPU (forced) over a fully-active depth-2 octree on 16×16×32 finest so the
// block counts match the device test (L0: 8192 parent checks, L1: 8192 child matches).

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/devices.h>
#include <carbon/kernel.h>
#include <carbon/mgrid.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using SBlock = carbon::DefaultMultiResBlock;
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
sum_level_host(const carbon::MultiResBlock<SBlock>& mblock,
               int level,
               const carbon::MultiResBlock<SBlock>::Field<int32_t,
                                                          1>& field) -> int64_t {
  int64_t total = 0;
  const auto& cpu = field.level_field(level).cpu_buffers();
  for (size_t g = 0; g < cpu.size(); ++g) {
    const size_t n =
        static_cast<size_t>(mblock.grid(level).block_count(g)) * static_cast<size_t>(VoxelCount);
    for (size_t i = 0; i < n; ++i) {
      total += cpu[g].data()[i];
    }
  }
  return total;
}

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("test_mgrid_host_kernel: no CUDA devices visible — skipping\n");
    return 0;
  }
  if (devices.size() > 1U) {
    devices.erase(devices.begin() + 1, devices.end());
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  const carbon::int64_3d finest{16, 16, 32};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_host_kernel");
  const int ref_factor = mblock.ref_factor();

  auto f = mblock.newField<int32_t, 1>(carbon::MemorySide::HostDevice, "f");
  auto out = mblock.newField<int32_t, 1>(carbon::MemorySide::HostDevice, "out");

  carbon::Stream stream = system.stream();

  for (int l = 0; l < mblock.depth(); ++l) {
    carbon::Kernel fill = mblock.make_host_kernel(l, [f, l](carbon::KernelLoader& loader) {
      auto p = loader.write(f.level_view(l));
      return [p, l](Span s, Idx idx) -> void {
        const auto g = s.global_location(idx);
        p.write(s, idx, 0,
                encode(l, static_cast<int>(g.x), static_cast<int>(g.y), static_cast<int>(g.z)));
      };
    });
    fill.launch(stream);
  }

  carbon::Kernel parent_check =
      mblock.make_host_kernel(0, [f, out, ref_factor](carbon::KernelLoader& loader) {
        auto pf = loader.read(f.level_view(0));
        auto po = loader.write(out.level_view(0));
        return [pf, po, ref_factor](Span s, Idx idx) -> void {
          const auto g = s.global_location(idx);
          const int32_t expected =
              encode(1, static_cast<int>(g.x / ref_factor), static_cast<int>(g.y / ref_factor),
                     static_cast<int>(g.z / ref_factor));
          const auto pv = pf.parent(s, idx, 0);
          po.write(s, idx, 0, (pv.valid && pv.value == expected) ? 1 : 0);
        };
      });
  parent_check.launch(stream);

  carbon::Kernel child_check =
      mblock.make_host_kernel(1, [f, out, ref_factor](carbon::KernelLoader& loader) {
        auto pf = loader.read(f.level_view(1));
        auto po = loader.write(out.level_view(1));
        return [pf, po, ref_factor](Span s, Idx idx) -> void {
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

  const int64_t total = sum_level_host(mblock, 0, out) + sum_level_host(mblock, 1, out);
  const int64_t expected_total = 8192 + 8192;
  if (total != expected_total) {
    std::fprintf(stderr, "test_mgrid_host_kernel: cross-level read total %ld != expected %ld\n",
                 static_cast<long>(total), static_cast<long>(expected_total));
    return 1;
  }

  carbon::ReductionResult<int64_t> parent_matches(system);
  carbon::Kernel parent_reduce = mblock.make_host_reduce_kernel<int64_t>(
      0,
      [f, ref_factor](carbon::KernelLoader& loader) {
        auto pf = loader.read(f.level_view(0));
        return [pf, ref_factor](Span s, Idx idx) -> int64_t {
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

  carbon::ReductionResult<int64_t> child_matches(system);
  carbon::Kernel child_reduce = mblock.make_host_reduce_kernel<int64_t>(
      1,
      [f, ref_factor](carbon::KernelLoader& loader) {
        auto pf = loader.read(f.level_view(1));
        return [pf, ref_factor](Span s, Idx idx) -> int64_t {
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

  const int64_t reduced_parents = parent_matches.host_value(carbon::DataView::Standard);
  const int64_t reduced_children = child_matches.host_value(carbon::DataView::Standard);
  if (reduced_parents != 8192 || reduced_children != 8192) {
    std::fprintf(stderr,
                 "test_mgrid_host_kernel: cross-level host reduce mismatch (parents=%ld "
                 "children=%ld, expected 8192/8192)\n",
                 static_cast<long>(reduced_parents), static_cast<long>(reduced_children));
    return 1;
  }

  std::printf("test_mgrid_host_kernel: OK (cross-level reads via OpenMP: %ld; reduce: %ld+%ld)\n",
              static_cast<long>(total), static_cast<long>(reduced_parents),
              static_cast<long>(reduced_children));
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
