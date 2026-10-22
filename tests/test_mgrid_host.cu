// Host mirror wiring for carbon::MultiResField::level_view — verifies
// @c host_partitions() is populated on a @c MemorySide::HostDevice tower field and
// that cross-level @ref mgrid::Partition writes land in @c cpu_buffers() . OpenMP host
// kernel launch over mGrid levels remains covered by the Block-family host suites;
// this test locks the mGrid-specific host partition gather ( @ref make_host_partition ).

#include <carbon/carbon.h>
#include <carbon/devices.h>
#include <carbon/mgrid.h>

#include <cmath>
#include <cstdio>

namespace {

using SBlock = carbon::DefaultMultiResBlock;
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;

constexpr int VoxelCount = SBlock::VoxelCount;

}  // namespace

int
main() {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("test_mgrid_host: no CUDA devices — skipping\n");
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
                                       "mgrid_host");
  auto u = mblock.newField<double, 1>(carbon::MemorySide::HostDevice, "u");

  const auto lv = u.level_view(0);
  if (lv.host_partitions().size() != system.gpu_group().size()) {
    std::fprintf(stderr, "test_mgrid_host: level_view host_partitions size mismatch\n");
    return 1;
  }
  const auto& part = lv.host_partitions()[0];
  if (part.ptr() == nullptr) {
    std::fprintf(stderr, "test_mgrid_host: null host partition pointer\n");
    return 1;
  }
  if (part.ptr() != u.level_field(0).cpu_buffers()[0].data()) {
    std::fprintf(stderr, "test_mgrid_host: host partition pointer != cpu_buffers data\n");
    return 1;
  }
  const Span& span = mblock.grid(0).host_spans()[0];
  size_t written = 0;
  const uint32_t iter_b = span.iter_block_count(carbon::DataView::Standard);
  for (uint32_t iter_idx = 0; iter_idx < iter_b; ++iter_idx) {
    const uint32_t b = span.view_block_idx(carbon::DataView::Standard, iter_idx);
    for (int iz = 0; iz < SBlock::MemZ; ++iz) {
      for (int iy = 0; iy < SBlock::MemY; ++iy) {
        for (int ix = 0; ix < SBlock::MemX; ++ix) {
          Idx idx{};
          if (!span.set_and_validate(b, ix, iy, iz, idx)) {
            continue;
          }
          const auto g = span.global_location(idx);
          part.write(span, idx, 0, static_cast<double>(g.x + g.y + g.z));
          ++written;
        }
      }
    }
  }
  if (written == 0U) {
    std::fprintf(stderr, "test_mgrid_host: wrote 0 cells\n");
    return 1;
  }

  size_t mismatches = 0;
  for (uint32_t b = 0; b < mblock.grid(0).block_count(0); ++b) {
    const auto& bm = mblock.grid(0).host_bitmasks(0)[b];
    for (int z = 0; z < SBlock::MemZ; ++z) {
      for (int y = 0; y < SBlock::MemY; ++y) {
        for (int x = 0; x < SBlock::MemX; ++x) {
          if (!bm.is_active(x, y, z)) {
            continue;
          }
          Idx idx{};
          if (!span.set_and_validate(b, x, y, z, idx)) {
            continue;
          }
          const double got = part.read(span, idx, 0);
          const auto g = span.global_location(idx);
          const double want = static_cast<double>(g.x + g.y + g.z);
          if (std::fabs(got - want) > 1e-9) {
            ++mismatches;
          }
        }
      }
    }
  }
  if (mismatches > 0U) {
    std::fprintf(stderr, "test_mgrid_host: %zu mismatches (wrote %zu cells)\n", mismatches,
                 written);
    return 1;
  }
  std::printf("test_mgrid_host: OK (wrote %zu cells, host_partitions=%zu)\n", written,
              lv.host_partitions().size());
  return 0;
}
