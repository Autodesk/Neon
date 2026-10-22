// Smoke test for carbon::Block::make_kernel: builds a Block over a sphere predicate,
// allocates two Field<int32_t, 1> fields ("source" and "destination"), and exercises
// the polymorphic Kernel handle on the block-sparse data plane:
//
//   1. Kernel `init` writes  source(b, x, y, z) = encode(b, x, y, z) into every active
//      voxel via Block::make_kernel + the standard Neon-style loading lambda.
//   2. Kernel `copy` reads source and writes the same value into destination, again
//      through Block::make_kernel.
//
// After running both kernels through Kernel::launch, we read `destination` back to the
// host and check every active voxel carries the expected encoded value while every
// inactive voxel remains zero. This locks both the launch shape (one CUDA block per
// retained brick, brick-sized thread block, bitmask-gated) and the KernelLoader's
// duck-typed Field contract (partitions() / gpu_buffers() / make_halo_kernel() over
// detail::block::Field).

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<int32_t, 1>;

[[nodiscard]] CARBON_HD constexpr auto
encode(uint32_t b,
       int x,
       int y,
       int z) noexcept -> int32_t {
  return static_cast<int32_t>(b * 1000U + static_cast<uint32_t>(x) +
                              static_cast<uint32_t>(y) * 10U + static_cast<uint32_t>(z) * 100U) +
         1;
}

[[nodiscard]] auto
sphere_predicate(int64_t domain_n,
                 const carbon::int64_3d& gc) -> bool {
  const double cx = (domain_n - 1) * 0.5;
  const double cy = (domain_n - 1) * 0.5;
  const double cz = (domain_n - 1) * 0.5;
  const double r = domain_n * 0.45;
  const double dx = static_cast<double>(gc.x) - cx;
  const double dy = static_cast<double>(gc.y) - cy;
  const double dz = static_cast<double>(gc.z) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("test_block_kernel: no CUDA devices visible — skipping\n");
    return 0;
  }
  if (devices.size() > 1U) {
    devices.erase(devices.begin() + 1, devices.end());
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  constexpr int64_t domain_n = 24;
  const carbon::int64_3d domain_extent{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };
  BlockGrid block(system, domain_extent, predicate, "test_block_kernel");
  if (block.block_count() == 0U) {
    std::fprintf(stderr, "test_block_kernel: predicate retained no blocks\n");
    return 1;
  }

  FieldT source = block.template newField<int32_t, 1>(carbon::MemorySide::Device, "source");
  FieldT dest = block.template newField<int32_t, 1>(carbon::MemorySide::Device, "dest");

  carbon::Stream stream = system.stream();

  // Pre-zero both fields so inactive-voxel zeros are a clean baseline (the field
  // allocation does not zero the buffer).
  const size_t total_elements = static_cast<size_t>(block.block_count()) * SB::VoxelCount;
  system.async_memset(source.device_buffer()[0], 0, total_elements * sizeof(int32_t), stream);
  system.async_memset(dest.device_buffer()[0], 0, total_elements * sizeof(int32_t), stream);
  stream.wait();

  carbon::Kernel init = block.make_kernel([source](carbon::KernelLoader& loader) {
    auto src = loader.write(source);
    return [=] CARBON_HD(Span s, Idx idx) -> void {
      const auto intra = idx.intra();
      src.write(s, idx, 0, encode(idx.block_idx(), intra.x, intra.y, intra.z));
    };
  });

  carbon::Kernel copy = block.make_kernel([source, dest](carbon::KernelLoader& loader) {
    auto src = loader.read(source);
    auto dst = loader.write(dest);
    return [=] CARBON_HD(Span s, Idx idx) -> void {
      dst.write(s, idx, 0, src.read(s, idx, 0));
    };
  });

  init.launch(stream);
  copy.launch(stream);
  stream.wait();

  // Read-back and verify.
  auto host_alloc = carbon::Cpu::buffer<int32_t>(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    host_alloc.data()[i] = 0;
  }
  system.async_copy(dest.device_buffer()[0], host_alloc, total_elements * sizeof(int32_t), stream);
  stream.wait();

  int64_t active_checked = 0;
  int64_t inactive_checked = 0;
  for (uint32_t b = 0; b < block.block_count(); ++b) {
    const auto& bm = block.host_bitmasks()[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          const size_t lin = static_cast<size_t>(b) * SB::VoxelCount + static_cast<size_t>(x) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                 static_cast<size_t>(z);
          const int32_t got = host_alloc.data()[lin];
          if (bm.is_active(x, y, z)) {
            const int32_t want = encode(b, x, y, z);
            if (got != want) {
              std::fprintf(stderr,
                           "test_block_kernel: ACTIVE mismatch b=%u (%d,%d,%d) got %d want %d\n", b,
                           x, y, z, got, want);
              return 1;
            }
            ++active_checked;
          } else {
            if (got != 0) {
              std::fprintf(stderr, "test_block_kernel: INACTIVE leak b=%u (%d,%d,%d) got %d\n", b,
                           x, y, z, got);
              return 1;
            }
            ++inactive_checked;
          }
        }
      }
    }
  }
  if (active_checked != block.active_voxel_count()) {
    std::fprintf(stderr, "test_block_kernel: active count mismatch (checked %ld, expected %ld)\n",
                 static_cast<long>(active_checked), static_cast<long>(block.active_voxel_count()));
    return 1;
  }
  std::printf("test_block_kernel: OK (active=%ld, inactive_zero=%ld, blocks=%u)\n",
              static_cast<long>(active_checked), static_cast<long>(inactive_checked),
              block.block_count());
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
