// Smoke + parity test for Block::make_universal_kernel and
// Block::make_universal_reduce_kernel: one loading lambda, two launch paths
// (device + host), bit-identical results on every active voxel and on the
// reduction scalar. The Universal kernels are stored in a heterogeneous
// std::vector<Kernel> and dispatched through Kernel::launch_on_device /
// Kernel::launch_on_host exactly as user code would.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/system.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<int32_t, 1>;
using ReduceField = BlockGrid::Field<int64_t, 1>;

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
  const double r = domain_n * 0.4;
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
    std::printf("test_block_universal: no devices visible -- skipping\n");
    return 0;
  }
  // Restrict to a single GPU so the test exercises the simplest stage-2
  // single-GPU short-circuit and runs the same way on hosts with 1 vs N GPUs.
  if (devices.size() > 1U) {
    devices.erase(devices.begin() + 1, devices.end());
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  constexpr int64_t domain_n = 24;
  const carbon::int64_3d domain_extent{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };
  BlockGrid block(system, domain_extent, predicate, "test_block_universal");
  if (block.block_count() == 0U) {
    std::fprintf(stderr, "test_block_universal: predicate retained no blocks\n");
    return 1;
  }

  FieldT field_dev = block.template newField<int32_t, 1>(carbon::MemorySide::HostDevice, "u_dev");
  FieldT field_host = block.template newField<int32_t, 1>(carbon::MemorySide::HostDevice, "u_host");

  carbon::Stream stream = system.stream();

  const size_t total = static_cast<size_t>(block.block_count()) * SB::VoxelCount;
  for (size_t i = 0; i < total; ++i) {
    field_dev.cpu_buffers()[0].data()[i] = 0;
    field_host.cpu_buffers()[0].data()[i] = 0;
  }
  system.async_memset(field_dev.device_buffer()[0], 0, total * sizeof(int32_t), stream);
  stream.wait();

  // One Universal Kernel handle drives both targets. The loader resolves
  // partitions per-target (device vs host) automatically; the same per-cell
  // closure runs unchanged on both sides.
  carbon::Kernel init_dev = block.make_universal_kernel([field_dev](carbon::KernelLoader& loader) {
    auto dst = loader.write(field_dev);
    return [=] CARBON_HD(Span s, Idx idx) -> void {
      const auto intra = idx.intra();
      dst.write(s, idx, 0, encode(idx.block_idx(), intra.x, intra.y, intra.z));
    };
  });
  carbon::Kernel init_host =
      block.make_universal_kernel([field_host](carbon::KernelLoader& loader) {
        auto dst = loader.write(field_host);
        return [=] CARBON_HD(Span s, Idx idx) -> void {
          const auto intra = idx.intra();
          dst.write(s, idx, 0, encode(idx.block_idx(), intra.x, intra.y, intra.z));
        };
      });

  init_dev.launch_on_device(stream);
  system.async_copy(field_dev.device_buffer()[0], field_dev.cpu_buffers()[0],
                    total * sizeof(int32_t), stream);
  init_host.launch_on_host(stream);
  stream.wait();

  int64_t active_checked = 0;
  for (uint32_t b = 0; b < block.block_count(); ++b) {
    const auto& bm = block.host_bitmasks()[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          const size_t lin = static_cast<size_t>(b) * SB::VoxelCount + static_cast<size_t>(x) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                 static_cast<size_t>(z);
          const int32_t got_dev = field_dev.cpu_buffers()[0].data()[lin];
          const int32_t got_host = field_host.cpu_buffers()[0].data()[lin];
          if (bm.is_active(x, y, z)) {
            const int32_t want = encode(b, x, y, z);
            if (got_dev != want || got_host != want) {
              std::fprintf(stderr,
                           "test_block_universal: parity miss b=%u (%d,%d,%d) dev=%d host=%d "
                           "want=%d\n",
                           b, x, y, z, got_dev, got_host, want);
              return 1;
            }
            ++active_checked;
          } else if (got_dev != 0 || got_host != 0) {
            std::fprintf(stderr,
                         "test_block_universal: inactive leak b=%u (%d,%d,%d) dev=%d host=%d\n", b,
                         x, y, z, got_dev, got_host);
            return 1;
          }
        }
      }
    }
  }

  // Universal reduce: one Kernel handle, exercise launch_on_device then
  // launch_on_host into the same ReductionResult — must agree.
  ReduceField sum_input =
      block.template newField<int64_t, 1>(carbon::MemorySide::HostDevice, "sum_input");
  carbon::Kernel init_sum = block.make_universal_kernel([sum_input](carbon::KernelLoader& loader) {
    auto dst = loader.write(sum_input);
    return [=] CARBON_HD(Span s, Idx idx) -> void {
      const auto intra = idx.intra();
      const int64_t v = static_cast<int64_t>(intra.x) + static_cast<int64_t>(intra.y) +
                        static_cast<int64_t>(intra.z) + static_cast<int64_t>(idx.block_idx());
      dst.write(s, idx, 0, v);
    };
  });
  init_sum.launch_on_device(stream);
  system.async_copy(sum_input.device_buffer()[0], sum_input.cpu_buffers()[0],
                    total * sizeof(int64_t), stream);
  stream.wait();

  carbon::ReductionResult<int64_t> dst_dev(system);
  carbon::ReductionResult<int64_t> dst_host(system);
  auto sum_loading = [sum_input](carbon::KernelLoader& loader) {
    auto src = loader.read(sum_input);
    return [=] CARBON_HD(Span s, Idx idx) -> int64_t {
      return src.read(s, idx, 0);
    };
  };

  carbon::Kernel reduce_dev = block.template make_universal_reduce_kernel<int64_t>(
      sum_loading, dst_dev, carbon::reduce::Sum<int64_t>{});
  carbon::Kernel reduce_host = block.template make_universal_reduce_kernel<int64_t>(
      sum_loading, dst_host, carbon::reduce::Sum<int64_t>{});

  reduce_dev.launch_on_device(stream);
  dst_dev.sync(stream);
  reduce_host.launch_on_host(stream);
  stream.wait();

  const int64_t got_dev_sum = dst_dev.host_value();
  const int64_t got_host_sum = dst_host.host_value();
  int64_t expected = 0;
  for (uint32_t b = 0; b < block.block_count(); ++b) {
    const auto& bm = block.host_bitmasks()[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          if (bm.is_active(x, y, z)) {
            expected += static_cast<int64_t>(x) + static_cast<int64_t>(y) +
                        static_cast<int64_t>(z) + static_cast<int64_t>(b);
          }
        }
      }
    }
  }
  if (got_dev_sum != expected || got_host_sum != expected) {
    std::fprintf(stderr, "test_block_universal: reduce mismatch dev=%lld host=%lld expected=%lld\n",
                 static_cast<long long>(got_dev_sum), static_cast<long long>(got_host_sum),
                 static_cast<long long>(expected));
    return 1;
  }

  // Heterogeneous vector: device + host kernels mixed; iterate with the
  // polymorphic launch_on_device / launch_on_host overloads.
  std::vector<carbon::Kernel> pipeline;
  pipeline.push_back(init_dev);
  pipeline.push_back(init_host);
  for (auto& k : pipeline) {
    k.launch_on_device(stream);
  }
  for (auto& k : pipeline) {
    k.launch_on_host(stream);
  }
  stream.wait();

  std::printf("test_block_universal: OK (blocks=%u, active=%ld, sum=%lld)\n", block.block_count(),
              static_cast<long>(active_checked), static_cast<long long>(got_host_sum));
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
