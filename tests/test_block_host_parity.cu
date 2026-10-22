// Cross-target parity test for ::carbon::Kernel on the Block (block-sparse) family:
// the same loading lambda fed to Block::make_kernel (device) and Block::make_host_kernel
// (CPU / OpenMP) must produce bit-identical results on every active voxel. Mirrors
// test_kernel_host_parity.cu (Dense flavour) one-to-one so the matrix coverage row
// "OpenMP host kernel" can flip from no -> yes for Block in current_status.md.
//
// Setup:
//   - Single-GPU System (sliced off carbon::get_all_devices()).
//   - Block<DefaultStaticBlock> over a sphere predicate on a 24^3 domain.
//   - Field<int32_t, 1> built with MemorySide::HostDevice so both buffer sides exist.
//
// Compute parity:
//   - init lambda writes encode(b, x, y, z) into every active voxel.
//   - Run the *device* version into source.gpu_buffers, transfer back to source.cpu_buffers
//     via System::async_copy.
//   - Run the *host* version into a second field's cpu_buffers directly through OpenMP.
//   - Assert the two host mirrors agree on every active voxel and remain zero on every
//     inactive voxel.
//
// Reduction parity:
//   - same per-voxel `f(s, idx) -> int64` lambda used through Block::make_reduce_kernel and
//     Block::make_host_reduce_kernel. The host_value() returned by each ReductionResult
//     must match exactly (int64 sum has no rounding).
//
// Heterogeneous std::vector<Kernel>:
//   - device-compute -> Transfer (D->H) -> host-compute kernel sequence on a single
//     stream; the host kernel's stream.wait() flushes the prior Transfer, so the host
//     loop sees the device kernel's writes.

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
#include <carbon/transfer.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    std::printf("test_block_host_parity: no CUDA devices visible -- skipping\n");
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
  BlockGrid block(system, domain_extent, predicate, "test_block_host_parity");
  if (block.block_count() == 0U) {
    std::fprintf(stderr, "test_block_host_parity: predicate retained no blocks\n");
    return 1;
  }

  // ----- Compute parity -----
  FieldT dev_field =
      block.template newField<int32_t, 1>(carbon::MemorySide::HostDevice, "dev_field");
  FieldT host_field =
      block.template newField<int32_t, 1>(carbon::MemorySide::HostDevice, "host_field");

  carbon::Stream stream = system.stream();

  const size_t total_elements = static_cast<size_t>(block.block_count()) * SB::VoxelCount;
  // Pre-zero both host mirrors and the device side of dev_field so inactive voxels stay
  // at the documented zero baseline.
  for (size_t i = 0; i < total_elements; ++i) {
    dev_field.cpu_buffers()[0].data()[i] = 0;
    host_field.cpu_buffers()[0].data()[i] = 0;
  }
  system.async_memset(dev_field.device_buffer()[0], 0, total_elements * sizeof(int32_t), stream);
  stream.wait();

  // Device launch: writes into dev_field's device buffer.
  carbon::Kernel init_device = block.make_kernel([dev_field](carbon::KernelLoader& loader) {
    auto dst = loader.write(dev_field);
    return [=] CARBON_HD(Span s, Idx idx) -> void {
      const auto intra = idx.intra();
      dst.write(s, idx, 0, encode(idx.block_idx(), intra.x, intra.y, intra.z));
    };
  });

  // Host launch: writes directly into host_field's host mirror via the same closure.
  carbon::Kernel init_host = block.make_host_kernel([host_field](carbon::KernelLoader& loader) {
    auto dst = loader.write(host_field);
    return [=] CARBON_HD(Span s, Idx idx) -> void {
      const auto intra = idx.intra();
      dst.write(s, idx, 0, encode(idx.block_idx(), intra.x, intra.y, intra.z));
    };
  });

  init_device.launch(stream);
  // Copy the device-written result back to dev_field's CPU mirror so we can diff.
  system.async_copy(dev_field.device_buffer()[0], dev_field.cpu_buffers()[0],
                    total_elements * sizeof(int32_t), stream);
  init_host.launch(stream);
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
          const int32_t got_dev = dev_field.cpu_buffers()[0].data()[lin];
          const int32_t got_host = host_field.cpu_buffers()[0].data()[lin];
          if (bm.is_active(x, y, z)) {
            const int32_t want = encode(b, x, y, z);
            if (got_dev != want) {
              std::fprintf(stderr,
                           "test_block_host_parity: device ACTIVE mismatch b=%u (%d,%d,%d) got %d "
                           "want %d\n",
                           b, x, y, z, got_dev, want);
              return 1;
            }
            if (got_host != want) {
              std::fprintf(stderr,
                           "test_block_host_parity: host ACTIVE mismatch b=%u (%d,%d,%d) got %d "
                           "want %d\n",
                           b, x, y, z, got_host, want);
              return 1;
            }
            if (got_dev != got_host) {
              std::fprintf(stderr,
                           "test_block_host_parity: device/host disagree b=%u (%d,%d,%d) dev=%d "
                           "host=%d\n",
                           b, x, y, z, got_dev, got_host);
              return 1;
            }
            ++active_checked;
          } else {
            if (got_dev != 0 || got_host != 0) {
              std::fprintf(stderr,
                           "test_block_host_parity: INACTIVE leak b=%u (%d,%d,%d) dev=%d host=%d\n",
                           b, x, y, z, got_dev, got_host);
              return 1;
            }
            ++inactive_checked;
          }
        }
      }
    }
  }
  if (active_checked != block.active_voxel_count()) {
    std::fprintf(stderr,
                 "test_block_host_parity: active count mismatch (checked %ld, expected %ld)\n",
                 static_cast<long>(active_checked), static_cast<long>(block.active_voxel_count()));
    return 1;
  }

  // ----- Reduction parity -----
  // Reuse the host mirror written above as the input field; sum (intra.x + intra.y +
  // intra.z + block_idx) over every active voxel through both reduction paths and
  // check the totals agree.
  ReduceField sum_input =
      block.template newField<int64_t, 1>(carbon::MemorySide::HostDevice, "sum_input");

  // Init sum_input through the device kernel (use the parity closure) then mirror back
  // for the host reducer.
  carbon::Kernel init_sum_input = block.make_kernel([sum_input](carbon::KernelLoader& loader) {
    auto dst = loader.write(sum_input);
    return [=] CARBON_HD(Span s, Idx idx) -> void {
      const auto intra = idx.intra();
      const int64_t v = static_cast<int64_t>(intra.x) + static_cast<int64_t>(intra.y) +
                        static_cast<int64_t>(intra.z) + static_cast<int64_t>(idx.block_idx());
      dst.write(s, idx, 0, v);
    };
  });
  init_sum_input.launch(stream);
  system.async_copy(sum_input.device_buffer()[0], sum_input.cpu_buffers()[0],
                    total_elements * sizeof(int64_t), stream);
  stream.wait();

  carbon::ReductionResult<int64_t> dst_dev(system);
  carbon::ReductionResult<int64_t> dst_host(system);

  auto sum_loading = [sum_input](carbon::KernelLoader& loader) {
    auto src = loader.read(sum_input);
    return [=] CARBON_HD(Span s, Idx idx) -> int64_t {
      return src.read(s, idx, 0);
    };
  };

  carbon::Kernel reduce_dev = block.template make_reduce_kernel<int64_t>(
      sum_loading, dst_dev, carbon::reduce::Sum<int64_t>{});
  carbon::Kernel reduce_host = block.template make_host_reduce_kernel<int64_t>(
      sum_loading, dst_host, carbon::reduce::Sum<int64_t>{});

  reduce_dev.launch(stream);
  dst_dev.sync(stream);
  reduce_host.launch(stream);
  stream.wait();

  const int64_t got_dev_sum = dst_dev.host_value();
  const int64_t got_host_sum = dst_host.host_value();
  if (got_dev_sum != got_host_sum) {
    std::fprintf(stderr, "test_block_host_parity: reduce dev=%lld host=%lld\n",
                 static_cast<long long>(got_dev_sum), static_cast<long long>(got_host_sum));
    return 1;
  }

  // Reference sum: iterate the host bitmasks directly.
  int64_t expected_sum = 0;
  for (uint32_t b = 0; b < block.block_count(); ++b) {
    const auto& bm = block.host_bitmasks()[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          if (bm.is_active(x, y, z)) {
            expected_sum +=
                static_cast<int64_t>(x) + static_cast<int64_t>(y) + static_cast<int64_t>(z) + b;
          }
        }
      }
    }
  }
  if (got_host_sum != expected_sum) {
    std::fprintf(stderr, "test_block_host_parity: reduce expected %lld got %lld\n",
                 static_cast<long long>(expected_sum), static_cast<long long>(got_host_sum));
    return 1;
  }

  std::printf("test_block_host_parity: OK (active=%ld, inactive=%ld, blocks=%u, sum=%lld)\n",
              static_cast<long>(active_checked), static_cast<long>(inactive_checked),
              block.block_count(), static_cast<long long>(got_host_sum));
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
