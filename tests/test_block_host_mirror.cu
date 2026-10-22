// Block::newField + MemorySide::HostDevice / MemorySide::Host smoke test.
//
// Verifies that allocating a carbon::Block::Field with a CPU mirror does the
// right thing:
//   - cpu_buffers() exposes one host Allocation<T> per local GPU sized
//     identically to gpu_buffers() (n_total bricks * VoxelCount * Components
//     scalars).
//   - host_partitions() carries one Partition entry per GPU pointing at the
//     CPU mirror.
//   - The CPU mirror is a real, writable allocation: writes to it land in
//     the host_partition's data pointer and survive a round-trip through
//     System::async_copy to the device buffer and back.
//   - A MemorySide::Device field still has empty cpu_buffers / host_partitions,
//     so the existing device-only contract is unchanged.
//
// The test stays single-GPU on purpose — multi-GPU behaviour is exercised by
// the existing test_block_multi_gpu suite and the layout is identical
// (BufferGroup-of-Allocation indexed by local GPU). What this test locks down
// is the new "allocate + expose + round-trip" plumbing.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/devices.h>
#include <carbon/integer3d.h>
#include <carbon/memory_side.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using FieldT = BlockGrid::Field<int32_t, 1>;

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
    std::printf("test_block_host_mirror: no CUDA devices visible — skipping\n");
    return 0;
  }
  if (devices.size() > 1U) {
    devices.erase(devices.begin() + 1, devices.end());
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  constexpr int64_t domain_n = 16;
  const carbon::int64_3d domain_extent{domain_n, domain_n, domain_n};
  const auto predicate = [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_predicate(domain_n, gc);
  };
  BlockGrid block(system, domain_extent, predicate, "test_block_host_mirror");
  if (block.block_count() == 0U) {
    std::fprintf(stderr, "test_block_host_mirror: predicate retained no blocks\n");
    return 1;
  }

  // Reference: Device-only field. Should expose an empty CPU mirror.
  FieldT dev_only = block.template newField<int32_t, 1>(carbon::MemorySide::Device, "dev_only");
  if (!dev_only.cpu_buffers().empty()) {
    std::fprintf(stderr,
                 "test_block_host_mirror: MemorySide::Device field has non-empty cpu_buffers "
                 "(size=%zu); expected 0\n",
                 dev_only.cpu_buffers().size());
    return 1;
  }
  if (!dev_only.host_partitions().empty()) {
    std::fprintf(stderr,
                 "test_block_host_mirror: MemorySide::Device field has non-empty "
                 "host_partitions (size=%zu); expected 0\n",
                 dev_only.host_partitions().size());
    return 1;
  }

  // Field-under-test: HostDevice mirror — both sides allocated.
  FieldT hd_field = block.template newField<int32_t, 1>(carbon::MemorySide::HostDevice, "hostdev");
  if (hd_field.cpu_buffers().size() != hd_field.gpu_buffers().size()) {
    std::fprintf(stderr,
                 "test_block_host_mirror: HostDevice cpu_buffers size (%zu) != gpu_buffers size "
                 "(%zu)\n",
                 hd_field.cpu_buffers().size(), hd_field.gpu_buffers().size());
    return 1;
  }
  if (hd_field.host_partitions().size() != hd_field.partitions().size()) {
    std::fprintf(stderr,
                 "test_block_host_mirror: HostDevice host_partitions size (%zu) != partitions "
                 "size (%zu)\n",
                 hd_field.host_partitions().size(), hd_field.partitions().size());
    return 1;
  }
  for (size_t g = 0; g < hd_field.gpu_buffers().size(); ++g) {
    if (hd_field.cpu_buffers()[g].size() != hd_field.gpu_buffers()[g].size()) {
      std::fprintf(stderr,
                   "test_block_host_mirror: HostDevice cpu_buffers[%zu].size (%zu) != "
                   "gpu_buffers[%zu].size (%zu)\n",
                   g, hd_field.cpu_buffers()[g].size(), g, hd_field.gpu_buffers()[g].size());
      return 1;
    }
  }

  // Round-trip: write a sentinel pattern into the CPU mirror, copy host -> device
  // -> host (into a fresh allocation) and assert the bytes round-trip cleanly.
  // Per-GPU sized identically to gpu_buffers, so one allocation pass suffices.
  int32_t* host_ptr = hd_field.cpu_buffers()[0].data();
  const size_t n_elements = hd_field.cpu_buffers()[0].size();
  if (n_elements == 0U) {
    std::fprintf(stderr, "test_block_host_mirror: HostDevice mirror is empty (n_elements=0)\n");
    return 1;
  }
  for (size_t i = 0; i < n_elements; ++i) {
    host_ptr[i] = static_cast<int32_t>(i) ^ 0x5A5A;
  }

  carbon::Stream stream = system.stream();
  try {
    system.async_copy(hd_field.cpu_buffers()[0], hd_field.gpu_buffers()[0],
                      n_elements * sizeof(int32_t), stream);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_block_host_mirror: H->D async_copy failed: %s\n", e.what());
    return 1;
  }
  stream.wait();

  // Zero the CPU mirror so we know the readback came from the device.
  for (size_t i = 0; i < n_elements; ++i) {
    host_ptr[i] = 0;
  }

  try {
    system.async_copy(hd_field.gpu_buffers()[0], hd_field.cpu_buffers()[0],
                      n_elements * sizeof(int32_t), stream);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_block_host_mirror: D->H async_copy failed: %s\n", e.what());
    return 1;
  }
  stream.wait();

  for (size_t i = 0; i < n_elements; ++i) {
    const int32_t want = static_cast<int32_t>(i) ^ 0x5A5A;
    if (host_ptr[i] != want) {
      std::fprintf(stderr,
                   "test_block_host_mirror: round-trip mismatch at i=%zu: got %d, want %d\n", i,
                   host_ptr[i], want);
      return 1;
    }
  }

  // Verify host_partition[0].data() points at the same memory as cpu_buffers[0].
  if (hd_field.host_partitions()[0].data() != hd_field.cpu_buffers()[0].data()) {
    std::fprintf(stderr,
                 "test_block_host_mirror: host_partition[0].data() (%p) != cpu_buffers[0].data() "
                 "(%p)\n",
                 static_cast<const void*>(hd_field.host_partitions()[0].data()),
                 static_cast<const void*>(hd_field.cpu_buffers()[0].data()));
    return 1;
  }

  // Field built with MemorySide::Host (CPU mirror only, but device buffer still
  // exists — we keep the bidirectional shape symmetric with Dense::Field).
  FieldT host_field = block.template newField<int32_t, 1>(carbon::MemorySide::Host, "host_only");
  if (host_field.cpu_buffers().size() != host_field.gpu_buffers().size() ||
      host_field.cpu_buffers().empty()) {
    std::fprintf(stderr,
                 "test_block_host_mirror: MemorySide::Host cpu_buffers size (%zu) inconsistent "
                 "with gpu_buffers (%zu)\n",
                 host_field.cpu_buffers().size(), host_field.gpu_buffers().size());
    return 1;
  }

  std::printf(
      "test_block_host_mirror: OK (n_elements=%zu, gpus=%zu, dev_only.cpu=%zu, "
      "hostdev.cpu=%zu, host.cpu=%zu)\n",
      n_elements, hd_field.cpu_buffers().size(), dev_only.cpu_buffers().size(),
      hd_field.cpu_buffers().size(), host_field.cpu_buffers().size());
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
