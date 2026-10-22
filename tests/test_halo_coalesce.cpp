#include <carbon/cpu.h>
#include <carbon/detail/allocation.h>
#include <carbon/detail/halo_build.h>

#include <cassert>
#include <cstddef>
#include <vector>

namespace {

using carbon::detail::Allocation;
using carbon::detail::AllocationLocation;
using carbon::detail::halo_build::coalesce_contiguous_host_intra;
using carbon::detail::halo_build::coalesce_contiguous_transfers;

struct HostIntraEntry {
  size_t src_gpu{};
  size_t dst_gpu{};
  size_t src_offset{};
  size_t dst_offset{};
  size_t count{};
};

auto
test_merge_contiguous_run() -> void {
  auto buf = carbon::Cpu::buffer<float>(64U);
  std::vector<Allocation<float>> src(3, buf);
  std::vector<Allocation<float>> dst(3, buf);
  std::vector<size_t> src_first{0U, 4U, 8U};
  std::vector<size_t> src_counts{4U, 4U, 4U};
  std::vector<size_t> dst_first{16U, 20U, 24U};
  std::vector<size_t> dst_counts{4U, 4U, 4U};

  coalesce_contiguous_transfers(src, dst, src_first, src_counts, dst_first, dst_counts);

  assert(src.size() == 1U);
  assert(src_first.size() == 1U && src_first[0] == 0U);
  assert(src_counts[0] == 12U);
  assert(dst_first[0] == 16U);
  assert(dst_counts[0] == 12U);
}

auto
test_gap_splits_merge() -> void {
  auto buf = carbon::Cpu::buffer<float>(64U);
  std::vector<Allocation<float>> src(2, buf);
  std::vector<Allocation<float>> dst(2, buf);
  std::vector<size_t> src_first{0U, 8U};
  std::vector<size_t> src_counts{4U, 4U};
  std::vector<size_t> dst_first{16U, 32U};
  std::vector<size_t> dst_counts{4U, 4U};

  coalesce_contiguous_transfers(src, dst, src_first, src_counts, dst_first, dst_counts);

  assert(src.size() == 2U);
  assert(src_first[0] == 0U && src_first[1] == 8U);
}

auto
test_never_merge_across_allocations() -> void {
  auto buf_a = carbon::Cpu::buffer<float>(32U);
  auto buf_b = carbon::Cpu::buffer<float>(32U);
  std::vector<Allocation<float>> src{buf_a, buf_b};
  std::vector<Allocation<float>> dst{buf_a, buf_b};
  std::vector<size_t> src_first{0U, 0U};
  std::vector<size_t> src_counts{4U, 4U};
  std::vector<size_t> dst_first{8U, 8U};
  std::vector<size_t> dst_counts{4U, 4U};

  coalesce_contiguous_transfers(src, dst, src_first, src_counts, dst_first, dst_counts);

  assert(src.size() == 2U);
}

auto
test_host_merge_contiguous_run() -> void {
  std::vector<HostIntraEntry> entries{
      {0U, 1U, 0U, 16U, 4U},
      {0U, 1U, 4U, 20U, 4U},
      {0U, 1U, 8U, 24U, 4U},
  };
  coalesce_contiguous_host_intra(entries);
  assert(entries.size() == 1U);
  assert(entries[0].count == 12U);
}

auto
test_host_gap_splits_merge() -> void {
  std::vector<HostIntraEntry> entries{
      {0U, 1U, 0U, 16U, 4U},
      {0U, 1U, 8U, 32U, 4U},
  };
  coalesce_contiguous_host_intra(entries);
  assert(entries.size() == 2U);
}

auto
test_host_never_merge_across_gpus() -> void {
  std::vector<HostIntraEntry> entries{
      {0U, 1U, 0U, 8U, 4U},
      {1U, 0U, 0U, 8U, 4U},
  };
  coalesce_contiguous_host_intra(entries);
  assert(entries.size() == 2U);
}

}  // namespace

int
main() {
  test_merge_contiguous_run();
  test_gap_splits_merge();
  test_never_merge_across_allocations();
  test_host_merge_contiguous_run();
  test_host_gap_splits_merge();
  test_host_never_merge_across_gpus();
  return 0;
}
