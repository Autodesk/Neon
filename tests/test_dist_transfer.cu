// Cross-rank NCCL transfer test.
//
// Launched via `mpiexec -n 2 ./test_dist_transfer`. Each rank fills its single GPU buffer with
// a rank-tagged pattern, then issues one DistTransfer that posts (a) a send to the peer rank
// of the lower half of the buffer and (b) a recv from the peer rank into the upper half. After
// the group runs, the upper half on each rank must contain the lower-half pattern produced by
// the other rank. Skipped gracefully when fewer than two MPI ranks are launched or when no
// CUDA devices are visible.

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/dist_transfer.h>
#include <carbon/distributed.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace {

using carbon::detail::Allocation;
using carbon::detail::BufferGroup;

constexpr size_t kHalf = 64U;
constexpr size_t kCount = kHalf * 2U;

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_dist_transfer: world_size={} < 2, skipping", distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_dist_transfer: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int peer = (rank == 0) ? 1 : 0;

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_dist_transfer: expected gpu_group size 1, got {}",
                     sys.gpu_group().size());
    return 1;
  }
  const auto& gpu = sys.gpu_group().devices()[0];

  // Build a single-entry device buffer of 2 * kHalf elements: [0, kHalf) is "ours" (filled
  // with a rank-tagged pattern); [kHalf, 2*kHalf) is the receive slot for the peer's lower
  // half.
  std::vector<Allocation<std::int32_t>> dev_v;
  dev_v.push_back(gpu.buffer<std::int32_t>(kCount, /*relative_gpu_id=*/0));
  carbon::Buffer<std::int32_t> dev = BufferGroup<std::int32_t>::from_vector(std::move(dev_v));

  auto h_src = carbon::Cpu::buffer<std::int32_t>(kCount);
  const std::int32_t base = (rank + 1) * 1000;
  for (size_t i = 0; i < kHalf; ++i) {
    h_src.data()[i] = base + static_cast<std::int32_t>(i);
  }
  for (size_t i = kHalf; i < kCount; ++i) {
    // Sentinel that must be overwritten by the peer's send.
    h_src.data()[i] = -1;
  }
  std::vector<Allocation<std::int32_t>> h_src_v;
  h_src_v.push_back(std::move(h_src));
  carbon::Buffer<std::int32_t> h_src_grp =
      BufferGroup<std::int32_t>::from_vector(std::move(h_src_v));

  auto h_dst = carbon::Cpu::buffer<std::int32_t>(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    h_dst.data()[i] = 0;
  }
  std::vector<Allocation<std::int32_t>> h_dst_v;
  h_dst_v.push_back(std::move(h_dst));
  carbon::Buffer<std::int32_t> h_dst_grp =
      BufferGroup<std::int32_t>::from_vector(std::move(h_dst_v));

  carbon::Stream stream = sys.stream();

  carbon::Transfer<std::int32_t> upload(sys, h_src_grp, dev);
  upload(stream);

  // One DistTransfer with two entries: send our [0, kHalf) to peer; recv into [kHalf, kCount)
  // from peer. Both go into one ncclGroupStart/End block.
  std::vector<int> peer_ranks{peer, peer};
  std::vector<carbon::DistOp> ops{carbon::DistOp::Send, carbon::DistOp::Recv};
  std::vector<size_t> first_elements{0U, kHalf};
  std::vector<size_t> counts{kHalf, kHalf};
  carbon::DistTransfer<std::int32_t> exch(sys, dev, std::move(peer_ranks), std::move(ops),
                                          std::move(first_elements), std::move(counts));
  exch(stream);

  carbon::Transfer<std::int32_t> download(sys, dev, h_dst_grp);
  download(stream);

  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dist_transfer: stream.wait() failed: {}", e.what());
    return 1;
  }

  // Validate: lower half is unchanged (still our pattern); upper half is the peer's lower
  // half (peer base + i).
  const std::int32_t peer_base = (peer + 1) * 1000;
  for (size_t i = 0; i < kHalf; ++i) {
    const std::int32_t got = h_dst_grp[0].data()[i];
    const std::int32_t want = base + static_cast<std::int32_t>(i);
    if (got != want) {
      CARBON_LOG_ERROR("test_dist_transfer: rank {} lower mismatch at i={}: got {} want {}", rank,
                       i, got, want);
      return 1;
    }
  }
  for (size_t i = 0; i < kHalf; ++i) {
    const std::int32_t got = h_dst_grp[0].data()[kHalf + i];
    const std::int32_t want = peer_base + static_cast<std::int32_t>(i);
    if (got != want) {
      CARBON_LOG_ERROR("test_dist_transfer: rank {} upper mismatch at i={}: got {} want {}", rank,
                       i, got, want);
      return 1;
    }
  }

  CARBON_LOG_INFO("test_dist_transfer: rank {} of {} ok", rank, distributed->world_size());
  return 0;
}
