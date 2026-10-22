// Cross-GPU peer transfer test.
//
// Exercises the peer-copy path of System::async_copy by building two single-entry device
// BufferGroups whose allocations live on two different GPUs and running a Transfer from
// one to the other. Skipped gracefully when fewer than two devices are visible. The
// underlying primitive is cudaMemcpyPeerAsync under the CUDA backend and a SYCL queue
// memcpy across the process-wide shared sycl::context under the SYCL backend.

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/transfer.h>

#include <cassert>
#include <cstdint>
#include <exception>
#include <utility>
#include <vector>

namespace {

using carbon::detail::Allocation;
using carbon::detail::BufferGroup;

constexpr size_t kCount = 64U;

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    // Two distinct devices are required to drive the peer-copy path; treat fewer as a no-op
    // pass so the test slot stays meaningful in single-GPU CI without false negatives.
    CARBON_LOG_INFO("test_transfer_peer: fewer than 2 devices, skipping");
    return 0;
  }

  carbon::System sys(carbon::get_all_devices());
  if (sys.gpu_group().size() < 2) {
    CARBON_LOG_INFO("test_transfer_peer: GpuGroup has <2 GPUs, skipping");
    return 0;
  }

  const auto& gpu0 = sys.gpu_group().devices()[0];
  const auto& gpu1 = sys.gpu_group().devices()[1];

  // Build two single-entry BufferGroups: src lives on gpu0 (relative_gpu_id 0), dst on gpu1
  // (relative_gpu_id 1). The Transfer below sees one pair whose endpoints sit on different
  // devices, which forces System::async_copy onto the cross-device peer branch.
  std::vector<Allocation<float>> src_v;
  src_v.push_back(gpu0.buffer<float>(kCount, /*relative_gpu_id=*/0));
  carbon::Buffer<float> src = BufferGroup<float>::from_vector(std::move(src_v));

  std::vector<Allocation<float>> dst_v;
  dst_v.push_back(gpu1.buffer<float>(kCount, /*relative_gpu_id=*/1));
  carbon::Buffer<float> dst = BufferGroup<float>::from_vector(std::move(dst_v));

  // Stage a known pattern on the host, push it onto src (gpu0), peer-copy to dst (gpu1),
  // pull it back to a separate host buffer, and check it survived the round-trip.
  auto h_src = carbon::Cpu::buffer<float>(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    h_src.data()[i] = 100.0F + static_cast<float>(i);
  }
  std::vector<Allocation<float>> h_src_v;
  h_src_v.push_back(std::move(h_src));
  carbon::Buffer<float> h_src_grp = BufferGroup<float>::from_vector(std::move(h_src_v));

  auto h_dst = carbon::Cpu::buffer<float>(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    h_dst.data()[i] = -1.0F;
  }
  std::vector<Allocation<float>> h_dst_v;
  h_dst_v.push_back(std::move(h_dst));
  carbon::Buffer<float> h_dst_grp = BufferGroup<float>::from_vector(std::move(h_dst_v));

  carbon::Stream stream = sys.stream();

  // h_src(host) -> src(gpu0). async_copy picks the stream by the device side of the copy, so
  // this lands on stream[0] (gpu0 is the destination).
  carbon::Transfer<float> upload(sys, h_src_grp, src);
  upload(stream);

  // The peer copy below is issued on stream[1] (gpu1 is its destination) but reads gpu0's
  // buffer, which the upload above fills on stream[0]. Carbon does not order one GPU's
  // stream against another's implicitly — that is the cross-GPU producer hazard documented
  // on carbon::Halo, which fences internally for exactly this reason. A raw Transfer leaves
  // the fence to the caller, so drain here. Without this the peer copy can race ahead and
  // read gpu0's still-unwritten (zeroed) allocation; the window only opens when the GPUs are
  // busy enough for the upload to lag, which is why this needs concurrent load to show up.
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_transfer_peer: stream.wait after upload failed: {}", e.what());
    return 1;
  }

  // src(gpu0) -> dst(gpu1) -- peer copy
  carbon::Transfer<float> peer(sys, src, dst);
  peer(stream);

  // dst(gpu1) -> h_dst(host). Same stream[1] as the peer copy above, so stream ordering
  // alone sequences it behind the copy.
  carbon::Transfer<float> download(sys, dst, h_dst_grp);
  download(stream);

  // Walking the whole stream group keeps the test correct if the allocator ever changes.
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_transfer_peer: stream.wait failed: {}", e.what());
    return 1;
  }

  for (size_t i = 0; i < kCount; ++i) {
    const float got = h_dst_grp[0].data()[i];
    const float want = h_src_grp[0].data()[i];
    if (got != want) {
      CARBON_LOG_ERROR("test_transfer_peer: mismatch at i={}: got {}, want {}", i, got, want);
      return 1;
    }
  }

  return 0;
}
