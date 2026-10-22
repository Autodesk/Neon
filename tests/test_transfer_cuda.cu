#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/transfer.h>

#include <cassert>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

constexpr size_t kCount = 32U;

using carbon::detail::Allocation;
using carbon::detail::BufferGroup;

// Build an N-entry host carbon::Buffer<T>, one entry per GPU all sized @p n.
template <typename T>
auto
make_host_group(size_t gpu_count,
                size_t n) -> carbon::Buffer<T> {
  std::vector<Allocation<T>> v;
  v.reserve(gpu_count);
  for (size_t i = 0; i < gpu_count; ++i) {
    v.push_back(carbon::Cpu::buffer<T>(n));
  }
  return BufferGroup<T>::from_vector(std::move(v));
}

auto
fill_host(carbon::Buffer<float>& bg,
          float base) -> void {
  for (size_t g = 0; g < bg.size(); ++g) {
    auto& buf = bg[g];
    for (size_t i = 0; i < buf.size(); ++i) {
      buf.data()[i] = base + static_cast<float>(i);
    }
  }
}

// H2D + D2H round-trip: copy host -> device -> host and verify the payload survives on every GPU.
auto
test_round_trip(carbon::System& sys) -> void {
  const size_t n_gpus = sys.gpu_group().size();
  auto h_src = make_host_group<float>(n_gpus, kCount);
  fill_host(h_src, 1.0F);

  auto d = sys.gpu_group().buffer<float>(std::vector<size_t>(n_gpus, kCount));
  auto h_dst = make_host_group<float>(n_gpus, kCount);
  for (size_t g = 0; g < n_gpus; ++g) {
    for (size_t i = 0; i < kCount; ++i) {
      h_dst[g].data()[i] = -1.0F;
    }
  }

  carbon::Stream stream = sys.stream();
  carbon::Transfer<float> h2d(sys, h_src, d);
  carbon::Transfer<float> d2h(sys, d, h_dst);

  h2d(stream);
  d2h(stream);
  stream.wait();

  for (size_t g = 0; g < n_gpus; ++g) {
    for (size_t i = 0; i < kCount; ++i) {
      assert(h_dst[g].data()[i] == h_src[g].data()[i]);
    }
  }
}

// Same-device D2D using explicit per-GPU sub-range pointers (per-GPU window [start, start+len)).
auto
test_same_device_d2d(carbon::System& sys) -> void {
  const size_t n_gpus = sys.gpu_group().size();
  auto h_src = make_host_group<float>(n_gpus, kCount);
  fill_host(h_src, 100.0F);

  auto d_src = sys.gpu_group().buffer<float>(std::vector<size_t>(n_gpus, kCount));
  auto d_dst = sys.gpu_group().buffer<float>(std::vector<size_t>(n_gpus, kCount));

  carbon::Stream stream = sys.stream();
  carbon::Transfer<float> upload(sys, h_src, d_src);
  upload(stream);

  constexpr size_t start = 4U;
  constexpr size_t len = 12U;
  std::vector<size_t> sf(n_gpus, start);
  std::vector<size_t> df(n_gpus, 0U);
  std::vector<size_t> sc(n_gpus, len);
  std::vector<size_t> dc(n_gpus, len);
  carbon::Transfer<float> d2d(sys, d_src, std::move(sf), std::move(sc), d_dst, std::move(df),
                              std::move(dc));
  d2d(stream);

  auto h_check = make_host_group<float>(n_gpus, len);
  std::vector<size_t> sf2(n_gpus, 0U);
  std::vector<size_t> df2(n_gpus, 0U);
  std::vector<size_t> sc2(n_gpus, len);
  std::vector<size_t> dc2(n_gpus, len);
  carbon::Transfer<float> d2h(sys, d_dst, std::move(sf2), std::move(sc2), h_check, std::move(df2),
                              std::move(dc2));
  d2h(stream);
  stream.wait();

  for (size_t g = 0; g < n_gpus; ++g) {
    for (size_t i = 0; i < len; ++i) {
      assert(h_check[g].data()[i] == h_src[g].data()[start + i]);
    }
  }
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_transfer_cuda: no CUDA devices");
    return 1;
  }

  carbon::System sys(carbon::get_all_devices());
  if (sys.gpu_group().empty()) {
    CARBON_LOG_ERROR("test_transfer_cuda: empty GpuGroup");
    return 1;
  }

  test_round_trip(sys);
  test_same_device_d2d(sys);
  return 0;
}
