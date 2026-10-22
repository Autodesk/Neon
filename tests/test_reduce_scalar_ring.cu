// ReductionResult slot ring, pinned host, wait_value after Inline reduce.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>

namespace {

constexpr int64_t kW = 8;
constexpr int64_t kH = 8;
constexpr int64_t kD = 4;

}  // namespace

int
main() {
  try {
    carbon::init();
    std::vector<carbon::detail::Gpu> gpus = carbon::get_all_devices();
    if (gpus.empty()) {
      CARBON_LOG_INFO("test_reduce_scalar_ring: no GPU, skip");
      return EXIT_SUCCESS;
    }
    carbon::System sys(std::vector<carbon::detail::Gpu>{std::move(gpus.front())});
    const carbon::int64_3d global{kW, kH, kD};
    carbon::Dense grid(sys, global, carbon::Stencil{});

    carbon::Dense::Field<float, 1> f =
        grid.newField<float, 1>(carbon::MemorySide::HostDevice, "f");
    f.fill_owned_host([](carbon::int64_3d, int) { return 1.0F; });
    carbon::Stream up = sys.stream();
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(up);
    up.wait();

    carbon::ReductionResult<float> dst(sys, carbon::reduce::ReductionResultOptions{.slots = 2U});
    auto reduce_k = grid.make_reduce_kernel<float>(
        [&f](carbon::KernelLoader& loader) {
          const auto in = loader(f);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> float {
            return in.read(s, idx, 0);
          };
        },
        dst, carbon::reduce::Sum<float>{});

    carbon::Stream compute = sys.stream();
    reduce_k.launch(compute);
    compute.wait();

    const float v0 = dst.wait_value();
    const float want = static_cast<float>(kW * kH * kD);
    if (std::fabs(v0 - want) > 1.0e-4F) {
      CARBON_LOG_ERROR("test_reduce_scalar_ring: first sum got {} want {}", v0, want);
      return EXIT_FAILURE;
    }

    f.fill_owned_host([](carbon::int64_3d, int) { return 2.0F; });
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(up);
    up.wait();
    reduce_k.launch(compute);
    compute.wait();

    const float v1 = dst.wait_value();
    const float want2 = static_cast<float>(2.0 * static_cast<double>(want));
    if (std::fabs(v1 - want2) > 1.0e-3F) {
      CARBON_LOG_ERROR("test_reduce_scalar_ring: second sum got {} want {}", v1, want2);
      return EXIT_FAILURE;
    }

    CARBON_LOG_INFO("test_reduce_scalar_ring: PASS");
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    CARBON_LOG_ERROR("test_reduce_scalar_ring: {}", ex.what());
    return EXIT_FAILURE;
  }
}
