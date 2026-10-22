// Multi-GPU deferred scalar readback via Skeleton (root-GPU D→H on copy stream).

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/devices.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>

#include <cmath>
#include <cstdlib>
#include <exception>

int
main() {
  try {
    carbon::init();
    std::vector<carbon::detail::Gpu> gpus = carbon::get_all_devices();
    if (gpus.size() < 2U) {
      CARBON_LOG_INFO("test_skeleton_scalar_pattern_multi_gpu: <2 GPUs, skip");
      return EXIT_SUCCESS;
    }

    carbon::System sys(gpus);
    const carbon::int64_3d global{16, 1, 16};
    carbon::Dense grid(sys, global, carbon::Stencil{});

    auto field = grid.newField<float, 1>(carbon::MemorySide::HostDevice, "f");
    field.fill_owned_host([](carbon::int64_3d, int) { return 1.0F; });
    carbon::Stream up = sys.stream();
    carbon::Transfer<float>(sys, field.cpu_buffers(), field.gpu_buffers())(up);
    up.wait();

    carbon::ReductionResult<float> dst(sys, carbon::reduce::ReductionResultOptions{.slots = 2U});
    carbon::reduce::ReduceOptions opts{};
    opts.host_copy = carbon::reduce::ReduceHostCopy::Deferred;

    carbon::Kernel reduce_k = grid.make_reduce_kernel<float>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> float {
            return in.read(s, idx, 0);
          };
        },
        dst, carbon::reduce::Sum<float>{}, 0U, opts);

    carbon::Skeleton sk(sys);
    sk.add(std::move(reduce_k));
    sk.compile();

    if (sk.effective_kernels().size() != 2U) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern_multi_gpu: expected 2 effective kernels");
      return EXIT_FAILURE;
    }

    sk.run();
    sk.wait();
    const float v = dst.wait_value();
    const float want = static_cast<float>(global.x * global.y * global.z);
    if (std::fabs(v - want) > 1.0e-3F * want) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern_multi_gpu: sum {} expected ~{}", v, want);
      return EXIT_FAILURE;
    }

    CARBON_LOG_INFO("test_skeleton_scalar_pattern_multi_gpu: PASS ({} GPUs)", gpus.size());
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    CARBON_LOG_ERROR("test_skeleton_scalar_pattern_multi_gpu: {}", ex.what());
    return EXIT_FAILURE;
  }
}
