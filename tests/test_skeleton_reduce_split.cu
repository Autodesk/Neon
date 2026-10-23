#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/system.h>

#include <cmath>
#include <cstdlib>

namespace {

[[nodiscard]] auto
build_sum_kernel(carbon::Dense& dense,
                 carbon::Dense::Field<float,
                                      1>& field,
                 carbon::ReductionResult<float>& out,
                 std::string name) -> carbon::Kernel {
  return carbon::Kernel::reduce(
      dense,
      [&field](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(field);
        return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> float {
          return in.read(s, idx, 0);
        };
      },
      out, carbon::reduce::Sum<float>{}, std::move(name));
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    return EXIT_SUCCESS;
  }
  carbon::System sys(carbon::get_all_devices());
  carbon::Dense dense(sys, {16, 8, 16});
  auto u = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u");
  u.fill_owned_host([](carbon::int64_3d, int) {
    return 1.0F;
  });
  carbon::Stream s_up = sys.stream();
  carbon::Transfer<float>(sys, u.cpu_buffers(), u.gpu_buffers())(s_up);
  s_up.wait();

  carbon::ReductionResult<float> sum(sys);
  carbon::Skeleton sk(sys, carbon::Skeleton::Options{carbon::OverlapMode::Standard});
  sk.add(u.make_halo_kernel("halo_u"));
  sk.add(build_sum_kernel(dense, u, sum, "sum_u"));
  sk.compile();
  sk.run();
  sk.wait();
  sum.sync(sys.stream());
  const float split_val = sum.host_value();

  carbon::ReductionResult<float> sum_ref(sys);
  carbon::Kernel whole = build_sum_kernel(dense, u, sum_ref, "sum_whole");
  whole.launch(sys.stream(), carbon::DataView::Standard);
  sys.stream().wait();
  sum_ref.sync(sys.stream());
  const float ref_val = sum_ref.host_value();

  if (std::fabs(split_val - ref_val) > 1e-3F * std::fabs(ref_val)) {
    CARBON_LOG_ERROR("test_skeleton_reduce_split: split {} != ref {}", split_val, ref_val);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
