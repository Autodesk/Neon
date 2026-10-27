#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/system.h>

#include <cmath>
#include <cstdlib>
#include <string_view>

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

[[nodiscard]] auto
build_scale_kernel(carbon::Dense& dense,
                   carbon::Dense::Field<float,
                                        1>& src,
                   carbon::Dense::Field<float,
                                        1>& dst,
                   std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&src, &dst](carbon::KernelLoader& loader) {
        const auto in = loader.read(src);
        const auto out = loader.write(dst);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          out.write(s, idx, 0, 2.0F * in.read(s, idx, 0));
        };
      },
      std::move(name), /*shared_mem=*/0U);
}

[[nodiscard]] auto
mode_name(carbon::OverlapMode mode) -> std::string_view {
  switch (mode) {
    case carbon::OverlapMode::None:
      return "None";
    case carbon::OverlapMode::Standard:
      return "Standard";
    case carbon::OverlapMode::Extended:
      return "Extended";
    case carbon::OverlapMode::TwoWayExtended:
      return "TwoWayExtended";
  }
  return "?";
}

[[nodiscard]] auto
matches(float got,
        float ref) -> bool {
  return std::fabs(got - ref) <= 1e-3F * std::fabs(ref);
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
  auto v = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v");
  carbon::Stream s_up = sys.stream();
  carbon::Transfer<float>(sys, u.cpu_buffers(), u.gpu_buffers())(s_up);
  s_up.wait();

  carbon::ReductionResult<float> sum_ref(sys);
  carbon::Kernel whole = build_sum_kernel(dense, u, sum_ref, "sum_whole");
  whole.launch(sys.stream(), carbon::DataView::Standard);
  sys.stream().wait();
  sum_ref.sync(sys.stream());
  const float ref_val = sum_ref.host_value();

  bool ok = true;
  for (const carbon::OverlapMode mode :
       {carbon::OverlapMode::Standard, carbon::OverlapMode::Extended,
        carbon::OverlapMode::TwoWayExtended}) {
    // Stencil reduction straight off a halo: the split must merge its partials.
    {
      carbon::ReductionResult<float> sum(sys);
      carbon::Skeleton sk(sys, carbon::Skeleton::Options{mode});
      sk.add(u.make_halo_kernel("halo_u"));
      sk.add(build_sum_kernel(dense, u, sum, "sum_u"));
      sk.compile();
      sk.run();
      sk.wait();
      sum.sync(sys.stream());
      if (!matches(sum.host_value(), ref_val)) {
        CARBON_LOG_ERROR("test_skeleton_reduce_split[{}]: split {} != ref {}", mode_name(mode),
                         sum.host_value(), ref_val);
        ok = false;
      }
    }
    // Map producer feeding the stencil reduction: Extended splits the producer too.
    {
      carbon::ReductionResult<float> sum(sys);
      carbon::Skeleton sk(sys, carbon::Skeleton::Options{mode});
      sk.add(build_scale_kernel(dense, u, v, "scale_u_v"));
      sk.add(build_sum_kernel(dense, v, sum, "sum_v"));
      sk.compile();
      sk.run();
      sk.wait();
      sum.sync(sys.stream());
      if (!matches(sum.host_value(), 2.0F * ref_val)) {
        CARBON_LOG_ERROR("test_skeleton_reduce_split[{} + producer]: split {} != ref {}",
                         mode_name(mode), sum.host_value(), 2.0F * ref_val);
        ok = false;
      }
    }
  }
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
