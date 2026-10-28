#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/system.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>

namespace {

[[nodiscard]] auto
build_stencil_sum_kernel(carbon::Dense& dense,
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
build_map_sum_kernel(carbon::Dense& dense,
                     carbon::Dense::Field<float,
                                          1>& field,
                     carbon::ReductionResult<float>& out,
                     std::string name) -> carbon::Kernel {
  return carbon::Kernel::reduce(
      dense,
      [&field](carbon::KernelLoader& loader) {
        const auto in = loader.read(field);
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

// Stencil Map: w := neighbour-z average of v (needs halo on v).
[[nodiscard]] auto
build_stencil_avg_kernel(carbon::Dense& dense,
                         carbon::Dense::Field<float,
                                              1>& vin,
                         carbon::Dense::Field<float,
                                              1>& wout,
                         std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&vin, &wout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(vin);
        const auto out = loader.write(wout);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const auto zp = in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
          const auto zm = in.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
          const float self = in.read(s, idx, 0);
          if (!zp.valid || !zm.valid) {
            out.write(s, idx, 0, self);
            return;
          }
          out.write(s, idx, 0, (zp.value + zm.value) * 0.5F);
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

[[nodiscard]] auto
count_substr(std::string_view hay,
             std::string_view needle) -> size_t {
  size_t n = 0;
  for (size_t pos = 0; (pos = hay.find(needle, pos)) != std::string_view::npos;
       pos += needle.size()) {
    ++n;
  }
  return n;
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
  auto w = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "w");
  carbon::Stream s_up = sys.stream();
  carbon::Transfer<float>(sys, u.cpu_buffers(), u.gpu_buffers())(s_up);
  s_up.wait();

  carbon::ReductionResult<float> sum_ref(sys);
  carbon::Kernel whole = build_stencil_sum_kernel(dense, u, sum_ref, "sum_whole");
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
      sk.add(build_stencil_sum_kernel(dense, u, sum, "sum_u"));
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
      sk.add(build_stencil_sum_kernel(dense, v, sum, "sum_v"));
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

  // Neon Two-way Extended: Map reduce after a stencil is Partial+Merge; Extended leaves it whole.
  {
    carbon::ReductionResult<float> sum_none(sys);
    carbon::Skeleton sk_none(sys, carbon::Skeleton::Options{carbon::OverlapMode::None});
    sk_none.add(build_scale_kernel(dense, u, v, "scale_u_v"));
    sk_none.add(build_stencil_avg_kernel(dense, v, w, "stencil_v_w"));
    sk_none.add(build_map_sum_kernel(dense, w, sum_none, "sum_w"));
    sk_none.compile();
    sk_none.run();
    sk_none.wait();
    sum_none.sync(sys.stream());
    const float none_val = sum_none.host_value();

    carbon::ReductionResult<float> sum_ext(sys);
    carbon::Skeleton sk_ext(sys, carbon::Skeleton::Options{carbon::OverlapMode::Extended});
    sk_ext.add(build_scale_kernel(dense, u, v, "scale_u_v"));
    sk_ext.add(build_stencil_avg_kernel(dense, v, w, "stencil_v_w"));
    sk_ext.add(build_map_sum_kernel(dense, w, sum_ext, "sum_w"));
    sk_ext.compile();
    const std::string order_ext = sk_ext.export_launch_order();
    sk_ext.run();
    sk_ext.wait();
    sum_ext.sync(sys.stream());
    if (!matches(sum_ext.host_value(), none_val)) {
      CARBON_LOG_ERROR("test_skeleton_reduce_split[Extended trailing Map reduce]: {} != None {}",
                       sum_ext.host_value(), none_val);
      ok = false;
    }
    if (count_substr(order_ext, "launch  sum_w [reduction/internal]") != 0U ||
        count_substr(order_ext, "launch  sum_w [reduction/boundary]") != 0U ||
        count_substr(order_ext, "launch  sum_w [reduction/standard]") != 1U) {
      CARBON_LOG_ERROR(
          "test_skeleton_reduce_split[Extended]: expected one sum_w reduction/standard, got:\n{}",
          order_ext);
      ok = false;
    }

    carbon::ReductionResult<float> sum_tw(sys);
    carbon::Skeleton sk_tw(sys, carbon::Skeleton::Options{carbon::OverlapMode::TwoWayExtended});
    sk_tw.add(build_scale_kernel(dense, u, v, "scale_u_v"));
    sk_tw.add(build_stencil_avg_kernel(dense, v, w, "stencil_v_w"));
    sk_tw.add(build_map_sum_kernel(dense, w, sum_tw, "sum_w"));
    sk_tw.compile();
    const std::string order_tw = sk_tw.export_launch_order();
    sk_tw.run();
    sk_tw.wait();
    sum_tw.sync(sys.stream());
    if (!matches(sum_tw.host_value(), none_val)) {
      CARBON_LOG_ERROR(
          "test_skeleton_reduce_split[TwoWayExtended trailing Map reduce]: {} != None {}",
          sum_tw.host_value(), none_val);
      ok = false;
    }
    if (count_substr(order_tw, "launch  sum_w [reduction/internal]") != 1U ||
        count_substr(order_tw, "launch  sum_w [reduction/boundary]") != 1U ||
        count_substr(order_tw, "launch  sum_w [reduction/standard]") != 1U) {
      CARBON_LOG_ERROR(
          "test_skeleton_reduce_split[TwoWayExtended]: expected Partial Internal/Boundary + "
          "Merge for sum_w, got:\n{}",
          order_tw);
      ok = false;
    }
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
