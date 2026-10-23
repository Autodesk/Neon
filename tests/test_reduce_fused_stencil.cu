// Fused stencil write + sum reduction in one Kernel::reduce launch inside a Skeleton.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <vector>

namespace {

constexpr int64_t kWidth = 12;
constexpr int64_t kHeight = 8;
constexpr int64_t kDepth = 10;

[[nodiscard]] auto
make_stencil() -> carbon::Stencil {
  return carbon::Stencil(
      std::vector<carbon::int32_3d>{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}},
      /*remove_center=*/true);
}

using Dense = carbon::Dense;
using Span = Dense::Span;
using Idx = Dense::Idx;
using Field = Dense::Field<float, 1>;

[[nodiscard]] auto
fields_match(const Dense& dense,
             const Field& got,
             const Field& expect) -> bool {
  for (size_t g = 0; g < dense.gpu_count(); ++g) {
    const Span& span = dense.spans()[g];
    const float* h_got = got.cpu_buffers()[g].data();
    const float* h_exp = expect.cpu_buffers()[g].data();
    const carbon::int64_3d owned = span.logic_extent();
    for (int64_t iz = 0; iz < owned.z; ++iz) {
      for (int64_t iy = 0; iy < owned.y; ++iy) {
        for (int64_t ix = 0; ix < owned.x; ++ix) {
          const auto vi = span.set_and_validate(carbon::DataView::Standard, ix, iy, iz);
          if (!vi.valid) {
            continue;
          }
          const int64_t off = span.linear_offset(vi.idx, 0);
          if (std::fabs(h_got[off] - h_exp[off]) > 1.0e-5F) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] auto
build_fused(Dense& dense,
            Field& u,
            Field& u_next,
            carbon::ReductionResult<float>& sumsq) -> carbon::Kernel {
  return dense.make_reduce_kernel<float>(
      [&u, &u_next](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(u);
        const auto old_p = loader.read(u);
        const auto out = loader.write(u_next);
        return [in, old_p, out] CARBON_HOST_DEVICE(Span s, Idx idx) -> float {
          const float xp = in.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0).value;
          const float xm = in.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0).value;
          const float yp = in.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0).value;
          const float ym = in.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0).value;
          const float old = old_p.read(s, idx, 0);
          const float next = 0.25F * (xp + xm + yp + ym);
          out.write(s, idx, 0, next);
          const float d = next - old;
          return d * d;
        };
      },
      sumsq);
}

[[nodiscard]] auto
build_reference(Dense& dense,
                Field& u,
                Field& u_next,
                carbon::ReductionResult<float>& ref_sum) -> std::pair<carbon::Kernel,
                                                                      carbon::Kernel> {
  auto compute = carbon::Kernel::compute(
      dense,
      [&u, &u_next](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(u);
        const auto out = loader.write(u_next);
        return [in, out] CARBON_HOST_DEVICE(Span s, Idx idx) {
          const float xp = in.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0).value;
          const float xm = in.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0).value;
          const float yp = in.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0).value;
          const float ym = in.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0).value;
          out.write(s, idx, 0, 0.25F * (xp + xm + yp + ym));
        };
      },
      "ref_compute");

  auto reduce = dense.make_reduce_kernel<float>(
      [&u, &u_next](carbon::KernelLoader& loader) {
        const auto old_p = loader.read(u);
        const auto new_p = loader.read(u_next);
        return [old_p, new_p] CARBON_HOST_DEVICE(Span s, Idx idx) -> float {
          const float d = new_p.read(s, idx, 0) - old_p.read(s, idx, 0);
          return d * d;
        };
      },
      ref_sum);
  return {std::move(compute), std::move(reduce)};
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_reduce_fused_stencil: no CUDA devices");
    return EXIT_FAILURE;
  }

  carbon::System sys(carbon::get_all_devices());
  Dense dense(sys, {kWidth, kHeight, kDepth}, make_stencil());

  Field u = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u");
  Field u_next = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_next");
  Field u_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_ref");
  Field u_next_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_next_ref");

  u.fill_owned_host([](carbon::int64_3d g, int) {
    return static_cast<float>(g.x + 13.0F * g.y + 0.01F * g.z);
  });
  u_next.fill_owned_host([](carbon::int64_3d, int) {
    return 0.0F;
  });
  u_ref.fill_owned_host([&u](carbon::int64_3d g, int c) {
    (void)c;
    return static_cast<float>(g.x + 13.0F * g.y + 0.01F * g.z);
  });
  u_next_ref.fill_owned_host([](carbon::int64_3d, int) {
    return 0.0F;
  });

  carbon::Stream st = sys.stream();
  try {
    carbon::Transfer<float>(sys, u.cpu_buffers(), u.gpu_buffers())(st);
    carbon::Transfer<float>(sys, u_next.cpu_buffers(), u_next.gpu_buffers())(st);
    carbon::Transfer<float>(sys, u_ref.cpu_buffers(), u_ref.gpu_buffers())(st);
    carbon::Transfer<float>(sys, u_next_ref.cpu_buffers(), u_next_ref.gpu_buffers())(st);
    st.wait();
  } catch (const std::exception& ex) {
    CARBON_LOG_ERROR("test_reduce_fused_stencil: upload failed: {}", ex.what());
    return EXIT_FAILURE;
  }

  carbon::ReductionResult<float> fused_sum(sys);
  auto fused = build_fused(dense, u, u_next, fused_sum);

  carbon::Skeleton sk(sys);
  sk.add(std::move(fused));
  sk.compile();

  carbon::ReductionResult<float> ref_sum(sys);
  auto [ref_compute, ref_reduce] = build_reference(dense, u_ref, u_next_ref, ref_sum);

  try {
    sk.run();
    fused_sum.sync(st);
    st.wait();

    ref_compute.launch(st);
    ref_reduce.launch(st);
    ref_sum.sync(st);
    st.wait();

    carbon::Transfer<float>(sys, u_next.gpu_buffers(), u_next.cpu_buffers())(st);
    carbon::Transfer<float>(sys, u_next_ref.gpu_buffers(), u_next_ref.cpu_buffers())(st);
    st.wait();
  } catch (const std::exception& ex) {
    CARBON_LOG_ERROR("test_reduce_fused_stencil: launch failed: {}", ex.what());
    return EXIT_FAILURE;
  }

  const double ref_val = static_cast<double>(ref_sum.host_value());
  const double fused_val = static_cast<double>(fused_sum.host_value());
  if (std::fabs(fused_val - ref_val) > 1.0e-4 * std::fmax(1.0, ref_val)) {
    CARBON_LOG_ERROR("test_reduce_fused_stencil: sum fused={} ref={}", fused_val, ref_val);
    return EXIT_FAILURE;
  }

  if (!fields_match(dense, u_next, u_next_ref)) {
    CARBON_LOG_ERROR(
        "test_reduce_fused_stencil: u_next field mismatch vs compute+reduce reference");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
