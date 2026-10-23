// Skeleton auto-inserts scalar readback after Deferred reduce (compile + DOT).
// Also checks copy-only schedules skip anchor fork/join and lagged wait_value.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/transfer.h>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <string>

namespace {

[[nodiscard]] auto
make_sum_reduce(carbon::Dense& grid,
                carbon::Dense::Field<float,
                                     1>& f,
                carbon::ReductionResult<float>& dst,
                carbon::reduce::ReduceOptions opts) -> carbon::Kernel {
  return grid.make_reduce_kernel<float>(
      [&f](carbon::KernelLoader& loader) {
        const auto in = loader(f);
        return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> float {
          return in.read(s, idx, 0);
        };
      },
      dst, carbon::reduce::Sum<float>{}, 0U, opts);
}

}  // namespace

int
main() {
  try {
    carbon::init();
    std::vector<carbon::detail::Gpu> gpus = carbon::get_all_devices();
    if (gpus.empty()) {
      CARBON_LOG_INFO("test_skeleton_scalar_pattern: no GPU, skip");
      return EXIT_SUCCESS;
    }
    carbon::System sys(std::vector<carbon::detail::Gpu>{std::move(gpus.front())});
    const carbon::int64_3d global{4, 4, 4};
    carbon::Dense grid(sys, global, carbon::Stencil{});
    carbon::Dense::Field<float, 1> f = grid.newField<float, 1>(carbon::MemorySide::HostDevice, "f");
    f.fill_owned_host([](carbon::int64_3d, int) {
      return 1.0F;
    });
    carbon::Stream up = sys.stream();
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(up);
    up.wait();

    carbon::ReductionResult<float> dst(sys, carbon::reduce::ReductionResultOptions{.slots = 2U});
    carbon::reduce::ReduceOptions opts{};
    opts.host_copy = carbon::reduce::ReduceHostCopy::Deferred;

    carbon::Kernel reduce_k = make_sum_reduce(grid, f, dst, opts);

    carbon::Skeleton sk(sys);
    sk.add(std::move(reduce_k));
    sk.compile();

    const auto& eff = sk.effective_kernels();
    if (eff.size() != 2U) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern: expected 2 effective kernels, got {}",
                       eff.size());
      return EXIT_FAILURE;
    }
    if (eff[0].kind() != carbon::KernelKind::Reduction ||
        eff[1].kind() != carbon::KernelKind::ScalarReadback) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern: unexpected kernel kinds");
      return EXIT_FAILURE;
    }

    if (sk.num_streams() != 2U) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern: expected 2 streams (compute + copy), got {}",
                       sk.num_streams());
      return EXIT_FAILURE;
    }
    if (sk.uses_anchor_fork_join()) {
      CARBON_LOG_ERROR(
          "test_skeleton_scalar_pattern: copy-only secondary stream must not use anchor fork/join");
      return EXIT_FAILURE;
    }

    const std::string dot = sk.export_dot();
    if (dot.find("scalar_readback") == std::string::npos) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern: DOT missing scalar_readback node");
      return EXIT_FAILURE;
    }
    if (dot.find("copy stream") == std::string::npos) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern: DOT missing detached copy stream label");
      return EXIT_FAILURE;
    }

    sk.run();
    sk.wait();
    const float v = dst.wait_value();
    const float want64 = 64.0F;
    if (std::fabs(v - want64) > 1.0e-3F) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern: deferred run sum {} (expected ~64)", v);
      return EXIT_FAILURE;
    }

    // Ping-pong lag: two back-to-back runs with different field values, one slot ring.
    f.fill_owned_host([](carbon::int64_3d, int) {
      return 2.0F;
    });
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(up);
    up.wait();
    sk.run();
    f.fill_owned_host([](carbon::int64_3d, int) {
      return 3.0F;
    });
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(up);
    up.wait();
    sk.run();
    sk.wait();

    const float latest = dst.wait_value(carbon::DataView::Standard, 0U);
    const float lagged = dst.wait_value(carbon::DataView::Standard, 1U);
    const float want128 = 128.0F;
    const float want192 = 192.0F;
    if (std::fabs(lagged - want128) > 1.0e-2F) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern: lag-1 sum {} expected ~{}", lagged, want128);
      return EXIT_FAILURE;
    }
    if (std::fabs(latest - want192) > 1.0e-2F) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern: lag-0 sum {} expected ~{}", latest, want192);
      return EXIT_FAILURE;
    }

    // Alternating skeleton ping-pong on the shared anchor (fresh result — prior launches consumed).
    carbon::ReductionResult<float> dst_ping(sys,
                                            carbon::reduce::ReductionResultOptions{.slots = 2U});
    carbon::Skeleton sk_a(sys);
    carbon::Skeleton sk_b(sys);
    sk_a.add(make_sum_reduce(grid, f, dst_ping, opts));
    sk_b.add(make_sum_reduce(grid, f, dst_ping, opts));
    sk_a.compile();
    sk_b.compile();
    if (sk_a.uses_anchor_fork_join() || sk_b.uses_anchor_fork_join()) {
      CARBON_LOG_ERROR(
          "test_skeleton_scalar_pattern: ping-pong skeletons must not use anchor events");
      return EXIT_FAILURE;
    }
    f.fill_owned_host([](carbon::int64_3d, int) {
      return 4.0F;
    });
    carbon::Transfer<float>(sys, f.cpu_buffers(), f.gpu_buffers())(up);
    up.wait();
    for (int i = 0; i < 4; ++i) {
      if ((i & 1) == 0) {
        sk_a.run();
      } else {
        sk_b.run();
      }
      // Lagged host read (like pipelined Jacobi): consume iteration i-1 while i runs.
      if (i >= 1) {
        (void)dst_ping.wait_value(carbon::DataView::Standard, 1U);
      }
    }
    sk_a.wait();
    sk_b.wait();
    const float ping = dst_ping.wait_value();
    if (std::fabs(ping - 256.0F) > 1.0e-2F) {
      CARBON_LOG_ERROR("test_skeleton_scalar_pattern: ping-pong sum {} expected ~256", ping);
      return EXIT_FAILURE;
    }

    CARBON_LOG_INFO("test_skeleton_scalar_pattern: PASS");
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    CARBON_LOG_ERROR("test_skeleton_scalar_pattern: {}", ex.what());
    return EXIT_FAILURE;
  }
}
