// Skeleton CUDA-graph option (Options::use_cuda_graph): a single-GPU,
// single-stream ping-pong pipeline replayed via a captured graph must produce
// bit-identical results to the eager launch path, and must actually engage the
// graph (uses_graph() == true) after the first eligible run.
//
// The pipeline mirrors the LBM benchmark's driver shape: two fixed-schedule
// skeletons (A->B and B->A) alternated across iterations. Each skeleton's
// schedule + captured kernel arguments are invariant, which is exactly the
// contract CUDA graph replay requires. We force a single-GPU System so the
// eligibility check (single stream / single GPU) holds deterministically on
// any machine.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/detail/backend/select.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/skeleton.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kIters = 200;

// Deterministic per-cell map: out = in * 1.001 + 0.5. No neighbour reads, so no
// halo is needed and the single-GPU schedule is a pure single-kernel step.
auto
make_step(carbon::Dense& dense,
          carbon::Dense::Field<float,
                               1>& in,
          carbon::Dense::Field<float,
                               1>& out,
          std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&in, &out](carbon::KernelLoader& loader) {
        const auto i = loader.read(in);
        const auto o = loader.write(out);
        return [i, o] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          o.write(s, idx, 0, i.read(s, idx, 0) * 1.001F + 0.5F);
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

// Run kIters alternating A<->B steps on a fresh pair of fields; return the
// host-side contents of whichever field holds the final state.
auto
run_pingpong(carbon::System& sys,
             carbon::Dense& dense,
             bool use_graph,
             bool& engaged_graph_out) -> std::vector<float> {
  auto a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "a");
  auto b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "b");
  a.fill_owned_host([&](carbon::int64_3d loc, int) {
    return static_cast<float>((loc.x + 2 * loc.y + 3 * loc.z) % 7) * 0.25F;
  });
  b.fill_owned_host([&](carbon::int64_3d, int) {
    return 0.0F;
  });

  carbon::Stream up = sys.stream();
  carbon::Transfer<float>(sys, a.cpu_buffers(), a.gpu_buffers())(up);
  carbon::Transfer<float>(sys, b.cpu_buffers(), b.gpu_buffers())(up);
  up.wait();

  const carbon::Skeleton::Options opts(carbon::OverlapMode::None, /*strict=*/false,
                                       /*max_streams=*/0U, /*dot_on_compile=*/false,
                                       /*cuda_graph=*/use_graph);
  carbon::Skeleton sk_ab(sys, opts);
  sk_ab.add(make_step(dense, a, b, "ab"));
  carbon::Skeleton sk_ba(sys, opts);
  sk_ba.add(make_step(dense, b, a, "ba"));

  engaged_graph_out = false;
  for (int it = 0; it < kIters; ++it) {
    carbon::Skeleton& sk = (it % 2 == 0) ? sk_ab : sk_ba;
    sk.run();
    sk.wait();
    if (use_graph && sk.uses_graph()) {
      engaged_graph_out = true;
    }
  }

  // After kIters (even), the last written field is `a` (it % 2 == 1 step ba
  // writes a). Read it back.
  carbon::Stream dl = sys.stream();
  carbon::Transfer<float>(sys, a.gpu_buffers(), a.cpu_buffers())(dl);
  dl.wait();

  const auto& span = dense.spans()[0];
  const int64_t pitch = span.xy_slab_elements();
  const auto haloed = span.haloed_extent();
  const int64_t nz = span.haloed_extent().z;
  std::vector<float> host(static_cast<size_t>(pitch) * static_cast<size_t>(nz));
  const float* src = a.cpu_buffers()[0].data();
  for (size_t i = 0; i < host.size(); ++i) {
    host[i] = src[i];
  }
  (void)haloed;
  return host;
}

}  // namespace

auto
main() -> int {
  carbon::init();

  auto all = carbon::get_all_devices();
  if (all.empty()) {
    CARBON_LOG_INFO("test_skeleton_cuda_graph: SKIP — no devices");
    return 0;
  }
  // Force single-GPU so the graph-eligibility check (single stream / single
  // GPU) holds regardless of how many GPUs the host exposes.
  std::vector<carbon::detail::Gpu> one;
  one.push_back(std::move(all.front()));
  carbon::System sys(std::move(one));

  const carbon::int64_3d global{16, 16, 16};
  carbon::Dense dense(sys, global);

  bool eager_engaged = false;
  const std::vector<float> eager = run_pingpong(sys, dense, /*use_graph=*/false, eager_engaged);

  bool graph_engaged = false;
  const std::vector<float> graphed = run_pingpong(sys, dense, /*use_graph=*/true, graph_engaged);

  if (eager.size() != graphed.size()) {
    CARBON_LOG_ERROR("test_skeleton_cuda_graph: size mismatch {} vs {}", eager.size(),
                     graphed.size());
    return EXIT_FAILURE;
  }

  int mismatches = 0;
  for (size_t i = 0; i < eager.size(); ++i) {
    if (eager[i] != graphed[i]) {
      if (mismatches < 4) {
        CARBON_LOG_ERROR("test_skeleton_cuda_graph: idx {} eager {} graph {}", i, eager[i],
                         graphed[i]);
      }
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    CARBON_LOG_ERROR(
        "test_skeleton_cuda_graph: FAIL — {} value mismatches (graph replay diverged "
        "from eager)",
        mismatches);
    return EXIT_FAILURE;
  }

#if CARBON_BACKEND_IS_CUDALIKE
  if (!graph_engaged) {
    CARBON_LOG_ERROR(
        "test_skeleton_cuda_graph: FAIL — CUDALIKE backend but the graph never "
        "engaged (uses_graph() stayed false on a single-GPU pipeline)");
    return EXIT_FAILURE;
  }
#else
  if (graph_engaged) {
    CARBON_LOG_ERROR("test_skeleton_cuda_graph: FAIL — non-CUDALIKE backend reported graph use");
    return EXIT_FAILURE;
  }
#endif

  CARBON_LOG_INFO("test_skeleton_cuda_graph: OK (graph_engaged={}, {} cells)", graph_engaged,
                  eager.size());
  return 0;
}
