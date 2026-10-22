// Skeleton CUDA-graph option on a MULTI-GPU intra-process pipeline
// (Options::use_cuda_graph): a single-Skeleton-stream ping-pong with a real
// cross-device halo exchange, replayed via a captured cross-device graph, must
// produce bit-identical results to the eager launch path and must actually
// engage the graph (uses_graph() == true) once eligible.
//
// This is the multi-GPU counterpart to test_skeleton_cuda_graph: it exercises
// the global-mode fork/join origin-stream capture, where the per-GPU kernel
// launches (OpenMP fan-out) and the halo's cross-GPU peer copies + event fences
// are all folded into one unified cross-device graph launched from the origin.
//
// Needs >=2 visible devices for a meaningful cross-device capture; skips (as a
// pass) otherwise so the test is a no-op on single-GPU hosts.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/detail/backend/select.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kIters = 120;

// Stencil compute step: out = 0.5*center + 0.25*(z+1) + 0.25*(z-1), with
// out-of-domain neighbours (Open edges) folding back to center. Reads the z
// neighbours, so it depends on a preceding halo exchange on multi-GPU layouts.
auto
make_stencil_step(carbon::Dense& dense,
                  carbon::Dense::Field<float,
                                       1>& in,
                  carbon::Dense::Field<float,
                                       1>& out,
                  std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&in, &out](carbon::KernelLoader& loader) {
        const auto i = loader.read_stencil(in);
        const auto o = loader.write(out);
        return [i, o] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const float c = i.read(s, idx, 0);
          const auto zp = i.neighbour(s, idx, carbon::int8_3d{0, 0, +1}, 0);
          const auto zm = i.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
          const float up = zp.valid ? zp.value : c;
          const float dn = zm.valid ? zm.value : c;
          o.write(s, idx, 0, 0.5F * c + 0.25F * up + 0.25F * dn);
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

// Download every slab's owned+halo storage into one flat host vector for
// bit-comparison between the eager and graph runs.
auto
download_all(carbon::System& sys,
             carbon::Dense& dense,
             carbon::Dense::Field<float,
                                  1>& field) -> std::vector<float> {
  carbon::Stream dl = sys.stream();
  carbon::Transfer<float>(sys, field.gpu_buffers(), field.cpu_buffers())(dl);
  dl.wait();
  std::vector<float> flat;
  for (size_t g = 0; g < dense.spans().size(); ++g) {
    const float* src = field.cpu_buffers()[g].data();
    const size_t n = field.cpu_buffers()[g].size();
    for (size_t i = 0; i < n; ++i) {
      flat.push_back(src[i]);
    }
  }
  return flat;
}

auto
run_pingpong(carbon::System& sys,
             carbon::Dense& dense,
             bool use_graph,
             bool& engaged_graph_out) -> std::vector<float> {
  auto a = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "a");
  auto b = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "b");
  a.fill_owned_host([&](carbon::int64_3d loc, int) {
    return static_cast<float>((loc.x + 2 * loc.y + 3 * loc.z) % 11) * 0.125F;
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
  sk_ab.add(a.make_halo_kernel("halo_a"));
  sk_ab.add(make_stencil_step(dense, a, b, "ab"));
  carbon::Skeleton sk_ba(sys, opts);
  sk_ba.add(b.make_halo_kernel("halo_b"));
  sk_ba.add(make_stencil_step(dense, b, a, "ba"));

  engaged_graph_out = false;
  for (int it = 0; it < kIters; ++it) {
    carbon::Skeleton& sk = (it % 2 == 0) ? sk_ab : sk_ba;
    sk.run();
    sk.wait();
    if (use_graph && sk.uses_graph()) {
      engaged_graph_out = true;
    }
  }

  // kIters even -> the final write lands in `a` (odd iteration runs sk_ba).
  return download_all(sys, dense, a);
}

}  // namespace

auto
main() -> int {
  carbon::init();

  auto all = carbon::get_all_devices();
  if (all.size() < 2U) {
    CARBON_LOG_INFO(
        "test_skeleton_cuda_graph_multi: SKIP — needs >=2 devices for a cross-device capture "
        "(have {})",
        all.size());
    return 0;
  }
  carbon::System sys(std::move(all));
  const size_t n_gpus = sys.gpu_group().size();

  // Z-only stencil so halo_z == 1 and the exchange runs along the partition axis.
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);
  const carbon::int64_3d global{8, 8, 4 * static_cast<int64_t>(n_gpus)};
  carbon::Dense dense(sys, global, std::move(stencil), carbon::BoundaryMode::PeriodicZ);

  bool eager_engaged = false;
  const std::vector<float> eager = run_pingpong(sys, dense, /*use_graph=*/false, eager_engaged);

  bool graph_engaged = false;
  const std::vector<float> graphed = run_pingpong(sys, dense, /*use_graph=*/true, graph_engaged);

  if (eager.size() != graphed.size()) {
    CARBON_LOG_ERROR("test_skeleton_cuda_graph_multi: size mismatch {} vs {}", eager.size(),
                     graphed.size());
    return EXIT_FAILURE;
  }

  int mismatches = 0;
  for (size_t i = 0; i < eager.size(); ++i) {
    if (eager[i] != graphed[i]) {
      if (mismatches < 4) {
        CARBON_LOG_ERROR("test_skeleton_cuda_graph_multi: idx {} eager {} graph {}", i, eager[i],
                         graphed[i]);
      }
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    CARBON_LOG_ERROR(
        "test_skeleton_cuda_graph_multi: FAIL — {} value mismatches (cross-device graph replay "
        "diverged from eager)",
        mismatches);
    return EXIT_FAILURE;
  }

#if CARBON_BACKEND_IS_CUDALIKE
  if (!graph_engaged) {
    CARBON_LOG_ERROR(
        "test_skeleton_cuda_graph_multi: FAIL — CUDALIKE backend with {} GPUs but the graph never "
        "engaged (uses_graph() stayed false)",
        n_gpus);
    return EXIT_FAILURE;
  }
#else
  if (graph_engaged) {
    CARBON_LOG_ERROR(
        "test_skeleton_cuda_graph_multi: FAIL — non-CUDALIKE backend reported graph use");
    return EXIT_FAILURE;
  }
#endif

  CARBON_LOG_INFO("test_skeleton_cuda_graph_multi: OK (n_gpus={}, graph_engaged={}, {} cells)",
                  n_gpus, graph_engaged, eager.size());
  return 0;
}
