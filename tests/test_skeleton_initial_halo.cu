// Regression: Skeleton::compile must auto-insert a halo before the first
// stencil-read of a freshly-uploaded multi-GPU field. fill_owned_host /
// Transfer<T>(cpu, gpu) only initialise the OWNED cells; the halo regions
// on device hold uninitialised bytes until the first cross-slab exchange.
// Without an initial auto-halo, the first stencil read across the slab
// boundary returns garbage, which is exactly what bit the LBM lid-driven
// cavity benchmark on 2 GPUs (rho dropped to 5/6 = 0.83333 at slab-boundary
// z planes — the missing 1/6 was the sum of D3Q19 weights of the 5
// directions whose pull-source crossed the slab boundary).
//
// The fix: every field appearing in any kernel token starts dirty, so the
// first stencil read triggers an auto-halo regardless of whether an
// in-skeleton kernel has yet written the field. A pre-existing manual halo
// in the pipeline still dedups the redundant initial halo (test_skeleton
// subtest B' covers that contract).
//
// This test exercises the multi-GPU path directly: a 1-component field with
// owned cells set to a constant, a single Skeleton kernel that stencil-reads
// from the +z neighbour and writes to a sink field, and an assertion that
// the sink at the slab-boundary z plane carries the constant — i.e. the
// across-slab read found valid data, not the uninitialised halo.

#include <carbon/carbon.h>
#include <carbon/dense.h>
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

constexpr float kSeed = 1.5F;

}  // namespace

auto
main() -> int {
  carbon::init();
  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();
  if (n_gpus < 2U) {
    CARBON_LOG_INFO("test_skeleton_initial_halo: SKIP — needs >= 2 GPUs (have {})", n_gpus);
    return 0;
  }

  const carbon::int64_3d global{8, 8, 16};
  carbon::Stencil stencil(std::vector<carbon::int32_3d>{{0, 0, -1}, {0, 0, 1}},
                          /*remove_center=*/true);
  carbon::Dense dense(sys, global, stencil);
  auto src = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "src");
  auto dst = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "dst");

  src.fill_owned_host([&](carbon::int64_3d, int) {
    return kSeed;
  });
  dst.fill_owned_host([&](carbon::int64_3d, int) {
    return 0.0F;
  });

  carbon::Stream up = sys.stream();
  carbon::Transfer<float>(sys, src.cpu_buffers(), src.gpu_buffers())(up);
  carbon::Transfer<float>(sys, dst.cpu_buffers(), dst.gpu_buffers())(up);
  up.wait();

  // Build a kernel that reads src[idx + (0,0,+1)] (i.e. pulls from the +z
  // neighbour, which crosses the slab boundary for the last owned z plane on
  // every GPU except the last). Use loader.read_stencil to opt into the
  // halo factory snapshot — the contract that hands the Skeleton enough
  // metadata to insert an auto-halo.
  carbon::Skeleton sk(sys);
  sk.add(carbon::Kernel::compute(
      dense,
      [&src, &dst](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(src);
        const auto out = loader.write(dst);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const auto loc = s.global_location(idx);
          // Skip the upper-most z plane (no +z neighbour anywhere in the
          // grid; would index into the open-mode out-of-domain halo).
          if (loc.z + 1 >= s.global_extent().z) {
            out.write(s, idx, 0, kSeed);
            return;
          }
          const auto c = idx.cell();
          const carbon::Dense::Idx ngh = s.make_idx(c.x, c.y, c.z + 1);
          out.write(s, idx, 0, in.read(s, ngh, 0));
        };
      },
      "stencil_pull_zp1",
      /*shared_mem=*/0U));

  sk.compile();

  // The fix in <carbon/skeleton.h> seeds the dirty set with every token's
  // field at compile entry. Without it, src is "clean" (no in-skeleton
  // write) and the auto-halo is silently skipped. With it, effective_kernels
  // is exactly [auto_halo_src, stencil_pull_zp1].
  if (sk.effective_kernels().size() != 2U) {
    CARBON_LOG_ERROR(
        "test_skeleton_initial_halo: expected effective_kernels()==2 (auto-halo + stencil), got "
        "{} — Skeleton::compile is not seeding the dirty set with token fields",
        sk.effective_kernels().size());
    return EXIT_FAILURE;
  }
  if (sk.effective_kernels()[0].kind() != carbon::KernelKind::Halo) {
    CARBON_LOG_ERROR("test_skeleton_initial_halo: effective_kernels()[0] is not a Halo (kind={})",
                     static_cast<int>(sk.effective_kernels()[0].kind()));
    return EXIT_FAILURE;
  }

  sk.run();
  sk.wait();

  carbon::Stream dl = sys.stream();
  carbon::Transfer<float>(sys, dst.gpu_buffers(), dst.cpu_buffers())(dl);
  dl.wait();

  // Inspect dst at the LAST owned z plane on each non-top slab — that's the
  // plane whose +z neighbour lives in the upper halo (and on a peer GPU).
  // Without the auto-halo, those reads would return uninitialised garbage;
  // with it, they should return kSeed for every (x, y) in the interior of
  // the plane.
  const auto& spans = dense.spans();
  int errors = 0;
  for (size_t g = 0; g < n_gpus; ++g) {
    const auto& span = spans[g];
    const auto haloed = span.haloed_extent();
    const int64_t pitch = span.xy_slab_elements();
    const int64_t halo_z = span.halo_z();
    const int64_t lim_z = span.logic_extent().z;
    const int64_t last_owned = halo_z + lim_z - 1;
    const int64_t global_z = span.z_offset() + last_owned;
    if (global_z + 1 >= global.z) {
      continue;  // top slab's last owned has no +z neighbour
    }
    const float* h = dst.cpu_buffers()[g].data();
    for (int64_t y = 0; y < haloed.y; ++y) {
      for (int64_t x = 0; x < haloed.x; ++x) {
        const float got = h[last_owned * pitch + y * haloed.x + x];
        if (got != kSeed) {
          if (errors < 4) {
            CARBON_LOG_ERROR(
                "test_skeleton_initial_halo: GPU {} buf_z={} (global z={}) (x={}, y={}) got "
                "{} want {}",
                g, last_owned, global_z, x, y, got, kSeed);
          }
          ++errors;
        }
      }
    }
  }
  if (errors > 0) {
    CARBON_LOG_ERROR(
        "test_skeleton_initial_halo: FAIL — {} cross-slab stencil reads returned "
        "uninitialised halo data",
        errors);
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_skeleton_initial_halo: OK ({} GPUs)", n_gpus);
  return 0;
}
