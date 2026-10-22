// Skeleton anchor stream + start/end fork-join.
//
// Two things are locked here.
//
// A. Wiring. A Skeleton's stream 0 IS System::default_stream(), and two Skeletons
//    built on the same System therefore share it. That sharing is the whole basis of
//    cross-skeleton ordering: stream-0 work from both skeletons lands in one FIFO.
//
// B. Correctness of the fork-join. Under OverlapMode::Standard the OCC rewrite moves
//    the halo onto a second stream, so run() actually exercises the anchor fork
//    (secondary streams wait on a start event recorded on the anchor) and the anchor
//    join (the anchor waits on every secondary stream's tail). We run an alternating
//    ping-pong of two such skeletons with NO host drain between iterations and require
//    the result to match, bit for bit, a serialised single-stream reference that issues
//    the equivalent halo+step sequence.
//
// Before the anchor existed, each Skeleton sat on its own private sys.stream() and
// nothing ordered sk_ab against sk_ba; on multiple GPUs iteration N+1's halo peer-copy
// could read a field iteration N was still writing. This test is the regression guard
// for that (it is only a *race* on >= 2 GPUs, but the equality check also guards
// against the fork-join events being mis-wired on any GPU count).

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/neighbour_value.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include "test_domain.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t kWidth = 8;
constexpr int64_t kHeight = 6;
constexpr int64_t kDepth = 12;
constexpr int kIters = 6;

[[nodiscard]] auto
build_jacobi(carbon::Dense& dense,
             carbon::Dense::Field<float,
                                  1>& fin,
             carbon::Dense::Field<float,
                                  1>& fout,
             std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      dense,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          const carbon::NeighbourValue<float> xp = in.neighbour(s, idx, {1, 0, 0}, 0);
          const carbon::NeighbourValue<float> xm = in.neighbour(s, idx, {-1, 0, 0}, 0);
          const carbon::NeighbourValue<float> yp = in.neighbour(s, idx, {0, 1, 0}, 0);
          const carbon::NeighbourValue<float> ym = in.neighbour(s, idx, {0, -1, 0}, 0);
          const carbon::NeighbourValue<float> zp = in.neighbour(s, idx, {0, 0, 1}, 0);
          const carbon::NeighbourValue<float> zm = in.neighbour(s, idx, {0, 0, -1}, 0);
          const float self = in.read(s, idx, 0);
          const bool all_valid =
              xp.valid && xm.valid && yp.valid && ym.valid && zp.valid && zm.valid;
          if (!all_valid) {
            out.write(s, idx, 0, self);
            return;
          }
          const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
          out.write(s, idx, 0, sum * (1.0F / 6.0F));
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

[[nodiscard]] auto
build_reduce(carbon::Dense& dense,
             carbon::Dense::Field<float,
                                  1>& field,
             carbon::ReductionResult<float>& dst) -> carbon::Kernel {
  return dense.make_reduce_kernel<float>(
      [&field](carbon::KernelLoader& loader) {
        const auto in = loader.read(field);
        return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> float {
          return in.read(s, idx, 0);
        };
      },
      dst, carbon::reduce::Sum<float>{});
}

// Seed owned cells with a z-varying pattern. A z-invariant seed would make the
// cross-device halo a no-op numerically (neighbouring slabs identical), which is
// exactly the case that hid the original race — so vary along z on purpose.
[[nodiscard]] auto
seed_and_upload(carbon::System& sys,
                const carbon::Stream& stream,
                carbon::Dense::Field<float,
                                     1>& field,
                bool ramp) -> bool {
  field.fill_owned_host([ramp](carbon::int64_3d gc, int) -> float {
    if (!ramp) {
      return 0.0F;
    }
    return 1.0F + static_cast<float>(gc.z) + 0.25F * static_cast<float>(gc.x) +
           0.5F * static_cast<float>(gc.y);
  });
  try {
    carbon::Transfer<float>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_anchor: upload failed: {}", e.what());
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_skeleton_anchor: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const int64_t depth = carbon::test::dense_slab_z(kDepth, n_gpus, stencil.radius());
  carbon::Dense dense(sys, {kWidth, kHeight, depth}, stencil);

  // ----- A. Wiring: stream 0 is the System's shared anchor stream -----
  //
  // get_uid() is the address of the shared per-GPU stream vector, so equal uids mean the
  // handles are shallow copies of one stream group — the portable spelling of "same
  // streams" (StreamGroup::native_at is CUDA-only).
  {
    const carbon::Stream def = sys.default_stream();
    const carbon::Skeleton probe_a(sys);
    const carbon::Skeleton probe_b(sys);
    if (probe_a.stream().get_uid() != def.get_uid()) {
      CARBON_LOG_ERROR("test_skeleton_anchor: Skeleton stream 0 is not default_stream()");
      return EXIT_FAILURE;
    }
    if (probe_a.stream().get_uid() != probe_b.stream().get_uid()) {
      CARBON_LOG_ERROR("test_skeleton_anchor: two Skeletons do not share stream 0");
      return EXIT_FAILURE;
    }
  }

  // ----- B1. Reference: serialised single-stream halo+step loop -----
  carbon::Stream s_ref = sys.stream();
  auto u_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_ref");
  auto v_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_ref");
  if (!seed_and_upload(sys, s_ref, u_ref, /*ramp=*/true) ||
      !seed_and_upload(sys, s_ref, v_ref, /*ramp=*/false)) {
    return EXIT_FAILURE;
  }

  carbon::ReductionResult<float> sum_ref(sys);
  auto halo_u_ref = u_ref.make_halo_kernel("ref_halo_u");
  auto halo_v_ref = v_ref.make_halo_kernel("ref_halo_v");
  auto step_uv_ref = build_jacobi(dense, u_ref, v_ref, "ref_step_uv");
  auto step_vu_ref = build_jacobi(dense, v_ref, u_ref, "ref_step_vu");
  for (int it = 0; it < kIters; ++it) {
    const carbon::Kernel& halo = (it % 2 == 0) ? halo_u_ref : halo_v_ref;
    const carbon::Kernel& step = (it % 2 == 0) ? step_uv_ref : step_vu_ref;
    halo.launch(s_ref);
    step.launch(s_ref);
  }
  carbon::Dense::Field<float, 1>& final_ref = (kIters % 2 == 0) ? v_ref : u_ref;
  auto reduce_ref = build_reduce(dense, final_ref, sum_ref);
  reduce_ref.launch(s_ref);
  sum_ref.sync(s_ref);
  s_ref.wait();
  const float ref_sum = sum_ref.host_value();

  // ----- B2. Anchored ping-pong: two OCC skeletons, NO drain between iterations -----
  carbon::Stream s_up = sys.stream();
  auto u_sk = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_sk");
  auto v_sk = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_sk");
  if (!seed_and_upload(sys, s_up, u_sk, /*ramp=*/true) ||
      !seed_and_upload(sys, s_up, v_sk, /*ramp=*/false)) {
    return EXIT_FAILURE;
  }

  // OverlapMode::Standard so the OCC rewrite puts the auto-inserted halo on its own
  // stream — that is what makes the anchor fork-join load-bearing here.
  const carbon::Skeleton::Options occ(carbon::OverlapMode::Standard);
  carbon::Skeleton sk_ab(sys, occ);
  sk_ab.add(build_jacobi(dense, u_sk, v_sk, "sk_step_uv"));
  carbon::Skeleton sk_ba(sys, occ);
  sk_ba.add(build_jacobi(dense, v_sk, u_sk, "sk_step_vu"));
  try {
    sk_ab.compile();
    sk_ba.compile();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_anchor: compile threw: {}", e.what());
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_skeleton_anchor: sk_ab num_streams={} sk_ba num_streams={} ({} GPUs)",
                  sk_ab.num_streams(), sk_ba.num_streams(), n_gpus);

  try {
    for (int it = 0; it < kIters; ++it) {
      carbon::Skeleton& sk = (it % 2 == 0) ? sk_ab : sk_ba;
      sk.run();
    }
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_anchor: run threw: {}", e.what());
    return EXIT_FAILURE;
  }
  sk_ab.wait();
  sk_ba.wait();

  carbon::ReductionResult<float> sum_sk(sys);
  carbon::Dense::Field<float, 1>& final_sk = (kIters % 2 == 0) ? v_sk : u_sk;
  carbon::Skeleton reduce_sk(sys);
  reduce_sk.add(build_reduce(dense, final_sk, sum_sk));
  try {
    reduce_sk.compile();
    reduce_sk.run();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_anchor: reduce compile/run threw: {}", e.what());
    return EXIT_FAILURE;
  }
  sum_sk.sync(reduce_sk.stream());
  reduce_sk.wait();
  const float sk_sum = sum_sk.host_value();

  if (sk_sum != ref_sum) {
    CARBON_LOG_ERROR(
        "test_skeleton_anchor: anchored ping-pong sum = {}, serialised reference = {} "
        "(mismatch — fork-join ordering is wrong)",
        sk_sum, ref_sum);
    return EXIT_FAILURE;
  }

  CARBON_LOG_INFO(
      "test_skeleton_anchor: OK — {} undrained alternating OCC skeleton iterations on {} GPU(s) "
      "matched the serialised reference ({})",
      kIters, n_gpus, sk_sum);
  return EXIT_SUCCESS;
}
