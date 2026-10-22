// Skeleton + OverlappedStencil interaction test.
//
// Locks the contract that an OverlappedStencil's compute_stream can be
// adopted by a Skeleton (constructor overload that takes a user-supplied
// Stream) and downstream Skeleton work sees the OverlappedStencil's
// final iteration output correctly.
//
// Pipeline:
//   1. Build a halo + a Jacobi-style stencil kernel.
//   2. Run N OverlappedStencil::step() calls; the orchestrator handles
//      the halo/compute overlap on its two streams.
//   3. Build a Skeleton bound to overlap.compute_stream() and add a
//      reduction over the overlapped field. The Skeleton's compile()
//      pass sees no halo/compute pair (only the reduce kernel) so it
//      runs Standard view.
//   4. Run the Skeleton; sync; assert the reduction matches a hand-
//      rolled reference loop that runs the same N iterations + reduce
//      directly on a single stream.
//
// The interaction value-add: the Skeleton is bound to a stream that
// already has pending OverlappedStencil work; the stream's FIFO order
// guarantees the reduction sees the final iteration's writes before
// it issues its own reads, no extra fence required from user code.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/halo.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/neighbour_value.h>
#include <carbon/overlapped_stencil.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include "test_domain.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace {

constexpr int64_t kWidth = 8;
constexpr int64_t kHeight = 6;
constexpr int64_t kDepth = 12;
constexpr int kIters = 4;

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

[[nodiscard]] auto
seed_and_upload(carbon::System& sys,
                carbon::Dense& dense,
                const carbon::Stream& stream,
                carbon::Dense::Field<float,
                                     1>& field,
                float value) -> bool {
  field.fill_owned_host([value](carbon::int64_3d, int) {
    return value;
  });
  try {
    carbon::Transfer<float>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_overlapped: upload failed: {}", e.what());
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_skeleton_overlapped: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  carbon::System sys(carbon::get_all_devices());
  carbon::Stencil stencil(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const int64_t depth =
      carbon::test::dense_slab_z(kDepth, sys.gpu_group().size(), stencil.radius());
  carbon::Dense dense(sys, {kWidth, kHeight, depth}, stencil);

  // ----- Path A: reference single-stream loop -----
  carbon::Stream s_ref = sys.stream();
  auto u_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_ref");
  auto v_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_ref");
  if (!seed_and_upload(sys, dense, s_ref, u_ref, /*value=*/2.5F) ||
      !seed_and_upload(sys, dense, s_ref, v_ref, /*value=*/0.0F)) {
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
  // After kIters, the latest write target alternates: iter 0 writes v, iter 1
  // writes u, ... so the final write target is (kIters % 2 == 0) ? v : u.
  carbon::Dense::Field<float, 1>& final_ref = (kIters % 2 == 0) ? v_ref : u_ref;
  auto reduce_ref = build_reduce(dense, final_ref, sum_ref);
  reduce_ref.launch(s_ref);
  sum_ref.sync(s_ref);
  s_ref.wait();
  const float ref_sum = sum_ref.host_value();

  // ----- Path B: OverlappedStencil + Skeleton chained on its compute_stream -----
  auto u_ovr = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_ovr");
  auto v_ovr = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "v_ovr");
  // Use a temporary upload stream — once the overlap takes over, every write
  // to u_ovr / v_ovr happens on overlap.compute_stream().
  carbon::Stream s_upload = sys.stream();
  if (!seed_and_upload(sys, dense, s_upload, u_ovr, /*value=*/2.5F) ||
      !seed_and_upload(sys, dense, s_upload, v_ovr, /*value=*/0.0F)) {
    return EXIT_FAILURE;
  }

  carbon::OverlappedStencil overlap(sys);
  auto halo_u_ovr = u_ovr.make_halo_kernel("ovr_halo_u");
  auto halo_v_ovr = v_ovr.make_halo_kernel("ovr_halo_v");
  auto step_uv_ovr = build_jacobi(dense, u_ovr, v_ovr, "ovr_step_uv");
  auto step_vu_ovr = build_jacobi(dense, v_ovr, u_ovr, "ovr_step_vu");

  for (int it = 0; it < kIters; ++it) {
    const carbon::Kernel& halo = (it % 2 == 0) ? halo_u_ovr : halo_v_ovr;
    const carbon::Kernel& step = (it % 2 == 0) ? step_uv_ovr : step_vu_ovr;
    overlap.step(halo, step);
  }

  // Skeleton bound to overlap.compute_stream(). The compute_stream is the
  // longer-lived queue (Internal + Boundary launch), so chaining a downstream
  // reduction onto it gives FIFO ordering after the last Boundary launch
  // without any explicit wait_for/event wiring on the user's part.
  carbon::ReductionResult<float> sum_ovr(sys);
  carbon::Dense::Field<float, 1>& final_ovr = (kIters % 2 == 0) ? v_ovr : u_ovr;
  carbon::Skeleton sk(sys, overlap.compute_stream());
  sk.add(build_reduce(dense, final_ovr, sum_ovr));

  try {
    sk.compile();
    sk.run();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_overlapped: skeleton compile/run threw: {}", e.what());
    return EXIT_FAILURE;
  }

  // Sync both — the reduction is on overlap.compute_stream(), but the halo
  // stream has its own pending work (the last halo launch).
  sum_ovr.sync(overlap.compute_stream());
  overlap.wait();
  sk.wait();

  const float ovr_sum = sum_ovr.host_value();
  if (ovr_sum != ref_sum) {
    CARBON_LOG_ERROR(
        "test_skeleton_overlapped: skeleton+overlap reduction = {}, reference = {} (mismatch)",
        ovr_sum, ref_sum);
    return EXIT_FAILURE;
  }

  CARBON_LOG_INFO(
      "test_skeleton_overlapped: OK — {} OverlappedStencil iters then a downstream Skeleton "
      "reduction on the same compute_stream produced the same total ({}) as a single-stream "
      "reference loop running the equivalent halo+step+reduce sequence",
      kIters, ovr_sum);
  return EXIT_SUCCESS;
}
