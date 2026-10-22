// Skeleton + Reduce integration test.
//
// Locks the contract that mixing carbon::Kernel::compute and
// dense.make_reduce_kernel inside a single Skeleton produces the same
// per-iteration reduction values as a hand-rolled loop that calls
// each kernel directly. Catches token-set / scheduling regressions in
// the interaction between compute and reduce kernels (e.g. a future
// change that makes ReduceKernelImpl::tokens() report an unexpected
// pattern, or a Skeleton phase that drops reduce-kernel record events).
//
// Pipeline (per iteration):
//   1. compute: u = u + 1.0  (Map-write on u)
//   2. reduce:  sum = sum_over_cells(u)  (Read on u)
//
// After N iterations the Skeleton's ReductionResult.host_value()
// must equal the reference loop's value, and both must equal the
// closed-form sum (initial_sum + N * cell_count) since each iter
// adds 1.0 to every owned cell.

#include <carbon/carbon.h>
#include <carbon/dense.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace {

constexpr int64_t kWidth = 8;
constexpr int64_t kHeight = 6;
constexpr int64_t kDepth = 12;
constexpr int kIters = 5;
constexpr float kInitial = 1.0F;

[[nodiscard]] auto
make_pipeline(carbon::Dense& dense,
              carbon::Dense::Field<float,
                                   1>& field,
              carbon::ReductionResult<float>& result) -> std::pair<carbon::Kernel,
                                                                   carbon::Kernel> {
  auto bump = carbon::Kernel::compute(
      dense,
      [&field](carbon::KernelLoader& loader) {
        const auto rw = loader.read_write(field);
        return [rw] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) {
          rw.write(s, idx, 0, rw.read(s, idx, 0) + 1.0F);
        };
      },
      "bump_u",
      /*shared_mem=*/0U);

  auto reduce = dense.make_reduce_kernel<float>(
      [&field](carbon::KernelLoader& loader) {
        const auto in = loader.read(field);
        return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> float {
          return in.read(s, idx, 0);
        };
      },
      result, carbon::reduce::Sum<float>{});

  return {std::move(bump), std::move(reduce)};
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_skeleton_reduce: no CUDA devices visible");
    return EXIT_FAILURE;
  }

  carbon::System sys(carbon::get_all_devices());
  carbon::Stencil stencil({}, /*remove_center=*/false);
  carbon::Dense dense(sys, {kWidth, kHeight, kDepth}, stencil);
  const int64_t cell_count = kWidth * kHeight * kDepth;

  // Reference path: hand-rolled loop on its own stream + dedicated kernels.
  carbon::Stream s_ref = sys.stream();
  auto u_ref = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_ref");
  carbon::ReductionResult<float> sum_ref(sys);

  u_ref.fill_owned_host([](carbon::int64_3d, int) {
    return kInitial;
  });
  try {
    carbon::Transfer<float>(sys, u_ref.cpu_buffers(), u_ref.gpu_buffers())(s_ref);
    s_ref.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_reduce: ref upload failed: {}", e.what());
    return EXIT_FAILURE;
  }

  auto [bump_ref, reduce_ref] = make_pipeline(dense, u_ref, sum_ref);
  float ref_last = 0.0F;
  for (int it = 0; it < kIters; ++it) {
    try {
      bump_ref.launch(s_ref);
      reduce_ref.launch(s_ref);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton_reduce: ref iter {} threw: {}", it, e.what());
      return EXIT_FAILURE;
    }
    sum_ref.sync(s_ref);
    s_ref.wait();
    ref_last = sum_ref.host_value();
  }

  // Skeleton path.
  carbon::Stream s_sk = sys.stream();
  auto u_sk = dense.newField<float, 1>(carbon::MemorySide::HostDevice, "u_sk");
  carbon::ReductionResult<float> sum_sk(sys);

  u_sk.fill_owned_host([](carbon::int64_3d, int) {
    return kInitial;
  });
  try {
    carbon::Transfer<float>(sys, u_sk.cpu_buffers(), u_sk.gpu_buffers())(s_sk);
    s_sk.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_reduce: skeleton upload failed: {}", e.what());
    return EXIT_FAILURE;
  }

  carbon::Skeleton sk(sys, s_sk);
  auto [bump_sk, reduce_sk] = make_pipeline(dense, u_sk, sum_sk);
  sk.add(std::move(bump_sk));
  sk.add(std::move(reduce_sk));

  try {
    sk.compile();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_skeleton_reduce: skeleton compile threw: {}", e.what());
    return EXIT_FAILURE;
  }

  // The two kernels conflict on u (R after W) so they must serialise.
  // With 2 kernels, compile() should infer exactly one edge 0 -> 1 and
  // report num_streams() == 1 (greedy keeps the chain on one stream).
  if (sk.edges().size() != 2U || sk.edges()[0] != std::vector<size_t>{1U} ||
      !sk.edges()[1].empty()) {
    CARBON_LOG_ERROR(
        "test_skeleton_reduce: unexpected DAG edges (expected {{1}, {{}}}); skeleton may have "
        "missed the bump→reduce token conflict");
    return EXIT_FAILURE;
  }
  if (sk.num_streams() != 1U) {
    CARBON_LOG_ERROR("test_skeleton_reduce: expected num_streams()==1 (single chain), got {}",
                     sk.num_streams());
    return EXIT_FAILURE;
  }

  float sk_last = 0.0F;
  for (int it = 0; it < kIters; ++it) {
    try {
      sk.run();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_skeleton_reduce: skeleton iter {} threw: {}", it, e.what());
      return EXIT_FAILURE;
    }
    sum_sk.sync(s_sk);
    s_sk.wait();
    sk_last = sum_sk.host_value();
  }

  if (sk_last != ref_last) {
    CARBON_LOG_ERROR("test_skeleton_reduce: final sum mismatch: skeleton={}, ref={}", sk_last,
                     ref_last);
    return EXIT_FAILURE;
  }

  // Closed-form check (catches the case where ref + skeleton both
  // happen to be wrong in the same way).
  const float expected = (kInitial + static_cast<float>(kIters)) * static_cast<float>(cell_count);
  if (sk_last != expected) {
    CARBON_LOG_ERROR("test_skeleton_reduce: closed-form mismatch: got {}, want {}", sk_last,
                     expected);
    return EXIT_FAILURE;
  }

  CARBON_LOG_INFO(
      "test_skeleton_reduce: OK — {} iter(s) of bump+reduce on {}x{}x{} ({} cells); skeleton sum "
      "= reference sum = closed-form sum = {}",
      kIters, kWidth, kHeight, kDepth, cell_count, sk_last);
  return EXIT_SUCCESS;
}
