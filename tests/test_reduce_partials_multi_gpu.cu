// Per-GPU partials of a multi-GPU reduction, checked one by one.
//
// test_reduce_sum_multi_gpu only checks the combined value, and an end-to-end check can't
// tell "GPU g's stage 1 never wrote its partial" from "the combine read it too early". This
// test reads every ReductionResult::gpu_partial(g) back after the reduce has fully drained
// and compares it with GPU g's owned-cell count, over many rounds that alternate two grids
// of different sizes and two kernels. That is the pattern under which the SYCL backend lost
// one GPU's partial (test_mgrid_locator counting 8192 of 16384 cells, the combine reading a
// stale value from a previous reduction).
//
// Skipped (returns 0) with fewer than 2 visible devices.

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/reduce.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kRounds = 40;

struct Outcome {
  int bad_rounds = 0;
  int first_bad_round = -1;
};

/** One round: reduce `scale` per owned cell, then check every per-GPU partial and the total. */
auto
run_round(carbon::System& sys,
          const carbon::Dense& dense,
          const carbon::Stream& stream,
          int64_t scale,
          int round,
          const char* label) -> bool {
  carbon::ReductionResult<int64_t> result(sys);
  auto reducer = dense.make_reduce_kernel<int64_t>(
      [scale](carbon::KernelLoader&) {
        return [scale] CARBON_HOST_DEVICE(carbon::Dense::Span, carbon::Dense::Idx) -> int64_t {
          return scale;
        };
      },
      result, carbon::reduce::Sum<int64_t>{});
  reducer.launch(stream, carbon::DataView::Standard);
  result.sync(stream);
  stream.wait();

  const size_t n = sys.gpu_group().size();
  bool ok = true;
  int64_t want_total = 0;
  for (size_t g = 0; g < n; ++g) {
    auto host = carbon::Cpu::pinned_buffer<int64_t>(1U);
    sys.async_copy(result.gpu_partial(carbon::DataView::Standard, g), host, sizeof(int64_t),
                   stream);
    stream.wait();
    const int64_t want = dense.spans()[g].iter_extent(carbon::DataView::Standard).volume() * scale;
    want_total += want;
    if (host.data()[0] != want) {
      CARBON_LOG_ERROR("test_reduce_partials_multi_gpu[{} round {}]: gpu {} partial {} != {}",
                       label, round, g, host.data()[0], want);
      ok = false;
    }
  }
  const int64_t got_total = result.host_value(carbon::DataView::Standard);
  if (got_total != want_total) {
    CARBON_LOG_ERROR("test_reduce_partials_multi_gpu[{} round {}]: total {} != {}", label, round,
                     got_total, want_total);
    ok = false;
  }
  return ok;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_INFO("test_reduce_partials_multi_gpu: <2 visible devices, skipping");
    return EXIT_SUCCESS;
  }
  carbon::System sys(carbon::get_all_devices());
  const carbon::Stream stream = sys.stream();

  // Two grids of different sizes, like two mGrid levels.
  const carbon::Dense big(sys, carbon::int64_3d{32, 32, 64});
  const carbon::Dense small(sys, carbon::int64_3d{16, 16, 32});

  Outcome out;
  for (int r = 0; r < kRounds; ++r) {
    const bool use_big = (r % 2) == 0;
    const bool ok =
        run_round(sys, use_big ? big : small, stream, 1 + (r % 3), r, use_big ? "big" : "small");
    if (!ok) {
      ++out.bad_rounds;
      if (out.first_bad_round < 0) {
        out.first_bad_round = r;
      }
    }
  }
  if (out.bad_rounds != 0) {
    CARBON_LOG_ERROR("test_reduce_partials_multi_gpu: {} of {} rounds wrong (first: {})",
                     out.bad_rounds, kRounds, out.first_bad_round);
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_reduce_partials_multi_gpu: OK ({} rounds, {} GPUs)", kRounds,
                  sys.gpu_group().size());
  return EXIT_SUCCESS;
}
