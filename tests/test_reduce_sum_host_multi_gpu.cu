// Multi-slab host (OpenMP) sum-reduction test — exercises the per-slab combine path
// added in step 6 that lifts the gpu_count == 1 gate from ReduceHostKernelImpl.
//
// Pipeline under test (host path):
//   for slab g in 0..gpu_count:
//     KernelLoader{g, KernelTarget::Host} -> per-cell host_fn
//     OpenMP parallel-for over span.iter_extent(view) -> thread-local partials
//     sequential combine -> per-slab scalar
//   sequential combine across slabs -> dst.host_data(view)[0]
//
// Stage 3 (cross-rank MPI_Allreduce on the host scalar) is not exercised here — that
// lands in step 7 / the docker MPI matrix via test_dist_reduce.
//
// Assertions:
//   - Standard view across multiple host-side slabs equals the host reference.
//   - Internal + Boundary recombine (per-slab + cross-slab) reproduces the Standard
//     total via host_value(Internal)+host_value(Boundary) and host_value_combined.
//   - Empty z-extent reduces to Sum::identity() == 0 across every slab.
//
// Skipped (returns 0) when fewer than 2 visible CUDA devices are available — the
// single-slab host path is already covered by test_kernel_host_parity.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/stencil.h>

#include "test_domain.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kWidth = 12;
constexpr int64_t kHeight = 16;
constexpr int64_t kDepth = 11;  // not divisible by 2 → uneven slab counts

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t z) -> int64_t {
  return z * 100000 + y * 100 + x - 50000;
}

[[nodiscard]] auto
host_reference_sum(int64_t W,
                   int64_t H,
                   int64_t D) -> int64_t {
  int64_t s = 0;
  for (int64_t z = 0; z < D; ++z) {
    for (int64_t y = 0; y < H; ++y) {
      for (int64_t x = 0; x < W; ++x) {
        s += encode(x, y, z);
      }
    }
  }
  return s;
}

}  // namespace

int
main() {
  carbon::init();
  if (carbon::device_count() < 2) {
    CARBON_LOG_INFO(
        "test_reduce_sum_host_multi_gpu: <2 visible CUDA devices, skipping (treated as PASS)");
    return EXIT_SUCCESS;
  }

  carbon::System sys(carbon::get_all_devices());
  if (sys.gpu_group().size() < 2U) {
    CARBON_LOG_INFO(
        "test_reduce_sum_host_multi_gpu: System has <2 GPUs after construction, skipping");
    return EXIT_SUCCESS;
  }

  const carbon::Stencil stencil_radius1(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  // Grow z so the slab split stays legal on a many-GPU host, then keep it indivisible by the
  // slab count so the uneven-remainder path this test exists for stays covered.
  const size_t gpu_n = sys.gpu_group().size();
  int64_t depth = carbon::test::dense_slab_z(kDepth, gpu_n, stencil_radius1.radius());
  if (depth % static_cast<int64_t>(gpu_n) == 0) {
    ++depth;
  }
  const carbon::int64_3d global{kWidth, kHeight, depth};
  carbon::Dense dense(sys, global, stencil_radius1);

  // Host reduce reads from the CPU mirror, so MemorySide must include Host. We do not
  // need to populate the device side — the host reducer never calls the GPU.
  auto field = dense.newField<int64_t, 1>(carbon::MemorySide::Host);
  field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
    return encode(gc.x, gc.y, gc.z);
  });

  carbon::Stream stream = sys.stream();
  const int64_t want = host_reference_sum(kWidth, kHeight, depth);

  auto reduce_loader = [&field](carbon::KernelLoader& loader) {
    const auto in = loader(field);
    return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
      return in.read(s, idx, 0);
    };
  };

  // ---- Case 1: Standard view across multiple slabs (host path) ----
  {
    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = dense.make_host_reduce_kernel<int64_t>(reduce_loader, result,
                                                          carbon::reduce::Sum<int64_t>{});

    try {
      reducer.launch(stream, carbon::DataView::Standard);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum_host_multi_gpu: Standard launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);

    const int64_t got = result.host_value(carbon::DataView::Standard);
    if (got != want) {
      CARBON_LOG_ERROR(
          "test_reduce_sum_host_multi_gpu: Standard mismatch: got {}, expected {} (host ref)", got,
          want);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_sum_host_multi_gpu: Standard sum = {} over {} host-side slabs",
                    got, sys.gpu_group().size());
  }

  // ---- Case 2: Internal + Boundary recombine across multiple slabs ----
  {
    carbon::ReductionResult<int64_t> split(sys);
    auto reducer = dense.make_host_reduce_kernel<int64_t>(reduce_loader, split,
                                                          carbon::reduce::Sum<int64_t>{});

    try {
      reducer.launch(stream, carbon::DataView::Internal);
      reducer.launch(stream, carbon::DataView::Boundary);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum_host_multi_gpu: Internal/Boundary launch failed: {}",
                       e.what());
      return EXIT_FAILURE;
    }
    split.sync(stream);

    const int64_t internal = split.host_value(carbon::DataView::Internal);
    const int64_t boundary = split.host_value(carbon::DataView::Boundary);
    if (internal + boundary != want) {
      CARBON_LOG_ERROR(
          "test_reduce_sum_host_multi_gpu: Internal({}) + Boundary({}) = {} != Standard({})",
          internal, boundary, internal + boundary, want);
      return EXIT_FAILURE;
    }
    const int64_t combined = split.host_value_combined(carbon::reduce::Sum<int64_t>{});
    if (combined != want) {
      CARBON_LOG_ERROR("test_reduce_sum_host_multi_gpu: host_value_combined({}) != Standard({})",
                       combined, want);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_reduce_sum_host_multi_gpu: Internal({}) + Boundary({}) = combined {} matches "
        "Standard {}",
        internal, boundary, combined, want);
  }

  // ---- Case 3: Empty z-extent across multiple slabs ----
  {
    const carbon::int64_3d empty_global{kWidth, kHeight, 0};
    carbon::Dense empty_dense(sys, empty_global, carbon::Stencil{});
    auto empty_field = empty_dense.newField<int64_t, 1>(carbon::MemorySide::Host);
    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = empty_dense.make_host_reduce_kernel<int64_t>(
        [&empty_field](carbon::KernelLoader& loader) {
          const auto in = loader(empty_field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        result, carbon::reduce::Sum<int64_t>{});
    try {
      reducer.launch(stream, carbon::DataView::Standard);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum_host_multi_gpu: empty-domain launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);
    const int64_t got = result.host_value(carbon::DataView::Standard);
    if (got != 0) {
      CARBON_LOG_ERROR(
          "test_reduce_sum_host_multi_gpu: empty-domain expected 0 (Sum identity), got {}", got);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_sum_host_multi_gpu: empty-domain reduce -> {} (identity)", got);
  }

  CARBON_LOG_INFO("test_reduce_sum_host_multi_gpu: PASS");
  return EXIT_SUCCESS;
}
