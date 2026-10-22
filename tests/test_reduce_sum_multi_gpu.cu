// Multi-GPU sum-reduction over a Dense field — exercises the stage-2 intra-process
// combine path that lifts the gpu_count == 1 gate on ReduceKernelImpl.
//
// Pipeline under test:
//   stage 1 per local GPU: dense_grid_reduce_kernel + cub::DeviceReduce::Reduce
//                          → ReductionResult::gpu_partial(g)
//   stage 2 (intra-process): peer-copy gpu_partial(g) → stage2_staging[g] on root
//                            GPU, fan-in cross-stream sync, combine_partials_kernel
//                            on root GPU → ReductionResult::device_data(view)
//   stage 3 (cross-rank, NCCL): NOT exercised here — that lands in the dist test
//                               under CARBON_ENABLE_MPI=ON in step 5
//   D→H copy device_data(view) → host_data(view).
//
// Assertions:
//   - Standard-view sum over a deterministic encoder equals the host reference (bit-
//     exact int64 sum: cub::DeviceReduce + the combine kernel are integer additions, so
//     the result is independent of slab partitioning).
//   - Internal+Boundary launches into the same destination satisfy
//       host_value(Internal) + host_value(Boundary) == host_value(Standard)
//     and host_value_combined(Sum) over a separate destination equals Standard.
//   - Empty domain (z = 0) reduces to Sum::identity() == 0 across every per-GPU stage-1.
//
// Skipped (returns 0) when fewer than 2 visible CUDA devices are available — the
// single-GPU path is already covered by test_reduce_sum.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/stencil.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kWidth = 24;
constexpr int64_t kHeight = 32;
// Depth large enough that with two GPUs each slab still has more than 2 * halo_z owned
// cells in z, so the Internal / Boundary split has a non-empty inner band on every slab.
constexpr int64_t kDepth = 16;

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
        "test_reduce_sum_multi_gpu: <2 visible CUDA devices, skipping (treated as PASS)");
    return EXIT_SUCCESS;
  }

  carbon::System sys(carbon::get_all_devices());
  if (sys.gpu_group().size() < 2U) {
    CARBON_LOG_INFO("test_reduce_sum_multi_gpu: System has <2 GPUs after construction, skipping");
    return EXIT_SUCCESS;
  }

  // Radius-1 stencil so halo_z == 1 and the Internal/Boundary split makes sense.
  const carbon::Stencil stencil_radius1(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  const carbon::int64_3d global{kWidth, kHeight, kDepth};
  carbon::Dense dense(sys, global, stencil_radius1);

  auto field = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);
  field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
    return encode(gc.x, gc.y, gc.z);
  });

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<int64_t>(sys, field.cpu_buffers(), field.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_reduce_sum_multi_gpu: upload failed: {}", e.what());
    return EXIT_FAILURE;
  }

  const int64_t want = host_reference_sum(kWidth, kHeight, kDepth);

  // ---- Case 1: Standard view across multiple GPUs ----
  {
    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = dense.make_reduce_kernel<int64_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        result, carbon::reduce::Sum<int64_t>{});

    try {
      reducer.launch(stream, carbon::DataView::Standard);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum_multi_gpu: Standard launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);

    const int64_t got = result.host_value(carbon::DataView::Standard);
    if (got != want) {
      CARBON_LOG_ERROR(
          "test_reduce_sum_multi_gpu: Standard mismatch: got {}, expected {} (host ref)", got,
          want);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_sum_multi_gpu: Standard sum = {} over {} GPUs", got,
                    sys.gpu_group().size());
  }

  // ---- Case 2: Internal + Boundary recombine across multiple GPUs ----
  {
    carbon::ReductionResult<int64_t> split_only(sys);
    auto reducer = dense.make_reduce_kernel<int64_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        split_only, carbon::reduce::Sum<int64_t>{});

    try {
      reducer.launch(stream, carbon::DataView::Internal);
      reducer.launch(stream, carbon::DataView::Boundary);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_reduce_sum_multi_gpu: Internal/Boundary launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    split_only.sync(stream);

    const int64_t internal = split_only.host_value(carbon::DataView::Internal);
    const int64_t boundary = split_only.host_value(carbon::DataView::Boundary);
    if (internal + boundary != want) {
      CARBON_LOG_ERROR(
          "test_reduce_sum_multi_gpu: Internal({}) + Boundary({}) = {} != Standard({})", internal,
          boundary, internal + boundary, want);
      return EXIT_FAILURE;
    }

    const int64_t combined = split_only.host_value_combined(carbon::reduce::Sum<int64_t>{});
    if (combined != want) {
      CARBON_LOG_ERROR("test_reduce_sum_multi_gpu: host_value_combined({}) != Standard({})",
                       combined, want);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_reduce_sum_multi_gpu: split Internal({}) + Boundary({}) = combined {} matches "
        "Standard {}",
        internal, boundary, combined, want);
  }

  // ---- Case 3: Empty z-extent across multiple GPUs ----
  {
    const carbon::int64_3d empty_global{kWidth, kHeight, 0};
    carbon::Dense empty_dense(sys, empty_global, carbon::Stencil{});
    auto empty_field = empty_dense.newField<int64_t, 1>(carbon::MemorySide::Device);
    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = empty_dense.make_reduce_kernel<int64_t>(
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
      CARBON_LOG_ERROR("test_reduce_sum_multi_gpu: empty-domain launch failed: {}", e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);
    const int64_t got = result.host_value(carbon::DataView::Standard);
    if (got != 0) {
      CARBON_LOG_ERROR("test_reduce_sum_multi_gpu: empty-domain expected 0 (Sum identity), got {}",
                       got);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_reduce_sum_multi_gpu: empty-domain reduce -> {} (identity)", got);
  }

  CARBON_LOG_INFO("test_reduce_sum_multi_gpu: PASS");
  return EXIT_SUCCESS;
}
