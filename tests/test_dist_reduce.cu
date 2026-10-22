// Cross-rank Sum reduction test (one rank per GPU).
//
// Launched under `mpiexec -n 2 ./test_dist_reduce`. Each rank owns a Z-partitioned slab of a
// global Dense field, and we verify that the three-stage carbon reduce pipeline
//
//   stage 1 (per local GPU)         dense_grid_reduce_kernel + cub::DeviceReduce
//   stage 2 (intra-process combine) — single-GPU per rank under MPI, so a no-op fast path
//   stage 3 (cross-rank, NCCL/RCCL) ncclAllReduce in-place via dist_allreduce_in_place
//
// gives the same value as a host-side reference computed over the entire global volume. Both
// ranks then read host_value(view) after sync(stream); the value must be identical on every
// rank (allreduce post-condition).
//
// Cases covered:
//   1. Standard view across ranks  -> equals global host reference.
//   2. Internal + Boundary view across ranks -> sum equals global host reference, and the
//      ReductionResult::host_value_combined helper reproduces it independently.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are visible — the
// rank-local pipeline already has dedicated coverage in test_reduce_sum* under the
// non-MPI configuration.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/distributed.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/reduce_dist.h>
#include <carbon/stencil.h>
#include <carbon/system.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kWidth = 12;
constexpr int64_t kHeight = 16;

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t z) -> int64_t {
  // Same encoder shape as test_reduce_sum_multi_gpu so the math is easy to reason about and
  // the values are deterministic across ranks regardless of partition.
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

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_dist_reduce: world_size={} < 2, skipping", distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_dist_reduce: no CUDA devices, skipping");
    return EXIT_SUCCESS;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_dist_reduce: expected gpu_group size 1 (one rank per GPU), got {}",
                     sys.gpu_group().size());
    return EXIT_FAILURE;
  }

  // Pick a global z that is *not* divisible by world_size so the per-rank slab counts differ
  // by one — this is the same partition shape exercised in test_dense_dist and stresses the
  // path where the halved-domain Internal/Boundary split has different cell counts per rank.
  const int64_t global_z = static_cast<int64_t>(world_size) * 4 + 1;
  const carbon::int64_3d global{kWidth, kHeight, global_z};

  // Radius-1 stencil so halo_z == 1 and the Internal / Boundary split has a non-empty inner
  // band on every rank-local slab.
  const carbon::Stencil stencil_radius1(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
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
    CARBON_LOG_ERROR("test_dist_reduce[rank={}]: upload failed: {}", rank, e.what());
    return EXIT_FAILURE;
  }

  const int64_t want = host_reference_sum(kWidth, kHeight, global_z);

  // ---- Case 1: Standard view across ranks ----
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
      CARBON_LOG_ERROR("test_dist_reduce[rank={}]: Standard launch failed: {}", rank, e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);

    const int64_t got = result.host_value(carbon::DataView::Standard);
    if (got != want) {
      CARBON_LOG_ERROR(
          "test_dist_reduce[rank={}]: Standard mismatch: got {}, expected {} (host ref over "
          "global {} x {} x {})",
          rank, got, want, kWidth, kHeight, global_z);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dist_reduce[rank={}]: Standard sum = {} matches global ref over {} ranks",
                    rank, got, world_size);
  }

  // ---- Case 2: Internal + Boundary cross-rank recombine ----
  {
    carbon::ReductionResult<int64_t> split(sys);
    auto reducer = dense.make_reduce_kernel<int64_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        split, carbon::reduce::Sum<int64_t>{});

    try {
      reducer.launch(stream, carbon::DataView::Internal);
      reducer.launch(stream, carbon::DataView::Boundary);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dist_reduce[rank={}]: Internal/Boundary launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    split.sync(stream);

    const int64_t internal = split.host_value(carbon::DataView::Internal);
    const int64_t boundary = split.host_value(carbon::DataView::Boundary);
    if (internal + boundary != want) {
      CARBON_LOG_ERROR(
          "test_dist_reduce[rank={}]: Internal({}) + Boundary({}) = {} != global ref {}", rank,
          internal, boundary, internal + boundary, want);
      return EXIT_FAILURE;
    }

    const int64_t combined = split.host_value_combined(carbon::reduce::Sum<int64_t>{});
    if (combined != want) {
      CARBON_LOG_ERROR("test_dist_reduce[rank={}]: host_value_combined({}) != global ref {}", rank,
                       combined, want);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_dist_reduce[rank={}]: split Internal({}) + Boundary({}) = combined {} matches "
        "global ref {}",
        rank, internal, boundary, combined, want);
  }

  // ---- Case 3: Host (OpenMP) reduce + MPI_Allreduce on the host scalar ----
  // Mirrors cases 1-2 but goes through Dense::make_host_reduce_kernel, which after
  // step 6 walks every slab on the host, combines per-slab partials, and then folds
  // across ranks via MPI_Allreduce(MPI_IN_PLACE, host_data(view), ...). The result
  // must match the same global host reference (allreduce post-condition + identical
  // associative-commutative op on both pipelines).
  {
    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = dense.make_host_reduce_kernel<int64_t>(
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
      CARBON_LOG_ERROR("test_dist_reduce[rank={}]: host Standard launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);

    const int64_t got = result.host_value(carbon::DataView::Standard);
    if (got != want) {
      CARBON_LOG_ERROR(
          "test_dist_reduce[rank={}]: host Standard mismatch: got {}, expected {} (host ref)", rank,
          got, want);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_dist_reduce[rank={}]: host Standard sum = {} matches global ref over {} ranks "
        "(MPI_Allreduce on host scalar)",
        rank, got, world_size);
  }

  CARBON_LOG_INFO("test_dist_reduce[rank={}]: PASS", rank);
  return EXIT_SUCCESS;
}
