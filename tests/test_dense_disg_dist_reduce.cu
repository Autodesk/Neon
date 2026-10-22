// Cross-rank DenseDisg Sum reduction test (one rank per GPU).
//
// Launched under `mpiexec -n 2 ./test_dense_disg_dist_reduce`. Each rank owns a
// Z-partitioned slab of a global DenseDisg field, and we verify that the same
// three-stage reduce pipeline that powers Dense + Block works on the disaggregated
// layout:
//
//   stage 1 (per local GPU)         dense_disg_grid_reduce_kernel + cub::DeviceReduce
//   stage 2 (intra-process combine) — single-GPU per rank under MPI, so a no-op fast path
//   stage 3 (cross-rank, NCCL/RCCL) ncclAllReduce in-place via dist_allreduce_in_place
//
// gives the same value as a host-side reference computed over the entire global volume.
// The reducer goes through DenseDisg::make_reduce_kernel which, after step 8 of
// current_status.md, instantiates UniversalReduceKernelImpl with DisgPolicy and
// reaches detail::dense_disg::launch_disg_reduce.
//
// Cases covered:
//   1. Standard view across ranks  -> equals global host reference.
//   2. Internal + Boundary view across ranks -> sum equals global host reference, and the
//      ReductionResult::host_value_combined helper reproduces it independently.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are visible — the
// rank-local pipeline already has dedicated coverage in test_dense_disg_reduce under
// the non-MPI configuration.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_disg.h>
#include <carbon/distributed.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/reduce_dist.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>

namespace {

constexpr int64_t kWidth = 6;
constexpr int64_t kHeight = 8;

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

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_dense_disg_dist_reduce: world_size={} < 2, skipping",
                    distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_dense_disg_dist_reduce: no CUDA devices, skipping");
    return EXIT_SUCCESS;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_dense_disg_dist_reduce: expected gpu_group size 1, got {}",
                     sys.gpu_group().size());
    return EXIT_FAILURE;
  }

  const int64_t global_z = static_cast<int64_t>(world_size) * 4 + 1;
  const carbon::int64_3d global{kWidth, kHeight, global_z};
  // halo_z = 1 so the Internal / Boundary split has a non-empty inner band on
  // every rank-local slab.
  const carbon::DenseDisg dense(sys, global, /*halo_z=*/1);

  auto field = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);

  // Seed every owned cell on the host using a Standard-view kernel.
  auto seeder = dense.make_kernel(
      [&field](carbon::KernelLoader& loader) {
        const auto out = loader(field);
        return [out] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s, carbon::DenseDisg::Idx idx) {
          const auto gc = s.global_location(idx);
          out.write(s, idx, 0, encode(gc.x, gc.y, gc.z));
        };
      },
      /*shared_mem=*/0U);

  carbon::Stream stream = sys.stream();
  try {
    seeder.launch(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_dist_reduce[rank={}]: seed launch failed: {}", rank,
                     e.what());
    return EXIT_FAILURE;
  }

  const int64_t want = host_reference_sum(kWidth, kHeight, global_z);

  // ---- Case 1: Standard view across ranks ----
  {
    carbon::ReductionResult<int64_t> result(sys);
    auto reducer = dense.make_reduce_kernel<int64_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s,
                                         carbon::DenseDisg::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        result, carbon::reduce::Sum<int64_t>{});

    try {
      reducer.launch(stream, carbon::DataView::Standard);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_dist_reduce[rank={}]: Standard launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    result.sync(stream);
    stream.wait();

    const int64_t got = result.host_value(carbon::DataView::Standard);
    if (got != want) {
      CARBON_LOG_ERROR(
          "test_dense_disg_dist_reduce[rank={}]: Standard mismatch: got {}, expected {} "
          "(host ref over global {} x {} x {})",
          rank, got, want, kWidth, kHeight, global_z);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_dense_disg_dist_reduce[rank={}]: Standard sum = {} matches global ref over {} "
        "ranks",
        rank, got, world_size);
  }

  // ---- Case 2: Internal + Boundary cross-rank recombine ----
  {
    carbon::ReductionResult<int64_t> split(sys);
    auto reducer = dense.make_reduce_kernel<int64_t>(
        [&field](carbon::KernelLoader& loader) {
          const auto in = loader(field);
          return [in] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s,
                                         carbon::DenseDisg::Idx idx) -> int64_t {
            return in.read(s, idx, 0);
          };
        },
        split, carbon::reduce::Sum<int64_t>{});

    try {
      reducer.launch(stream, carbon::DataView::Internal);
      reducer.launch(stream, carbon::DataView::Boundary);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_dist_reduce[rank={}]: Internal/Boundary launch failed: {}",
                       rank, e.what());
      return EXIT_FAILURE;
    }
    split.sync(stream);
    stream.wait();

    const int64_t internal = split.host_value(carbon::DataView::Internal);
    const int64_t boundary = split.host_value(carbon::DataView::Boundary);
    if (internal + boundary != want) {
      CARBON_LOG_ERROR(
          "test_dense_disg_dist_reduce[rank={}]: Internal({}) + Boundary({}) = {} != global "
          "ref {}",
          rank, internal, boundary, internal + boundary, want);
      return EXIT_FAILURE;
    }

    const int64_t combined = split.host_value_combined(carbon::reduce::Sum<int64_t>{});
    if (combined != want) {
      CARBON_LOG_ERROR(
          "test_dense_disg_dist_reduce[rank={}]: host_value_combined({}) != global ref {}", rank,
          combined, want);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_dense_disg_dist_reduce[rank={}]: split Internal({}) + Boundary({}) = combined "
        "{} matches global ref {}",
        rank, internal, boundary, combined, want);
  }

  CARBON_LOG_INFO("test_dense_disg_dist_reduce[rank={}]: PASS", rank);
  return EXIT_SUCCESS;
}
