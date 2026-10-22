// Cross-rank DenseDisg compute test (one rank per GPU).
//
// Launched under `mpiexec -n 2 ./test_dense_disg_dist`. Each rank constructs a
// distributed DenseDisg over a global volume and runs a compute kernel that writes
// encode(gx, gy, gz) into every owned cell (no halo dependencies). After download,
// every rank verifies that its local slab carries the expected encoded values,
// proving that DenseDisg::make_kernel — the DisgPolicy-tagged dispatch path
// through dense_disg::launch_disg_grid — works correctly under MPI.
//
// The check is intentionally halo-free because HaloDisg has no NCCL transport
// yet (see <carbon/halo_disg.h>); cross-rank reductions are exercised in a
// companion test_dense_disg_dist_reduce.cu via the existing stage-3 NCCL
// allreduce in <carbon/reduce_dist.h>.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are visible.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_disg.h>
#include <carbon/distributed.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/system.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>

namespace {

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t z) -> int64_t {
  return z * 100000 + y * 100 + x - 50000;
}

[[nodiscard]] auto
disg_addr(const carbon::DenseDisg::Span& span,
          int64_t bz,
          int64_t x,
          int64_t y,
          int c,
          int components) -> int64_t {
  const int64_t lz = span.logic_extent().z;
  const int64_t xdim = span.haloed_extent().x;
  const int64_t xy = span.xy_slab_elements();
  int64_t region_first_z = 0;
  int64_t region_z_dim = 1;
  int64_t offset_local_no_card = x + y * xdim;
  if (bz <= 1 || bz >= lz) {
    region_first_z = bz;
    region_z_dim = 1;
  } else {
    region_first_z = 2;
    region_z_dim = lz - 2;
    offset_local_no_card += (bz - 2) * xy;
  }
  return offset_local_no_card + xy * (region_first_z * static_cast<int64_t>(components) +
                                      static_cast<int64_t>(c) * region_z_dim);
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_dense_disg_dist: world_size={} < 2, skipping", distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_dense_disg_dist: no CUDA devices, skipping");
    return EXIT_SUCCESS;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_dense_disg_dist: expected gpu_group size 1, got {}",
                     sys.gpu_group().size());
    return EXIT_FAILURE;
  }

  // global_z not divisible by world_size so the partition has a non-empty remainder
  // — exercises the "first (z % world_size) slabs get one extra layer" branch.
  const int64_t global_z = static_cast<int64_t>(world_size) * 4 + 1;
  const carbon::int64_3d global{6, 8, global_z};
  const carbon::DenseDisg dense(sys, global, /*halo_z=*/1);

  if (dense.global_slab_count() != static_cast<size_t>(world_size)) {
    CARBON_LOG_ERROR("test_dense_disg_dist[rank={}]: global_slab_count={} expected {}", rank,
                     dense.global_slab_count(), world_size);
    return EXIT_FAILURE;
  }
  if (dense.local_slab_offset() != static_cast<size_t>(rank)) {
    CARBON_LOG_ERROR("test_dense_disg_dist[rank={}]: local_slab_offset={} expected {}", rank,
                     dense.local_slab_offset(), rank);
    return EXIT_FAILURE;
  }

  auto u = dense.newField<int64_t, 1>(carbon::MemorySide::HostDevice);

  const auto kernel = dense.make_kernel(
      [&u](carbon::KernelLoader& loader) {
        const auto out = loader(u);
        return [out] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s, carbon::DenseDisg::Idx idx) {
          const auto gc = s.global_location(idx);
          out.write(s, idx, 0, encode(gc.x, gc.y, gc.z));
        };
      },
      /*shared_mem=*/0U);

  carbon::Stream stream = sys.stream();
  try {
    kernel.launch(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_dist[rank={}]: launch failed: {}", rank, e.what());
    return EXIT_FAILURE;
  }

  try {
    carbon::Transfer<int64_t>(sys, u.gpu_buffers(), u.cpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_dist[rank={}]: download failed: {}", rank, e.what());
    return EXIT_FAILURE;
  }

  const auto& span = dense.spans()[0];
  const auto lim = span.logic_extent();
  const int64_t halo_z = span.halo_z();
  const int64_t z_origin = span.z_offset();
  const int64_t* h = u.cpu_buffers()[0].data();
  size_t mismatches = 0;
  for (int64_t z = 0; z < lim.z; ++z) {
    const int64_t bz = halo_z + z;
    const int64_t gz = z_origin + bz;
    for (int64_t y = 0; y < lim.y; ++y) {
      for (int64_t x = 0; x < lim.x; ++x) {
        const int64_t off = disg_addr(span, bz, x, y, 0, /*components=*/1);
        const int64_t got = h[off];
        const int64_t want = encode(x, y, gz);
        if (got != want) {
          if (mismatches < 4U) {
            CARBON_LOG_ERROR(
                "test_dense_disg_dist[rank={}]: mismatch at (x,y,z)=({},{},{}) gz={} got={} "
                "want={}",
                rank, x, y, z, gz, got, want);
          }
          ++mismatches;
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_dense_disg_dist[rank={}]: {} mismatches", rank, mismatches);
    return EXIT_FAILURE;
  }
  CARBON_LOG_INFO("test_dense_disg_dist[rank={}]: PASS (slab z_origin={}, lim_z={})", rank,
                  z_origin, lim.z);
  return EXIT_SUCCESS;
}
