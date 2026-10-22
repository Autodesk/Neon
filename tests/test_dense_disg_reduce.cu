// Sum-reduction over a DenseDisg field via DenseDisg::make_reduce_kernel + the
// generic ReduceKernelImpl. Exercises the DisgPolicy if-constexpr dispatch for
// stage 1 (disg_grid_reduce_kernel under CUDA, launch_disg_reduce under SYCL) and
// for the per-cell pitch math the dense reducer otherwise hardcodes against
// dense::Span.
//
// Runs on whatever local GPU count is visible: single-GPU paths exercise the
// stage-1 short-circuit; ≥2 GPUs route through stage-2 intra-process combine
// from reduce_stages, same as Dense's multi-GPU reduce.

#include <carbon/carbon.h>
#include <carbon/cuda_annotation.h>
#include <carbon/dense_disg.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/transfer.h>

#include <cstdint>
#include <exception>
#include <vector>

namespace {

constexpr int64_t kComponents = 1;

[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t gz) -> int64_t {
  return gz * 100000 + y * 100 + x - 50000;
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
  if (carbon::device_count() <= 0) {
    CARBON_LOG_ERROR("test_dense_disg_reduce: no CUDA devices");
    return 1;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  // Geometry: z must be at least 2 * halo_z * n_gpus so every slab has Lz >= 2 (the
  // Internal / Boundary split contract).
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{12, 7, z_per_slab * static_cast<int64_t>(n_gpus)};
  const carbon::DenseDisg dense(sys, global, /*halo_z=*/1);

  if (dense.gpu_count() != n_gpus) {
    CARBON_LOG_ERROR("test_dense_disg_reduce: gpu_count {} != n_gpus {}", dense.gpu_count(),
                     n_gpus);
    return 1;
  }

  auto field = dense.newField<int64_t, kComponents>(carbon::MemorySide::HostDevice);

  // Host fill: every owned cell carries encode(x, y, gz). Halos / ghosts are sentinel-
  // zero — the reducer only iterates owned cells (Standard view), so halo content does
  // not influence the result.
  for (size_t i = 0; i < n_gpus; ++i) {
    const auto& span = dense.spans()[i];
    const auto haloed = span.haloed_extent();
    int64_t* h = field.cpu_buffers()[i].data();
    for (size_t j = 0; j < field.cpu_buffers()[i].size(); ++j) {
      h[j] = 0;
    }
    const int64_t halo_z = span.halo_z();
    const int64_t z_origin = span.z_offset();
    const int64_t lim_z = span.logic_extent().z;
    for (int64_t bz = halo_z; bz < halo_z + lim_z; ++bz) {
      const int64_t gz = z_origin + bz;
      for (int64_t y = 0; y < haloed.y; ++y) {
        for (int64_t x = 0; x < haloed.x; ++x) {
          const int64_t off = disg_addr(span, bz, x, y, 0, kComponents);
          h[off] = encode(x, y, gz);
        }
      }
    }
  }

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<int64_t> upload(sys, field.cpu_buffers(), field.gpu_buffers());
    upload(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_reduce: upload failed: {}", e.what());
    return 1;
  }

  carbon::ReductionResult<int64_t> dst(sys);

  const auto kernel = dense.make_reduce_kernel<int64_t>(
      [&field](carbon::KernelLoader& loader) {
        const auto in = loader.read(field);
        return [in] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s,
                                       carbon::DenseDisg::Idx idx) -> int64_t {
          return in.read(s, idx, 0);
        };
      },
      dst, carbon::reduce::Sum<int64_t>{});

  try {
    kernel.launch(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_reduce: kernel launch failed: {}", e.what());
    return 1;
  }
  try {
    dst.sync(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_reduce: dst.sync failed: {}", e.what());
    return 1;
  }

  const int64_t got = dst.host_value();
  const int64_t want = host_reference_sum(global.x, global.y, global.z);
  if (got != want) {
    CARBON_LOG_ERROR("test_dense_disg_reduce: got={} want={} (delta={})", got, want, got - want);
    return 1;
  }
  CARBON_LOG_INFO("test_dense_disg_reduce: ok ({} GPUs, sum={})", n_gpus, got);
  return 0;
}
