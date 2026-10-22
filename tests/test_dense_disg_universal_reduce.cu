// Sum-reduction over a DenseDisg field via DenseDisg::make_universal_reduce_kernel
// — the dual-target factory exercised by neither test_dense_disg_reduce (device-
// only ReduceKernelImpl) nor test_dense_disg_host (host-only ReduceHostKernelImpl).
// Anchors gap #1 of P1.δ in docs/audit_2026_05_third_pass.md: every other
// DenseDisg factory had a test, but make_universal_reduce_kernel was uncovered;
// only Dense (test_kernel_host_parity) and Block (test_block_universal)
// exercised the universal-reduce factory.
//
// Verifies:
//   * The factory returns a UniversalReduction-kind kernel (runs_on_device +
//     runs_on_host both true).
//   * launch_on_device delivers the same numerical result as
//     test_dense_disg_reduce's device-only ReduceKernelImpl path.
//   * launch_on_host delivers the same numerical result via the OpenMP
//     per-thread-partials + sequential combine path (same body lambda; both
//     launch entry points instantiated from one CARBON_HOST_DEVICE callable).
//   * The two paths agree to the full-precision integer.
//
// Geometry / fill mirrors test_dense_disg_reduce's so the reference sum is
// computed identically.

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
    CARBON_LOG_ERROR("test_dense_disg_universal_reduce: no GPU devices");
    return 1;
  }

  carbon::System sys(carbon::get_all_devices());
  const size_t n_gpus = sys.gpu_group().size();

  // Geometry — identical to test_dense_disg_reduce so the reference sum
  // is comparable.
  const int64_t z_per_slab = 4;
  const carbon::int64_3d global{12, 7, z_per_slab * static_cast<int64_t>(n_gpus)};
  const carbon::DenseDisg dense(sys, global, /*halo_z=*/1);

  if (dense.gpu_count() != n_gpus) {
    CARBON_LOG_ERROR("test_dense_disg_universal_reduce: gpu_count {} != n_gpus {}",
                     dense.gpu_count(), n_gpus);
    return 1;
  }

  // HostDevice mirror — both launch paths need the host mirror populated:
  // launch_on_host iterates the host buffers directly, launch_on_device pulls
  // from the device buffers, and we upload to device after host fill.
  auto field = dense.newField<int64_t, kComponents>(carbon::MemorySide::HostDevice);

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
    CARBON_LOG_ERROR("test_dense_disg_universal_reduce: upload failed: {}", e.what());
    return 1;
  }

  // Two ReductionResults: one for the device path, one for the host path.
  carbon::ReductionResult<int64_t> dev_dst(sys);
  carbon::ReductionResult<int64_t> host_dst(sys);

  const auto reduce_loader = [&field](carbon::KernelLoader& loader) {
    const auto in = loader.read(field);
    return
        [in] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s, carbon::DenseDisg::Idx idx) -> int64_t {
          return in.read(s, idx, 0);
        };
  };

  const auto dev_reducer = dense.make_universal_reduce_kernel<int64_t>(
      reduce_loader, dev_dst, carbon::reduce::Sum<int64_t>{});
  const auto host_reducer = dense.make_universal_reduce_kernel<int64_t>(
      reduce_loader, host_dst, carbon::reduce::Sum<int64_t>{});

  // Kind / target probes — the universal-reduce factory must advertise both
  // targets and the UniversalReduction kind tag (so dispatch through
  // KernelKind switches lands on the right branch).
  if (dev_reducer.kind() != carbon::KernelKind::UniversalReduction ||
      !dev_reducer.runs_on_device() || !dev_reducer.runs_on_host()) {
    CARBON_LOG_ERROR(
        "test_dense_disg_universal_reduce: kind={} runs_on_device={} runs_on_host={} — expected "
        "UniversalReduction / true / true",
        static_cast<int>(dev_reducer.kind()), dev_reducer.runs_on_device(),
        dev_reducer.runs_on_host());
    return 1;
  }

  try {
    dev_reducer.launch_on_device(stream);
    dev_dst.sync(stream);
    host_reducer.launch_on_host(stream);
    host_dst.sync(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dense_disg_universal_reduce: launch failed: {}", e.what());
    return 1;
  }

  const int64_t got_dev = dev_dst.host_value();
  const int64_t got_host = host_dst.host_value();
  const int64_t want = host_reference_sum(global.x, global.y, global.z);

  if (got_dev != want) {
    CARBON_LOG_ERROR("test_dense_disg_universal_reduce (device): got={} want={} (delta={})",
                     got_dev, want, got_dev - want);
    return 1;
  }
  if (got_host != want) {
    CARBON_LOG_ERROR("test_dense_disg_universal_reduce (host): got={} want={} (delta={})", got_host,
                     want, got_host - want);
    return 1;
  }
  if (got_dev != got_host) {
    CARBON_LOG_ERROR("test_dense_disg_universal_reduce: device {} != host {}", got_dev, got_host);
    return 1;
  }

  CARBON_LOG_INFO("test_dense_disg_universal_reduce: ok ({} GPUs, sum={})", n_gpus, got_dev);
  return 0;
}
