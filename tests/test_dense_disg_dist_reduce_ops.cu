// Cross-rank DenseDisg Min / Max / Prod reduction test (one rank per GPU).
//
// Closes P1.γ of docs/audit_2026_05_fourth_pass.md (sibling gap to the third
// pass's `test_dist_reduce_ops.cu`). The Sum-only Dense-Disg cross-rank reduce
// path is locked by `test_dense_disg_dist_reduce.cu`; this file extends the
// matrix to {Min, Max, Prod} so a regression in the disg-specific stage-1 / 2
// wiring that fans into the same stage-3 NCCL/RCCL allreduce — or in
// `reduce::op_kind<Op>::value` once it travels through the disg dispatch — is
// no longer silent.
//
// Launched under `mpiexec -n 2 ./test_dense_disg_dist_reduce_ops`. Each rank
// owns a Z-partitioned slab of a global DenseDisg field. The same encoders
// from `test_dist_reduce_ops.cu` are reused so a single host reference works
// across every rank without coordination.
//
// Cases covered:
//   1. Min  device — `make_reduce_kernel` + `reduce::Min` over an int32 field;
//                    locks ncclMin / MPI_MIN routing through DisgPolicy.
//   2. Max  device — same, `reduce::Max` (ncclMax / MPI_MAX).
//   3. Prod device — `make_reduce_kernel` + `reduce::Prod` over a double
//                    field with values slightly above 1.0 to keep the running
//                    product in finite range (matches `encode_double` in
//                    `test_dist_reduce_ops.cu`).
//   4. Min  host   — `make_host_reduce_kernel` + `reduce::Min` to lock the
//                    cross-rank `MPI_Allreduce(MPI_MIN)` path on a non-Sum op
//                    on the disg host launcher as well.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are
// visible — the rank-local DenseDisg reduce path already has dedicated
// non-MPI coverage in `test_dense_disg_reduce.cu`.

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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>

namespace {

constexpr int64_t kWidth = 8;
constexpr int64_t kHeight = 6;

[[nodiscard]] CARBON_HD auto
encode_int(int64_t x,
           int64_t y,
           int64_t z) -> std::int32_t {
  return static_cast<std::int32_t>(z * 100 + y * 10 + x + 1);
}

[[nodiscard]] CARBON_HD auto
encode_double(int64_t x,
              int64_t y,
              int64_t z,
              int64_t global_z) -> double {
  return 1.0 + (static_cast<double>(x + y * kWidth + z * kWidth * kHeight) /
                static_cast<double>(kWidth * kHeight * global_z)) *
                   0.04;
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_dense_disg_dist_reduce_ops: world_size={} < 2, skipping",
                    distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_dense_disg_dist_reduce_ops: no CUDA devices, skipping");
    return EXIT_SUCCESS;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_dense_disg_dist_reduce_ops: expected gpu_group size 1, got {}",
                     sys.gpu_group().size());
    return EXIT_FAILURE;
  }

  // Pick a global z that is *not* divisible by world_size so per-rank slab counts
  // differ by one — mirrors the shape used by test_dense_disg_dist_reduce and
  // test_dist_reduce_ops.
  const int64_t global_z = static_cast<int64_t>(world_size) * 4 + 1;
  const carbon::int64_3d global{kWidth, kHeight, global_z};
  // halo_z=1 keeps the disg span shape consistent with the Sum-only sibling.
  const carbon::DenseDisg dense(sys, global, /*halo_z=*/1);

  carbon::Stream stream = sys.stream();

  // ---- int32 field, seeded on device through a Standard-view kernel ----
  auto i_field = dense.newField<std::int32_t, 1>(carbon::MemorySide::HostDevice);
  {
    auto seeder = dense.make_kernel(
        [&i_field](carbon::KernelLoader& loader) {
          const auto out = loader(i_field);
          return [out] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s,
                                          carbon::DenseDisg::Idx idx) -> void {
            const auto gc = s.global_location(idx);
            out.write(s, idx, 0, encode_int(gc.x, gc.y, gc.z));
          };
        },
        /*shared_mem=*/0U);
    try {
      seeder.launch(stream);
      // Mirror device -> host so the host Min path below reads the seeded
      // values. DenseDisg::Field has no fill_owned_host today (P2.α in the
      // fourth-pass audit), so we seed on device and pull back, which also
      // exercises the disg device->host Transfer<T> path on owned + halo
      // slabs in passing.
      carbon::Transfer<std::int32_t>(sys, i_field.gpu_buffers(), i_field.cpu_buffers())(stream);
      stream.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_dist_reduce_ops[rank={}]: int32 seed failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
  }

  // Host references — every rank computes the same value because `encode_*` is
  // deterministic over the entire global volume.
  std::int32_t host_min = std::numeric_limits<std::int32_t>::max();
  std::int32_t host_max = std::numeric_limits<std::int32_t>::lowest();
  for (int64_t z = 0; z < global_z; ++z) {
    for (int64_t y = 0; y < kHeight; ++y) {
      for (int64_t x = 0; x < kWidth; ++x) {
        host_min = std::min(host_min, encode_int(x, y, z));
        host_max = std::max(host_max, encode_int(x, y, z));
      }
    }
  }

  const auto i_loader = [&i_field](carbon::KernelLoader& loader) {
    const auto in = loader(i_field);
    return [in] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s,
                                   carbon::DenseDisg::Idx idx) -> std::int32_t {
      return in.read(s, idx, 0);
    };
  };

  // ---- Min (device path, NCCL/RCCL ncclMin) ----
  {
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k =
        dense.make_reduce_kernel<std::int32_t>(i_loader, r, carbon::reduce::Min<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_dist_reduce_ops[rank={}]: Min launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    stream.wait();
    if (r.host_value() != host_min) {
      CARBON_LOG_ERROR(
          "test_dense_disg_dist_reduce_ops[rank={}]: Min mismatch: got {}, expected global ref {}",
          rank, r.host_value(), host_min);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dense_disg_dist_reduce_ops[rank={}]: Min = {} (matches global ref)", rank,
                    r.host_value());
  }

  // ---- Max (device path, NCCL/RCCL ncclMax) ----
  {
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k =
        dense.make_reduce_kernel<std::int32_t>(i_loader, r, carbon::reduce::Max<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_dist_reduce_ops[rank={}]: Max launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    stream.wait();
    if (r.host_value() != host_max) {
      CARBON_LOG_ERROR(
          "test_dense_disg_dist_reduce_ops[rank={}]: Max mismatch: got {}, expected global ref {}",
          rank, r.host_value(), host_max);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dense_disg_dist_reduce_ops[rank={}]: Max = {} (matches global ref)", rank,
                    r.host_value());
  }

  // ---- Prod (device path, NCCL/RCCL ncclProd) over a double field ----
  {
    auto d_field = dense.newField<double, 1>(carbon::MemorySide::HostDevice);
    auto seeder = dense.make_kernel(
        [&d_field, global_z](carbon::KernelLoader& loader) {
          const auto out = loader(d_field);
          return [out, global_z] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s,
                                                    carbon::DenseDisg::Idx idx) -> void {
            const auto gc = s.global_location(idx);
            out.write(s, idx, 0, encode_double(gc.x, gc.y, gc.z, global_z));
          };
        },
        /*shared_mem=*/0U);
    try {
      seeder.launch(stream);
      stream.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_dist_reduce_ops[rank={}]: double seed failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }

    double host_prod = 1.0;
    for (int64_t z = 0; z < global_z; ++z) {
      for (int64_t y = 0; y < kHeight; ++y) {
        for (int64_t x = 0; x < kWidth; ++x) {
          host_prod *= encode_double(x, y, z, global_z);
        }
      }
    }

    carbon::ReductionResult<double> r(sys);
    auto k = dense.make_reduce_kernel<double>(
        [&d_field](carbon::KernelLoader& loader) {
          const auto in = loader(d_field);
          return [in] CARBON_HOST_DEVICE(carbon::DenseDisg::Span s,
                                         carbon::DenseDisg::Idx idx) -> double {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Prod<double>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_dist_reduce_ops[rank={}]: Prod launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    stream.wait();

    const double got = r.host_value();
    const double rel = std::abs(got - host_prod) / std::abs(host_prod);
    constexpr double kRelTol = 1e-9;
    if (rel > kRelTol) {
      CARBON_LOG_ERROR(
          "test_dense_disg_dist_reduce_ops[rank={}]: Prod mismatch: got {:.10g}, expected {:.10g} "
          "(rel diff {:.3e})",
          rank, got, host_prod, rel);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dense_disg_dist_reduce_ops[rank={}]: Prod = {:.10g} (rel diff {:.3e})",
                    rank, got, rel);
  }

  // ---- Min via host (OpenMP) reduce + MPI_Allreduce ----
  // Locks make_host_reduce_kernel's cross-rank path on a non-Sum op for the
  // disg host launcher (the host-side fold uses MPI_MIN through the same
  // op_kind trait that powers the device path). Reuses i_field's host mirror
  // — DenseDisg::Field<T, C> with MemorySide::HostDevice keeps both sides in
  // sync after the seeder kernel, so the host launcher reads the same values
  // as the host reference.
  {
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k = dense.make_host_reduce_kernel<std::int32_t>(
        [&i_field](carbon::KernelLoader& loader) {
          const auto in = loader(i_field);
          return [in](carbon::DenseDisg::Span s, carbon::DenseDisg::Idx idx) -> std::int32_t {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Min<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dense_disg_dist_reduce_ops[rank={}]: host Min launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    stream.wait();
    if (r.host_value() != host_min) {
      CARBON_LOG_ERROR(
          "test_dense_disg_dist_reduce_ops[rank={}]: host Min mismatch: got {}, expected global "
          "ref {}",
          rank, r.host_value(), host_min);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_dense_disg_dist_reduce_ops[rank={}]: host Min = {} (MPI_Allreduce on host scalar)",
        rank, r.host_value());
  }

  CARBON_LOG_INFO("test_dense_disg_dist_reduce_ops[rank={}]: PASS", rank);
  return EXIT_SUCCESS;
}
