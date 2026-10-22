// Cross-rank Min / Max / Prod reduction test (one rank per GPU).
//
// Anchors gap #2 of P1.δ in docs/audit_2026_05_third_pass.md: test_reduce_ops
// covers reduce::{Min, Max, Prod} on a single GPU and test_dist_reduce
// covers Sum cross-rank, but no test exercised the non-Sum operators
// through the cross-rank stage-3 NCCL/RCCL allreduce path. A bug in
// reduce::op_kind<Op>::value (the trait that maps Min / Max / Prod to
// ncclMin / ncclMax / ncclProd) would have survived the existing matrix.
//
// Launched under `mpiexec -n 2 ./test_dist_reduce_ops`. Each rank owns a
// Z-partitioned slab of a global Dense field, runs the same three-stage
// pipeline as test_dist_reduce but with each of the three non-Sum ops in
// turn, and verifies the host_value() agrees on every rank with a
// host-side reference computed over the full global volume. Both the
// device and host (OpenMP + MPI_Allreduce) reduce kernels are exercised
// for Min, Max, and Prod — locking make_host_reduce_kernel's cross-rank
// path on every non-Sum op.
//
// Skipped (returns 0) when world_size < 2 or no GPUs are visible.

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
#include <carbon/transfer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

constexpr int64_t kWidth = 8;
constexpr int64_t kHeight = 6;

// Per-cell encoder — strictly positive, deterministic across ranks.
[[nodiscard]] CARBON_HD auto
encode(int64_t x,
       int64_t y,
       int64_t z) -> std::int32_t {
  return static_cast<std::int32_t>(z * 100 + y * 10 + x + 1);
}

// Per-cell encoder for Prod — keep contributions slightly above 1 so the
// running product stays in finite range. Same shape as test_reduce_ops's
// encode_double.
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
    CARBON_LOG_INFO("test_dist_reduce_ops: world_size={} < 2, skipping", distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_dist_reduce_ops: no GPU devices, skipping");
    return EXIT_SUCCESS;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_dist_reduce_ops: expected gpu_group size 1, got {}",
                     sys.gpu_group().size());
    return EXIT_FAILURE;
  }

  // Pick a global z that is *not* divisible by world_size so per-rank slab counts
  // differ by one — same shape used by test_dist_reduce.
  const int64_t global_z = static_cast<int64_t>(world_size) * 4 + 1;
  const carbon::int64_3d global{kWidth, kHeight, global_z};

  const carbon::Stencil stencil_radius1(
      std::vector<carbon::int32_3d>{
          {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}},
      /*remove_center=*/true);
  carbon::Dense dense(sys, global, stencil_radius1);

  // ---- int32 field for Min / Max ----
  auto i_field = dense.newField<std::int32_t, 1>(carbon::MemorySide::HostDevice);
  i_field.fill_owned_host([](carbon::int64_3d gc, int /*c*/) {
    return encode(gc.x, gc.y, gc.z);
  });

  carbon::Stream stream = sys.stream();
  try {
    carbon::Transfer<std::int32_t>(sys, i_field.cpu_buffers(), i_field.gpu_buffers())(stream);
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_dist_reduce_ops[rank={}]: int32 upload failed: {}", rank, e.what());
    return EXIT_FAILURE;
  }

  // Host references over the global volume. encode is deterministic, so every
  // rank computes the same reference.
  std::int32_t host_min = std::numeric_limits<std::int32_t>::max();
  std::int32_t host_max = std::numeric_limits<std::int32_t>::lowest();
  for (int64_t z = 0; z < global_z; ++z) {
    for (int64_t y = 0; y < kHeight; ++y) {
      for (int64_t x = 0; x < kWidth; ++x) {
        host_min = std::min(host_min, encode(x, y, z));
        host_max = std::max(host_max, encode(x, y, z));
      }
    }
  }

  const auto i_loader = [&i_field](carbon::KernelLoader& loader) {
    const auto in = loader(i_field);
    return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> std::int32_t {
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
      CARBON_LOG_ERROR("test_dist_reduce_ops[rank={}]: Min launch failed: {}", rank, e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    if (r.host_value() != host_min) {
      CARBON_LOG_ERROR(
          "test_dist_reduce_ops[rank={}]: Min mismatch: got {}, expected global ref {}", rank,
          r.host_value(), host_min);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dist_reduce_ops[rank={}]: Min = {} (matches global ref)", rank,
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
      CARBON_LOG_ERROR("test_dist_reduce_ops[rank={}]: Max launch failed: {}", rank, e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    if (r.host_value() != host_max) {
      CARBON_LOG_ERROR(
          "test_dist_reduce_ops[rank={}]: Max mismatch: got {}, expected global ref {}", rank,
          r.host_value(), host_max);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dist_reduce_ops[rank={}]: Max = {} (matches global ref)", rank,
                    r.host_value());
  }

  // ---- Prod (device path, NCCL/RCCL ncclProd) over a double field ----
  {
    auto d_field = dense.newField<double, 1>(carbon::MemorySide::HostDevice);
    d_field.fill_owned_host([global_z](carbon::int64_3d gc, int /*c*/) {
      return encode_double(gc.x, gc.y, gc.z, global_z);
    });
    try {
      carbon::Transfer<double>(sys, d_field.cpu_buffers(), d_field.gpu_buffers())(stream);
      stream.wait();
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dist_reduce_ops[rank={}]: double upload failed: {}", rank, e.what());
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
          return [in] CARBON_HOST_DEVICE(carbon::Dense::Span s, carbon::Dense::Idx idx) -> double {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Prod<double>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dist_reduce_ops[rank={}]: Prod launch failed: {}", rank, e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);

    const double got = r.host_value();
    const double rel = std::abs(got - host_prod) / std::abs(host_prod);
    constexpr double kRelTol = 1e-9;
    if (rel > kRelTol) {
      CARBON_LOG_ERROR(
          "test_dist_reduce_ops[rank={}]: Prod mismatch: got {:.10g}, expected {:.10g} "
          "(rel diff {:.3e})",
          rank, got, host_prod, rel);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dist_reduce_ops[rank={}]: Prod = {:.10g} (rel diff {:.3e})", rank, got,
                    rel);
  }

  // ---- Min via host (OpenMP) reduce + MPI_Allreduce ----
  // Locks make_host_reduce_kernel's cross-rank path on a non-Sum op (the
  // host-side fold uses MPI_MIN through the same op_kind trait).
  {
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k = dense.make_host_reduce_kernel<std::int32_t>(
        [&i_field](carbon::KernelLoader& loader) {
          const auto in = loader(i_field);
          return [in](carbon::Dense::Span s, carbon::Dense::Idx idx) -> std::int32_t {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Min<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dist_reduce_ops[rank={}]: host Min launch failed: {}", rank, e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    if (r.host_value() != host_min) {
      CARBON_LOG_ERROR(
          "test_dist_reduce_ops[rank={}]: host Min mismatch: got {}, expected global ref {}", rank,
          r.host_value(), host_min);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dist_reduce_ops[rank={}]: host Min = {} (MPI_Allreduce on host scalar)",
                    rank, r.host_value());
  }

  // ---- Max via host (OpenMP) reduce + MPI_Allreduce ----
  {
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k = dense.make_host_reduce_kernel<std::int32_t>(
        [&i_field](carbon::KernelLoader& loader) {
          const auto in = loader(i_field);
          return [in](carbon::Dense::Span s, carbon::Dense::Idx idx) -> std::int32_t {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Max<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dist_reduce_ops[rank={}]: host Max launch failed: {}", rank, e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    if (r.host_value() != host_max) {
      CARBON_LOG_ERROR(
          "test_dist_reduce_ops[rank={}]: host Max mismatch: got {}, expected global ref {}", rank,
          r.host_value(), host_max);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_dist_reduce_ops[rank={}]: host Max = {} (MPI_Allreduce on host scalar)",
                    rank, r.host_value());
  }

  // ---- Prod via host (OpenMP) reduce + MPI_Allreduce ----
  {
    auto d_field = dense.newField<double, 1>(carbon::MemorySide::Host);
    d_field.fill_owned_host([global_z](carbon::int64_3d gc, int /*c*/) {
      return encode_double(gc.x, gc.y, gc.z, global_z);
    });

    double host_prod = 1.0;
    for (int64_t z = 0; z < global_z; ++z) {
      for (int64_t y = 0; y < kHeight; ++y) {
        for (int64_t x = 0; x < kWidth; ++x) {
          host_prod *= encode_double(x, y, z, global_z);
        }
      }
    }

    carbon::ReductionResult<double> r(sys);
    auto k = dense.make_host_reduce_kernel<double>(
        [&d_field](carbon::KernelLoader& loader) {
          const auto in = loader(d_field);
          return [in](carbon::Dense::Span s, carbon::Dense::Idx idx) -> double {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Prod<double>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_dist_reduce_ops[rank={}]: host Prod launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);

    const double got = r.host_value();
    const double rel = std::abs(got - host_prod) / std::abs(host_prod);
    constexpr double kRelTol = 1e-9;
    if (rel > kRelTol) {
      CARBON_LOG_ERROR(
          "test_dist_reduce_ops[rank={}]: host Prod mismatch: got {:.10g}, expected {:.10g} "
          "(rel diff {:.3e})",
          rank, got, host_prod, rel);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_dist_reduce_ops[rank={}]: host Prod = {:.10g} (MPI_Allreduce on host scalar)", rank,
        got);
  }

  CARBON_LOG_INFO("test_dist_reduce_ops[rank={}]: PASS", rank);
  return EXIT_SUCCESS;
}
