// Cross-rank Block Min / Max / Prod reduction test (one rank per GPU).
//
// Closes P1.γ of docs/audit_2026_05_fourth_pass.md (sibling gap to the third
// pass's `test_dist_reduce_ops.cu`). The Sum-only Block cross-rank reduce path
// is locked by `test_block_dist_reduce.cu`; this file extends the matrix to
// {Min, Max, Prod} so a regression in `BlockReduceKernelImpl`'s stage-1 / 2
// wiring that fans into the same stage-3 NCCL/RCCL allreduce — or in
// `reduce::op_kind<Op>::value` once it travels through the block dispatch — is
// no longer silent.
//
// Launched under `mpiexec -n 2 ./test_block_dist_reduce_ops`. Same setup as
// the Sum-only sibling: every voxel active so the host reference reduces to a
// closed-form value over the entire global volume, and a global Z extent that
// gives at least one full brick row per rank. Same encoders as
// `test_dist_reduce_ops.cu` so a single host reference works on every rank
// without coordination.
//
// Cases covered:
//   1. Min  device — `make_reduce_kernel` + `reduce::Min` over an int32 field;
//                    locks ncclMin / MPI_MIN routing through the block
//                    dispatch path.
//   2. Max  device — same, `reduce::Max` (ncclMax / MPI_MAX).
//   3. Prod device — `make_reduce_kernel` + `reduce::Prod` over a double
//                    field with values slightly above 1.0 to keep the running
//                    product in finite range.
//   4. Min  host   — `make_host_reduce_kernel` + `reduce::Min` to lock the
//                    cross-rank `MPI_Allreduce(MPI_MIN)` path on a non-Sum op
//                    on the block host launcher as well. Field is built with
//                    `MemorySide::HostDevice`; the staging loop populates both
//                    `cpu_buffers()[0]` and the staging buffer that feeds the
//                    device side, mirroring the pattern in
//                    `test_block_host_parity.cu`.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are
// visible — the rank-local Block reduce path already has dedicated non-MPI
// coverage in `test_block_reduce.cu`.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/distributed.h>
#include <carbon/integer3d.h>
#include <carbon/kernel.h>
#include <carbon/logging.h>
#include <carbon/memory_side.h>
#include <carbon/reduce.h>
#include <carbon/reduce_dist.h>
#include <carbon/system.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;

[[nodiscard]] CARBON_HD auto
encode_int(int64_t gx,
           int64_t gy,
           int64_t gz) -> std::int32_t {
  // Strictly positive, deterministic — matches the shape used by
  // test_dist_reduce_ops's encode().
  return static_cast<std::int32_t>(gz * 100 + gy * 10 + gx + 1);
}

[[nodiscard]] CARBON_HD auto
encode_double(int64_t gx,
              int64_t gy,
              int64_t gz,
              int64_t global_z,
              int64_t W,
              int64_t H) -> double {
  return 1.0 +
         (static_cast<double>(gx + gy * W + gz * W * H) / static_cast<double>(W * H * global_z)) *
             0.04;
}

// Stage owned-brick data into both the host mirror and a contiguous staging
// buffer, then async_copy the staging buffer to the device side of the field.
// Returns false on async_copy failure. Mirrors the pattern in
// test_block_host_parity / test_block_dist_reduce, with the addition that the
// host mirror is filled in lock-step (so make_host_reduce_kernel below sees
// the same values without an extra device->host transfer).
template <typename T,
          typename Encoder>
[[nodiscard]] auto
seed_block_field(carbon::System& sys,
                 const BlockGrid& block,
                 BlockGrid::Field<T,
                                  1>& field,
                 carbon::Stream& stream,
                 Encoder enc) -> bool {
  const std::uint32_t n_owned = block.block_count(0);
  const std::uint32_t n_storage = block.storage_block_count(0);
  const size_t total_slots = static_cast<size_t>(n_storage) * static_cast<size_t>(SB::VoxelCount);
  if (total_slots == 0U) {
    return true;
  }
  auto host_buf = carbon::Cpu::buffer<T>(total_slots);
  std::vector<T> stage(total_slots, T{0});
  for (std::uint32_t b = 0; b < n_owned; ++b) {
    const auto& origin = block.host_origins(0)[b];
    const auto& bm = block.host_bitmasks(0)[b];
    for (int iz = 0; iz < SB::MemZ; ++iz) {
      for (int iy = 0; iy < SB::MemY; ++iy) {
        for (int ix = 0; ix < SB::MemX; ++ix) {
          if (!bm.is_active(ix, iy, iz)) {
            continue;
          }
          const int64_t gx = static_cast<int64_t>(origin.x) + ix;
          const int64_t gy = static_cast<int64_t>(origin.y) + iy;
          const int64_t gz = static_cast<int64_t>(origin.z) + iz;
          const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                             static_cast<size_t>(ix) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(iy) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                 static_cast<size_t>(iz);
          stage[lin] = enc(gx, gy, gz);
        }
      }
    }
  }
  // Mirror the staging buffer into both the host_buf (-> async_copy source)
  // and the field's CPU mirror (read by make_host_reduce_kernel below).
  std::memcpy(host_buf.data(), stage.data(), total_slots * sizeof(T));
  std::memcpy(field.cpu_buffers()[0].data(), stage.data(), total_slots * sizeof(T));
  try {
    sys.async_copy(host_buf, field.gpu_buffers()[0], total_slots * sizeof(T), stream);
    stream.wait();
  } catch (const std::exception&) {
    return false;
  }
  return true;
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_block_dist_reduce_ops: world_size={} < 2, skipping",
                    distributed->world_size());
    return EXIT_SUCCESS;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_block_dist_reduce_ops: no CUDA devices, skipping");
    return EXIT_SUCCESS;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR(
        "test_block_dist_reduce_ops: expected gpu_group size 1 (one rank per GPU), got {}",
        sys.gpu_group().size());
    return EXIT_FAILURE;
  }

  const int64_t global_z = static_cast<int64_t>(world_size) * SB::MemZ * 2;
  const carbon::int64_3d global{static_cast<int64_t>(SB::MemX) * 2,
                                static_cast<int64_t>(SB::MemY) * 2, global_z};

  const auto predicate = [](const carbon::int64_3d&) -> bool {
    return true;
  };
  BlockGrid block(sys, global, predicate, "dist_block_reduce_ops");
  if (block.gpu_count() != 1U) {
    CARBON_LOG_ERROR("test_block_dist_reduce_ops[rank={}]: gpu_count == {} (expected 1)", rank,
                     block.gpu_count());
    return EXIT_FAILURE;
  }

  carbon::Stream stream = sys.stream();

  // ---- int32 field, seeded on host + uploaded to device ----
  auto i_field =
      block.template newField<std::int32_t, 1>(carbon::MemorySide::HostDevice, "i_field");
  if (!seed_block_field(sys, block, i_field, stream,
                        [](int64_t gx, int64_t gy, int64_t gz) -> std::int32_t {
                          return encode_int(gx, gy, gz);
                        })) {
    CARBON_LOG_ERROR("test_block_dist_reduce_ops[rank={}]: int32 seed/upload failed", rank);
    return EXIT_FAILURE;
  }

  // Host references — every rank computes the same value because every voxel
  // is active and the encoder is deterministic.
  std::int32_t host_min = std::numeric_limits<std::int32_t>::max();
  std::int32_t host_max = std::numeric_limits<std::int32_t>::lowest();
  for (int64_t z = 0; z < global.z; ++z) {
    for (int64_t y = 0; y < global.y; ++y) {
      for (int64_t x = 0; x < global.x; ++x) {
        host_min = std::min(host_min, encode_int(x, y, z));
        host_max = std::max(host_max, encode_int(x, y, z));
      }
    }
  }

  const auto i_loader = [&i_field](carbon::KernelLoader& loader) {
    const auto in = loader.read(i_field);
    return [in] CARBON_HD(Span s, Idx idx) -> std::int32_t {
      return in.read(s, idx, 0);
    };
  };

  // ---- Min (device path, NCCL/RCCL ncclMin) ----
  {
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k = block.template make_reduce_kernel<std::int32_t>(i_loader, r,
                                                             carbon::reduce::Min<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_block_dist_reduce_ops[rank={}]: Min launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    stream.wait();
    if (r.host_value() != host_min) {
      CARBON_LOG_ERROR(
          "test_block_dist_reduce_ops[rank={}]: Min mismatch: got {}, expected global ref {}", rank,
          r.host_value(), host_min);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_block_dist_reduce_ops[rank={}]: Min = {} (matches global ref)", rank,
                    r.host_value());
  }

  // ---- Max (device path, NCCL/RCCL ncclMax) ----
  {
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k = block.template make_reduce_kernel<std::int32_t>(i_loader, r,
                                                             carbon::reduce::Max<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_block_dist_reduce_ops[rank={}]: Max launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    stream.wait();
    if (r.host_value() != host_max) {
      CARBON_LOG_ERROR(
          "test_block_dist_reduce_ops[rank={}]: Max mismatch: got {}, expected global ref {}", rank,
          r.host_value(), host_max);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_block_dist_reduce_ops[rank={}]: Max = {} (matches global ref)", rank,
                    r.host_value());
  }

  // ---- Prod (device path, NCCL/RCCL ncclProd) over a double field ----
  {
    auto d_field = block.template newField<double, 1>(carbon::MemorySide::HostDevice, "d_field");
    if (!seed_block_field(sys, block, d_field, stream,
                          [global_z, &global](int64_t gx, int64_t gy, int64_t gz) -> double {
                            return encode_double(gx, gy, gz, global_z, global.x, global.y);
                          })) {
      CARBON_LOG_ERROR("test_block_dist_reduce_ops[rank={}]: double seed/upload failed", rank);
      return EXIT_FAILURE;
    }

    double host_prod = 1.0;
    for (int64_t z = 0; z < global.z; ++z) {
      for (int64_t y = 0; y < global.y; ++y) {
        for (int64_t x = 0; x < global.x; ++x) {
          host_prod *= encode_double(x, y, z, global_z, global.x, global.y);
        }
      }
    }

    carbon::ReductionResult<double> r(sys);
    auto k = block.template make_reduce_kernel<double>(
        [&d_field](carbon::KernelLoader& loader) {
          const auto in = loader.read(d_field);
          return [in] CARBON_HD(Span s, Idx idx) -> double {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Prod<double>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_block_dist_reduce_ops[rank={}]: Prod launch failed: {}", rank,
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
          "test_block_dist_reduce_ops[rank={}]: Prod mismatch: got {:.10g}, expected {:.10g} "
          "(rel diff {:.3e})",
          rank, got, host_prod, rel);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO("test_block_dist_reduce_ops[rank={}]: Prod = {:.10g} (rel diff {:.3e})", rank,
                    got, rel);
  }

  // ---- Min via host (OpenMP) reduce + MPI_Allreduce ----
  // Locks make_host_reduce_kernel's cross-rank path on a non-Sum op for the
  // block host launcher. Reuses i_field's host mirror — `seed_block_field`
  // populated cpu_buffers()[0] in lock-step with the device upload, so the
  // host launcher reads the same values as the host reference.
  {
    carbon::ReductionResult<std::int32_t> r(sys);
    auto k = block.template make_host_reduce_kernel<std::int32_t>(
        [&i_field](carbon::KernelLoader& loader) {
          const auto in = loader.read(i_field);
          return [in](Span s, Idx idx) -> std::int32_t {
            return in.read(s, idx, 0);
          };
        },
        r, carbon::reduce::Min<std::int32_t>{});
    try {
      k.launch(stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_block_dist_reduce_ops[rank={}]: host Min launch failed: {}", rank,
                       e.what());
      return EXIT_FAILURE;
    }
    r.sync(stream);
    stream.wait();
    if (r.host_value() != host_min) {
      CARBON_LOG_ERROR(
          "test_block_dist_reduce_ops[rank={}]: host Min mismatch: got {}, expected global ref {}",
          rank, r.host_value(), host_min);
      return EXIT_FAILURE;
    }
    CARBON_LOG_INFO(
        "test_block_dist_reduce_ops[rank={}]: host Min = {} (MPI_Allreduce on host scalar)", rank,
        r.host_value());
  }

  CARBON_LOG_INFO("test_block_dist_reduce_ops[rank={}]: PASS", rank);
  return EXIT_SUCCESS;
}
