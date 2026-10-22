// Cross-rank Block::Partition::neighbour regression for the connectivity wire-up.
//
// Launched under `mpiexec -n 2 ./test_block_neighbour_dist`. This is the
// neighbour-read complement of @c test_halo_block_dist : that one checked that
// @c HaloBlock delivers the right bytes into the ghost slots; this one checks
// that @c Partition::neighbour can then actually read those ghosts through the
// 27-way connectivity table.
//
// Each rank constructs a distributed Block on an all-active cube domain, fills
// the owned bricks of an int field with encode(global_x, global_y, global_z),
// runs a HaloBlock exchange, and then runs a compute kernel that probes the
// +z neighbour for every active voxel and writes its (value, valid) into a
// pair of output fields. After download we assert that for every voxel whose
// +z neighbour lives on the remote rank (the upper-edge brick row on rank 0,
// the lower-edge brick row on rank 1), @c neighbour(...).valid == true and
// the read value matches @c encode of the remote brick's global coordinates.
//
// The pre-fix bug was that @c block.inl::reserve_row populated
// @c candidate_to_ghost_local for the cross-rank ghost slots correctly, but
// the connectivity rewrite guarded the lookup behind
// @c else if (neigh.gpu_id != kInvalidBlockIdx) — under MPI gpu_count == 1, so
// remote-rank bricks had @c neigh.gpu_id == kInvalidBlockIdx and the
// connectivity slot stayed unset. Halo bytes arrived; @c neighbour was blind
// to them. This test would have flagged that immediately.
//
// Skipped (returns 0) when world_size < 2 or when no CUDA devices are visible.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/distributed.h>
#include <carbon/halo_block.h>
#include <carbon/kernel.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace {

[[nodiscard]] auto
sync_all(const carbon::Stream& stream,
         const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_block_neighbour_dist: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<int, 1>;
using FlagFieldT = BlockGrid::Field<int, 1>;

[[nodiscard]] CARBON_HD constexpr auto
encode(int gx,
       int gy,
       int gz) noexcept -> int {
  return gz * 1000000 + gy * 1000 + gx;
}

constexpr int kSentinel = -9999999;

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_block_neighbour_dist: world_size={} < 2, skipping",
                    distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_block_neighbour_dist: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR(
        "test_block_neighbour_dist: expected gpu_group size 1 (one rank per GPU), got {}",
        sys.gpu_group().size());
    return 1;
  }

  const int64_t bricks_per_dim = 4;
  const int64_t domain = bricks_per_dim * static_cast<int64_t>(SB::MemX);
  const carbon::int64_3d global{domain, domain, domain};
  const auto predicate = [](const carbon::int64_3d&) -> bool {
    return true;
  };

  BlockGrid block(sys, global, predicate, "neighbour_dist");
  if (block.gpu_count() != 1U) {
    CARBON_LOG_ERROR("test_block_neighbour_dist: expected gpu_count 1, got {}", block.gpu_count());
    return 1;
  }

  FieldT u = block.template newField<int, 1>(carbon::MemorySide::Device, "u");
  FieldT zp_val = block.template newField<int, 1>(carbon::MemorySide::Device, "zp_val");
  FlagFieldT zp_valid = block.template newField<int, 1>(carbon::MemorySide::Device, "zp_valid");
  carbon::Stream stream = sys.stream();

  const uint32_t n_owned = block.block_count(0);
  const uint32_t n_total = block.storage_block_count(0);
  const size_t total_elements = static_cast<size_t>(n_total) * static_cast<size_t>(SB::VoxelCount);

  // Upload encode() into owned bricks; ghosts stay at sentinel so we can prove
  // the halo overwrote them.
  auto host_u = carbon::Cpu::buffer<int>(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    host_u.data()[i] = kSentinel;
  }
  for (uint32_t b = 0; b < n_owned; ++b) {
    const auto origin = block.host_origins(0)[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                             static_cast<size_t>(x) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                 static_cast<size_t>(z);
          host_u.data()[lin] =
              encode(static_cast<int>(origin.x) + x, static_cast<int>(origin.y) + y,
                     static_cast<int>(origin.z) + z);
        }
      }
    }
  }
  try {
    sys.async_copy(host_u, u.gpu_buffers()[0], total_elements * sizeof(int), stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_block_neighbour_dist: upload failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "wait")) {
    return 1;
  }

  // Halo exchange.
  carbon::HaloBlock<int, 1> halo(block, u);
  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_block_neighbour_dist: halo failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "wait")) {
    return 1;
  }

  // Kernel: for every active voxel, read the +z neighbour and write
  // (value, valid) into zp_val / zp_valid.
  carbon::Kernel probe = block.make_kernel([u, zp_val, zp_valid](carbon::KernelLoader& loader) {
    const auto in = loader.read_stencil(u);
    const auto out_val = loader.write(zp_val);
    const auto out_valid = loader.write(zp_valid);
    return [in, out_val, out_valid] CARBON_HD(Span s, Idx idx) -> void {
      const auto zp = in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
      out_val.write(s, idx, 0, zp.valid ? zp.value : kSentinel);
      out_valid.write(s, idx, 0, zp.valid ? 1 : 0);
    };
  });
  try {
    probe.launch(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_block_neighbour_dist: probe failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "wait")) {
    return 1;
  }

  // Download results.
  auto host_val = carbon::Cpu::buffer<int>(total_elements);
  auto host_valid = carbon::Cpu::buffer<int>(total_elements);
  try {
    sys.async_copy(zp_val.gpu_buffers()[0], host_val, total_elements * sizeof(int), stream);
    sys.async_copy(zp_valid.gpu_buffers()[0], host_valid, total_elements * sizeof(int), stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_block_neighbour_dist: download failed: {}", e.what());
    return 1;
  }
  if (!sync_all(stream, "wait")) {
    return 1;
  }

  // Verification. For every active voxel (gx, gy, gz) on this rank, the +z
  // neighbour lives at (gx, gy, gz+1). It's "remote" when gz is the rank's
  // last z-row and there's a rank above (rank + 1 < world_size). The pre-fix
  // bug made those cross-rank neighbours show valid==false; the assertion
  // here is exactly: every cross-rank +z probe must be valid AND equal to
  // encode(gx, gy, gz+1).
  size_t mismatches = 0;
  size_t cross_rank_probes = 0;
  for (uint32_t b = 0; b < n_owned; ++b) {
    const auto origin = block.host_origins(0)[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      const int64_t gz = static_cast<int64_t>(origin.z) + z;
      const bool is_top_of_domain = (gz + 1 == domain);
      const bool is_top_of_my_slab = (gz + 1 == block.bz_end(0) * SB::MemZ);
      const bool is_cross_rank = is_top_of_my_slab && !is_top_of_domain;
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          const int64_t gx = static_cast<int64_t>(origin.x) + x;
          const int64_t gy = static_cast<int64_t>(origin.y) + y;
          const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                             static_cast<size_t>(x) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                 static_cast<size_t>(z);
          const int got_val = host_val.data()[lin];
          const int got_valid = host_valid.data()[lin];
          if (is_top_of_domain) {
            if (got_valid != 0) {
              ++mismatches;
              if (mismatches < 4U) {
                CARBON_LOG_ERROR(
                    "test_block_neighbour_dist: (gx={},gy={},gz={}): top of domain should be "
                    "invalid, got valid=1 val={} (rank {})",
                    gx, gy, gz, got_val, rank);
              }
            }
            continue;
          }
          const int want =
              encode(static_cast<int>(gx), static_cast<int>(gy), static_cast<int>(gz + 1));
          if (got_valid != 1 || got_val != want) {
            ++mismatches;
            if (mismatches < 4U) {
              CARBON_LOG_ERROR(
                  "test_block_neighbour_dist: (gx={},gy={},gz={}): +z neighbour got "
                  "(valid={},val={}) want (valid=1,val={}) cross_rank={} (rank {})",
                  gx, gy, gz, got_valid, got_val, want, is_cross_rank ? "yes" : "no", rank);
            }
          }
          if (is_cross_rank) {
            ++cross_rank_probes;
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_block_neighbour_dist: rank {} {} mismatches over {} cross-rank probes",
                     rank, mismatches, cross_rank_probes);
    return 1;
  }
  if (cross_rank_probes == 0U && rank + 1 < world_size) {
    CARBON_LOG_ERROR(
        "test_block_neighbour_dist: rank {} expected cross-rank probes but found none "
        "(bz_end={}, MemZ={}, domain={})",
        rank, block.bz_end(0), SB::MemZ, domain);
    return 1;
  }
  CARBON_LOG_INFO(
      "test_block_neighbour_dist: rank {} of {} ok ({} cross-rank +z neighbour probes verified)",
      rank, world_size, cross_rank_probes);
  return 0;
}
