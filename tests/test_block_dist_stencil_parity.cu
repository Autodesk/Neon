// MPI Block cross-rank stencil parity under a sparse (sphere) predicate.
//
// Launched under `mpiexec -n 2 ./test_block_dist_stencil_parity`. Closes the
// coverage gap left by test_block_neighbour_dist (all-active cube, +z only)
// and test_skeleton_dist_overlap_block (all-active, OCC matrix only): the
// jacobi_unified examples exercise a sphere predicate, and the historical
// Block+MPI connectivity bug surfaced there first.
//
// Part 1 — six-direction neighbour probe after HaloBlock on cross-rank slab
// boundary cells that are active under the sphere predicate.
//
// Part 2 — ping-pong Jacobi (Skeleton auto-halo, OverlapMode::None) for a
// fixed sweep count; rank 0 compares the distributed sum|u|^2 reduction to a
// host-computed global reference that mirrors the device Jacobi body exactly.
//
// Skipped (returns 0) when world_size < 2 or no CUDA devices are visible.

#include <carbon/block.h>
#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/cuda_annotation.h>
#include <carbon/distributed.h>
#include <carbon/halo_block.h>
#include <carbon/kernel.h>
#include <carbon/reduce.h>
#include <carbon/skeleton.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace {

using SB = carbon::detail::block::DefaultStaticBlock;
using BlockGrid = carbon::Block<SB>;
using Span = BlockGrid::Span;
using Idx = BlockGrid::Idx;
using FieldT = BlockGrid::Field<float, 1>;
using FlagFieldT = BlockGrid::Field<int, 1>;

constexpr int64_t kDomain = 32;
constexpr int kJacobiSweeps = 20;
constexpr double kSumTol = 1.0e-4;

[[nodiscard]] auto
sync_all(const carbon::Stream& stream,
         const char* tag) -> bool {
  try {
    stream.wait();
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_block_dist_stencil_parity: sync after {} failed: {}", tag, e.what());
    return false;
  }
  return true;
}

[[nodiscard]] auto
sphere_active(int64_t domain_n,
              int64_t gx,
              int64_t gy,
              int64_t gz) -> bool {
  const double cx = (static_cast<double>(domain_n) - 1.0) * 0.5;
  const double cy = (static_cast<double>(domain_n) - 1.0) * 0.5;
  const double cz = (static_cast<double>(domain_n) - 1.0) * 0.5;
  const double r = static_cast<double>(domain_n) * 0.42;
  const double dx = static_cast<double>(gx) - cx;
  const double dy = static_cast<double>(gy) - cy;
  const double dz = static_cast<double>(gz) - cz;
  return (dx * dx + dy * dy + dz * dz) <= (r * r);
}

[[nodiscard]] auto
sphere_predicate(int64_t domain_n) -> std::function<bool(const carbon::int64_3d&)> {
  return [domain_n](const carbon::int64_3d& gc) -> bool {
    return sphere_active(domain_n, gc.x, gc.y, gc.z);
  };
}

[[nodiscard]] CARBON_HD constexpr auto
encode_f(int gx,
         int gy,
         int gz) noexcept -> float {
  return static_cast<float>(gz * 1000000 + gy * 1000 + gx);
}

[[nodiscard]] auto
run_six_direction_probe(BlockGrid& block,
                        carbon::System& sys,
                        int rank,
                        int world_size) -> bool {
  FieldT u = block.template newField<float, 1>(carbon::MemorySide::Device, "u");
  FieldT fv0 = block.template newField<float, 1>(carbon::MemorySide::Device, "fv0");
  FieldT fv1 = block.template newField<float, 1>(carbon::MemorySide::Device, "fv1");
  FieldT fv2 = block.template newField<float, 1>(carbon::MemorySide::Device, "fv2");
  FieldT fv3 = block.template newField<float, 1>(carbon::MemorySide::Device, "fv3");
  FieldT fv4 = block.template newField<float, 1>(carbon::MemorySide::Device, "fv4");
  FieldT fv5 = block.template newField<float, 1>(carbon::MemorySide::Device, "fv5");
  FlagFieldT fd0 = block.template newField<int, 1>(carbon::MemorySide::Device, "fd0");
  FlagFieldT fd1 = block.template newField<int, 1>(carbon::MemorySide::Device, "fd1");
  FlagFieldT fd2 = block.template newField<int, 1>(carbon::MemorySide::Device, "fd2");
  FlagFieldT fd3 = block.template newField<int, 1>(carbon::MemorySide::Device, "fd3");
  FlagFieldT fd4 = block.template newField<int, 1>(carbon::MemorySide::Device, "fd4");
  FlagFieldT fd5 = block.template newField<int, 1>(carbon::MemorySide::Device, "fd5");
  carbon::Stream stream = sys.stream();

  const uint32_t n_owned = block.block_count(0);
  const uint32_t n_total = block.storage_block_count(0);
  const size_t total_elements = static_cast<size_t>(n_total) * static_cast<size_t>(SB::VoxelCount);

  auto host_u = carbon::Cpu::buffer<float>(total_elements);
  for (size_t i = 0; i < total_elements; ++i) {
    host_u.data()[i] = -1.0F;
  }
  for (uint32_t b = 0; b < n_owned; ++b) {
    const auto origin = block.host_origins(0)[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          const int64_t gx = origin.x + x;
          const int64_t gy = origin.y + y;
          const int64_t gz = origin.z + z;
          if (!sphere_active(kDomain, gx, gy, gz)) {
            continue;
          }
          const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                             static_cast<size_t>(x) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                 static_cast<size_t>(z);
          host_u.data()[lin] =
              encode_f(static_cast<int>(gx), static_cast<int>(gy), static_cast<int>(gz));
        }
      }
    }
  }
  try {
    sys.async_copy(host_u, u.gpu_buffers()[0], total_elements * sizeof(float), stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_block_dist_stencil_parity: probe upload failed: {}", e.what());
    return false;
  }
  if (!sync_all(stream, "probe_upload")) {
    return false;
  }

  carbon::HaloBlock<float, 1> halo(block, u);
  try {
    halo(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_block_dist_stencil_parity: probe halo failed: {}", e.what());
    return false;
  }
  if (!sync_all(stream, "probe_halo")) {
    return false;
  }

  static constexpr std::array<carbon::int8_3d, 6> kDirs{
      carbon::int8_3d{1, 0, 0},  carbon::int8_3d{-1, 0, 0}, carbon::int8_3d{0, 1, 0},
      carbon::int8_3d{0, -1, 0}, carbon::int8_3d{0, 0, 1},  carbon::int8_3d{0, 0, -1}};

  carbon::Kernel probe = block.make_kernel([u, fv0, fv1, fv2, fv3, fv4, fv5, fd0, fd1, fd2, fd3,
                                            fd4, fd5](carbon::KernelLoader& loader) {
    const auto in = loader.read_stencil(u);
    const auto out_fv0 = loader.write(fv0);
    const auto out_fv1 = loader.write(fv1);
    const auto out_fv2 = loader.write(fv2);
    const auto out_fv3 = loader.write(fv3);
    const auto out_fv4 = loader.write(fv4);
    const auto out_fv5 = loader.write(fv5);
    const auto out_fd0 = loader.write(fd0);
    const auto out_fd1 = loader.write(fd1);
    const auto out_fd2 = loader.write(fd2);
    const auto out_fd3 = loader.write(fd3);
    const auto out_fd4 = loader.write(fd4);
    const auto out_fd5 = loader.write(fd5);
    return [in, out_fv0, out_fv1, out_fv2, out_fv3, out_fv4, out_fv5, out_fd0, out_fd1, out_fd2,
            out_fd3, out_fd4, out_fd5] CARBON_HD(Span s, Idx idx) -> void {
      const auto n0 = in.neighbour(s, idx, kDirs[0], 0);
      const auto n1 = in.neighbour(s, idx, kDirs[1], 0);
      const auto n2 = in.neighbour(s, idx, kDirs[2], 0);
      const auto n3 = in.neighbour(s, idx, kDirs[3], 0);
      const auto n4 = in.neighbour(s, idx, kDirs[4], 0);
      const auto n5 = in.neighbour(s, idx, kDirs[5], 0);
      out_fv0.write(s, idx, 0, n0.valid ? n0.value : -1.0F);
      out_fv1.write(s, idx, 0, n1.valid ? n1.value : -1.0F);
      out_fv2.write(s, idx, 0, n2.valid ? n2.value : -1.0F);
      out_fv3.write(s, idx, 0, n3.valid ? n3.value : -1.0F);
      out_fv4.write(s, idx, 0, n4.valid ? n4.value : -1.0F);
      out_fv5.write(s, idx, 0, n5.valid ? n5.value : -1.0F);
      out_fd0.write(s, idx, 0, n0.valid ? 1 : 0);
      out_fd1.write(s, idx, 0, n1.valid ? 1 : 0);
      out_fd2.write(s, idx, 0, n2.valid ? 1 : 0);
      out_fd3.write(s, idx, 0, n3.valid ? 1 : 0);
      out_fd4.write(s, idx, 0, n4.valid ? 1 : 0);
      out_fd5.write(s, idx, 0, n5.valid ? 1 : 0);
    };
  });
  try {
    probe.launch(stream);
  } catch (const std::exception& e) {
    CARBON_LOG_ERROR("test_block_dist_stencil_parity: probe kernel failed: {}", e.what());
    return false;
  }
  if (!sync_all(stream, "probe_kernel")) {
    return false;
  }

  std::array<std::vector<float>, 6> host_vals{};
  std::array<std::vector<int>, 6> host_valid{};
  std::array<FieldT*, 6> face_val_ptrs{&fv0, &fv1, &fv2, &fv3, &fv4, &fv5};
  std::array<FlagFieldT*, 6> face_valid_ptrs{&fd0, &fd1, &fd2, &fd3, &fd4, &fd5};
  for (int f = 0; f < 6; ++f) {
    host_vals[static_cast<size_t>(f)].resize(total_elements);
    host_valid[static_cast<size_t>(f)].resize(total_elements);
    auto hv = carbon::Cpu::buffer<float>(total_elements);
    auto hd = carbon::Cpu::buffer<int>(total_elements);
    try {
      sys.async_copy(face_val_ptrs[static_cast<size_t>(f)]->gpu_buffers()[0], hv,
                     total_elements * sizeof(float), stream);
      sys.async_copy(face_valid_ptrs[static_cast<size_t>(f)]->gpu_buffers()[0], hd,
                     total_elements * sizeof(int), stream);
    } catch (const std::exception& e) {
      CARBON_LOG_ERROR("test_block_dist_stencil_parity: probe download failed: {}", e.what());
      return false;
    }
    if (!sync_all(stream, "probe_download")) {
      return false;
    }
    for (size_t i = 0; i < total_elements; ++i) {
      host_vals[static_cast<size_t>(f)][i] = hv.data()[i];
      host_valid[static_cast<size_t>(f)][i] = hd.data()[i];
    }
  }

  const int64_t slab_z_lo = static_cast<int64_t>(block.bz_begin(0)) * SB::MemZ;
  const int64_t slab_z_hi = static_cast<int64_t>(block.bz_end(0)) * SB::MemZ;
  size_t mismatches = 0;
  size_t cross_rank_checks = 0;

  for (uint32_t b = 0; b < n_owned; ++b) {
    const auto origin = block.host_origins(0)[b];
    for (int z = 0; z < SB::MemZ; ++z) {
      for (int y = 0; y < SB::MemY; ++y) {
        for (int x = 0; x < SB::MemX; ++x) {
          const int64_t gx = origin.x + x;
          const int64_t gy = origin.y + y;
          const int64_t gz = origin.z + z;
          if (!sphere_active(kDomain, gx, gy, gz)) {
            continue;
          }
          const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(SB::VoxelCount) +
                             static_cast<size_t>(x) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(y) +
                             static_cast<size_t>(SB::MemX) * static_cast<size_t>(SB::MemY) *
                                 static_cast<size_t>(z);
          const bool on_slab_top = (gz + 1 == slab_z_hi) && (gz + 1 < kDomain);
          const bool on_slab_bottom = (gz == slab_z_lo) && (gz > 0);
          if (!on_slab_top && !on_slab_bottom) {
            continue;
          }
          for (int f = 0; f < 6; ++f) {
            const int64_t ngx = gx + static_cast<int64_t>(kDirs[static_cast<size_t>(f)].x);
            const int64_t ngy = gy + static_cast<int64_t>(kDirs[static_cast<size_t>(f)].y);
            const int64_t ngz = gz + static_cast<int64_t>(kDirs[static_cast<size_t>(f)].z);
            if (ngx < 0 || ngx >= kDomain || ngy < 0 || ngy >= kDomain || ngz < 0 ||
                ngz >= kDomain) {
              continue;
            }
            const bool n_active = sphere_active(kDomain, ngx, ngy, ngz);
            const bool cross_rank_z = (kDirs[static_cast<size_t>(f)].z != 0) &&
                                      ((on_slab_top && kDirs[static_cast<size_t>(f)].z > 0) ||
                                       (on_slab_bottom && kDirs[static_cast<size_t>(f)].z < 0));
            if (!n_active) {
              continue;
            }
            const float want =
                encode_f(static_cast<int>(ngx), static_cast<int>(ngy), static_cast<int>(ngz));
            const int got_valid = host_valid[static_cast<size_t>(f)][lin];
            const float got_val = host_vals[static_cast<size_t>(f)][lin];
            if (got_valid != 1 || got_val != want) {
              ++mismatches;
              if (mismatches < 4U) {
                CARBON_LOG_ERROR(
                    "test_block_dist_stencil_parity: rank {} face {} at ({},{},{}) got "
                    "(valid={},val={}) want (valid=1,val={}) cross_rank={}",
                    rank, f, gx, gy, gz, got_valid, got_val, want, cross_rank_z ? "yes" : "no");
              }
            }
            if (cross_rank_z) {
              ++cross_rank_checks;
            }
          }
        }
      }
    }
  }

  if (mismatches > 0U) {
    CARBON_LOG_ERROR("test_block_dist_stencil_parity: rank {} {} six-dir mismatches", rank,
                     mismatches);
    return false;
  }
  if (cross_rank_checks == 0U && rank + 1 < world_size) {
    CARBON_LOG_ERROR(
        "test_block_dist_stencil_parity: rank {} expected cross-rank z probes but found none",
        rank);
    return false;
  }
  CARBON_LOG_INFO(
      "test_block_dist_stencil_parity: rank {} six-dir probe ok ({} cross-rank z checks)", rank,
      cross_rank_checks);
  return true;
}

[[nodiscard]] auto
host_neighbour_valid(int64_t gx,
                     int64_t gy,
                     int64_t gz,
                     int dx,
                     int dy,
                     int dz) -> bool {
  const int64_t nx = gx + dx;
  const int64_t ny = gy + dy;
  const int64_t nz = gz + dz;
  if (nx < 0 || nx >= kDomain || ny < 0 || ny >= kDomain || nz < 0 || nz >= kDomain) {
    return false;
  }
  return sphere_active(kDomain, nx, ny, nz);
}

auto
host_jacobi_step(const std::vector<float>& in,
                 std::vector<float>& out) -> void {
  out.assign(in.size(), 0.0F);
  for (int64_t gz = 0; gz < kDomain; ++gz) {
    for (int64_t gy = 0; gy < kDomain; ++gy) {
      for (int64_t gx = 0; gx < kDomain; ++gx) {
        const size_t idx = static_cast<size_t>(gz * kDomain * kDomain + gy * kDomain + gx);
        if (!sphere_active(kDomain, gx, gy, gz)) {
          continue;
        }
        if (!host_neighbour_valid(gx, gy, gz, 0, 0, -1)) {
          out[idx] = +1.0F;
          continue;
        }
        if (!host_neighbour_valid(gx, gy, gz, 0, 0, 1)) {
          out[idx] = -1.0F;
          continue;
        }
        if (!host_neighbour_valid(gx, gy, gz, 1, 0, 0) ||
            !host_neighbour_valid(gx, gy, gz, -1, 0, 0) ||
            !host_neighbour_valid(gx, gy, gz, 0, 1, 0) ||
            !host_neighbour_valid(gx, gy, gz, 0, -1, 0)) {
          out[idx] = 0.0F;
          continue;
        }
        const float sum =
            in[static_cast<size_t>((gz)*kDomain * kDomain + (gy)*kDomain + (gx + 1))] +
            in[static_cast<size_t>((gz)*kDomain * kDomain + (gy)*kDomain + (gx - 1))] +
            in[static_cast<size_t>((gz)*kDomain * kDomain + (gy + 1) * kDomain + gx)] +
            in[static_cast<size_t>((gz)*kDomain * kDomain + (gy - 1) * kDomain + gx)] +
            in[static_cast<size_t>((gz + 1) * kDomain * kDomain + gy * kDomain + gx)] +
            in[static_cast<size_t>((gz - 1) * kDomain * kDomain + gy * kDomain + gx)];
        out[idx] = sum * (1.0F / 6.0F);
      }
    }
  }
}

[[nodiscard]] auto
host_reference_sumsq() -> double {
  const size_t n =
      static_cast<size_t>(kDomain) * static_cast<size_t>(kDomain) * static_cast<size_t>(kDomain);
  std::vector<float> u(n, 0.0F);
  std::vector<float> u_next(n, 0.0F);
  const int cycles = kJacobiSweeps / 2;
  for (int c = 0; c < cycles; ++c) {
    host_jacobi_step(u, u_next);
    host_jacobi_step(u_next, u);
  }
  double sumsq = 0.0;
  for (int64_t gz = 0; gz < kDomain; ++gz) {
    for (int64_t gy = 0; gy < kDomain; ++gy) {
      for (int64_t gx = 0; gx < kDomain; ++gx) {
        if (!sphere_active(kDomain, gx, gy, gz)) {
          continue;
        }
        const size_t idx = static_cast<size_t>(gz * kDomain * kDomain + gy * kDomain + gx);
        const double v = static_cast<double>(u[idx]);
        sumsq += v * v;
      }
    }
  }
  return sumsq;
}

[[nodiscard]] auto
build_jacobi_step(BlockGrid& block,
                  FieldT& fin,
                  FieldT& fout,
                  std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      block,
      [&fin, &fout](carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(fin);
        const auto out = loader.write(fout);
        return [in, out] CARBON_HD(Span s, Idx idx) -> void {
          const auto xp = in.neighbour(s, idx, carbon::int8_3d{1, 0, 0}, 0);
          const auto xm = in.neighbour(s, idx, carbon::int8_3d{-1, 0, 0}, 0);
          const auto yp = in.neighbour(s, idx, carbon::int8_3d{0, 1, 0}, 0);
          const auto ym = in.neighbour(s, idx, carbon::int8_3d{0, -1, 0}, 0);
          const auto zp = in.neighbour(s, idx, carbon::int8_3d{0, 0, 1}, 0);
          const auto zm = in.neighbour(s, idx, carbon::int8_3d{0, 0, -1}, 0);
          if (!zm.valid) {
            out.write(s, idx, 0, +1.0F);
            return;
          }
          if (!zp.valid) {
            out.write(s, idx, 0, -1.0F);
            return;
          }
          if (!(xp.valid && xm.valid && yp.valid && ym.valid)) {
            out.write(s, idx, 0, 0.0F);
            return;
          }
          const float sum = xp.value + xm.value + yp.value + ym.value + zp.value + zm.value;
          out.write(s, idx, 0, sum * (1.0F / 6.0F));
        };
      },
      std::move(name));
}

[[nodiscard]] auto
build_zero_init(BlockGrid& block,
                FieldT& field,
                std::string name) -> carbon::Kernel {
  return carbon::Kernel::compute(
      block,
      [&field](carbon::KernelLoader& loader) {
        const auto out = loader.write(field);
        return [out] CARBON_HD(Span s, Idx idx) -> void {
          out.write(s, idx, 0, 0.0F);
        };
      },
      std::move(name));
}

[[nodiscard]] auto
run_jacobi_parity(BlockGrid& block,
                  carbon::System& sys,
                  int rank) -> bool {
  auto u = block.template newField<float, 1>(carbon::MemorySide::Device, "u");
  auto u_next = block.template newField<float, 1>(carbon::MemorySide::Device, "u_next");

  carbon::Skeleton init(sys);
  init.add(build_zero_init(block, u, "zero_u"));
  init.add(build_zero_init(block, u_next, "zero_u_next"));
  init.run();
  init.wait();

  carbon::Skeleton main_pipe(sys);
  main_pipe.add(build_jacobi_step(block, u, u_next, "step_ab"));
  main_pipe.add(build_jacobi_step(block, u_next, u, "step_ba"));
  main_pipe.compile();

  const int cycles = kJacobiSweeps / 2;
  for (int c = 0; c < cycles; ++c) {
    main_pipe.run();
  }
  main_pipe.wait();

  carbon::ReductionResult<float> sumsq_result(sys);
  carbon::Skeleton norm_pipe(sys);
  norm_pipe.add(carbon::Kernel::reduce<float>(
      block,
      [&u](carbon::KernelLoader& loader) {
        const auto in = loader.read(u);
        return [in] CARBON_HD(Span s, Idx idx) -> float {
          const float v = in.read(s, idx, 0);
          return v * v;
        };
      },
      sumsq_result, carbon::reduce::Sum<float>{}, "sumsq", 0U));
  norm_pipe.run();
  norm_pipe.wait();
  sumsq_result.sync(sys.stream());
  sys.stream().wait();

  if (rank != 0) {
    return true;
  }

  const double got = static_cast<double>(sumsq_result.host_value());
  const double want = host_reference_sumsq();
  const double rel = std::abs(got - want) / (std::abs(want) + 1.0e-12);
  if (rel > kSumTol) {
    CARBON_LOG_ERROR(
        "test_block_dist_stencil_parity: Jacobi sum|u|^2 mismatch got={:.9e} want={:.9e} "
        "rel={:.3e}",
        got, want, rel);
    return false;
  }
  CARBON_LOG_INFO("test_block_dist_stencil_parity: Jacobi sum|u|^2 ok got={:.9e} want={:.9e}", got,
                  want);
  return true;
}

}  // namespace

int
main() {
  carbon::init();

  auto distributed = std::make_shared<const carbon::Distributed>();
  if (distributed->world_size() < 2) {
    CARBON_LOG_INFO("test_block_dist_stencil_parity: world_size={} < 2, skipping",
                    distributed->world_size());
    return 0;
  }
  if (carbon::device_count() < 1) {
    CARBON_LOG_INFO("test_block_dist_stencil_parity: no CUDA devices, skipping");
    return 0;
  }

  const int rank = distributed->world_rank();
  const int world_size = distributed->world_size();

  carbon::System sys(distributed);
  if (sys.gpu_group().size() != 1U) {
    CARBON_LOG_ERROR("test_block_dist_stencil_parity: expected gpu_group size 1, got {}",
                     sys.gpu_group().size());
    return 1;
  }

  const carbon::int64_3d global{kDomain, kDomain, kDomain};
  BlockGrid block(sys, global, sphere_predicate(kDomain), "block_dist_stencil_parity");
  if (block.gpu_count() != 1U) {
    CARBON_LOG_ERROR("test_block_dist_stencil_parity: expected gpu_count 1, got {}",
                     block.gpu_count());
    return 1;
  }

  if (!run_six_direction_probe(block, sys, rank, world_size)) {
    return 1;
  }
  if (!run_jacobi_parity(block, sys, rank)) {
    return 1;
  }

  CARBON_LOG_INFO("test_block_dist_stencil_parity: rank {}/{} PASS", rank, world_size);
  return 0;
}
