// Multi-level Skeleton integration test for carbon::MultiResBlock (mGrid).
//
// The Skeleton is grid-agnostic: it sequences Kernels purely from their AccessToken's
// (field_uid + mode + pattern + optional halo factory). mGrid already produces correct
// per-level tokens (each (field, level) has its own uid), so the only gap to ride the
// Skeleton was making CROSS-LEVEL reads visible to the dependency graph — that is what
// MultiResField::bind_cross_level closes. This test proves both that closure and the
// per-level auto-halo insertion through one V-cycle-shaped pipeline assembled into a single
// carbon::Skeleton and replayed:
//
//   K0 fill L0      : u@L0 = f(x,y,z)                              (write u@L0)
//   K1 restrict L0->L1 : u@L1 = mean of 8 children                (write u@L1, read u@L0)
//   K2 smooth L1    : sm@L1 = 6-neighbour mean of u@L1 (STENCIL)  (read_stencil u@L1, write sm@L1)
//   K3 prolong L1->L0  : recon@L0 = parent value (u@L1)           (read u@L0, read u@L1, write recon@L0)
//
// K1/K3 declare their cross-level reads via bind_cross_level; K2 uses read_stencil on the level
// view so the Skeleton auto-inserts u@L1's HaloBlock before it. Structural assertions:
//
//   - effective_kernels() == user kernels + exactly one auto-halo (kNoKernel), positioned
//     immediately before the stencil consumer (K2).
//   - edges(): fill -> restrict (proves bind_cross_level's read_child token on u@L0) and
//     restrict -> prolong (proves read_parent token on u@L1). Without the helper's adjacent-level
//     tokens neither edge would exist (the kernels would share no field_uid).
//
// Numerical verification (linear seed f(x,y,z) = x + 2y + 3z, exact to fp tolerance): the
// restricted coarse field is linear in coarse coords, so interior 6-neighbour smoothing is the
// identity and prolongation injects the parent's coarse value.
//
// Domain: fully-active 16x16xN finest, depth-2 octree (N % (MemZ*R) == 0, so the multi-slab
// z-alignment precondition holds) across every visible GPU. CUDA-only registration (mGrid SYCL
// parity is a tracked follow-up — matches the rest of the mGrid suite).

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/devices.h>
#include <carbon/kernel.h>
#include <carbon/mgrid.h>
#include <carbon/skeleton.h>

#include "test_domain.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using SBlock = carbon::DefaultMultiResBlock;  // 8^3 memory brick / 2^3 user block (octree).
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;

constexpr int VoxelCount = SBlock::VoxelCount;
constexpr double kTol = 1e-9;

[[nodiscard]] CARBON_HD constexpr auto
field_value(int64_t x,
            int64_t y,
            int64_t z) noexcept -> double {
  return static_cast<double>(x) + 2.0 * static_cast<double>(y) + 3.0 * static_cast<double>(z);
}

// Sweep a downloaded (owned-only) per-GPU host buffer back to global voxel coordinates; returns
// false on the first mismatch reported by `check`.
template <typename Check>
[[nodiscard]] auto
sweep_active(const carbon::Block<SBlock>& grid,
             const std::vector<carbon::detail::Allocation<double>>& host,
             size_t n_gpus,
             Check&& check) -> bool {
  for (size_t g = 0; g < n_gpus; ++g) {
    const uint32_t nb = grid.block_count(g);
    for (uint32_t b = 0; b < nb; ++b) {
      const auto& bm = grid.host_bitmasks(g)[b];
      const auto origin = grid.host_origins(g)[b];
      for (int z = 0; z < SBlock::MemZ; ++z) {
        for (int y = 0; y < SBlock::MemY; ++y) {
          for (int x = 0; x < SBlock::MemX; ++x) {
            if (!bm.is_active(x, y, z)) {
              continue;
            }
            const size_t lin = static_cast<size_t>(b) * static_cast<size_t>(VoxelCount) +
                               static_cast<size_t>(x) +
                               static_cast<size_t>(SBlock::MemX) * static_cast<size_t>(y) +
                               static_cast<size_t>(SBlock::MemX) *
                                   static_cast<size_t>(SBlock::MemY) * static_cast<size_t>(z);
            const carbon::int64_3d gc{origin.x + x, origin.y + y, origin.z + z};
            if (!check(gc, host[g].data()[lin])) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] auto
run_test() -> int {
  carbon::init();
  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("test_mgrid_skeleton: no CUDA devices visible — skipping\n");
    return 0;
  }
  const size_t n_gpus = devices.size();
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  const carbon::int64_3d finest{16, 16, carbon::test::block_slab_z(32, n_gpus, SBlock::MemZ * 2)};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, /*ref_factor=*/2,
                                       "mgrid_skeleton");
  const int R = mblock.ref_factor();

  auto u = mblock.newField<double, 1>(carbon::MemorySide::Device, "u");
  auto sm = mblock.newField<double, 1>(carbon::MemorySide::Device, "sm");
  auto recon = mblock.newField<double, 1>(carbon::MemorySide::Device, "recon");

  carbon::Skeleton sk(system);

  // K0: seed the finest level with f(x,y,z).
  sk.add(mblock.make_kernel(0, [u](carbon::KernelLoader& loader) {
    auto p = loader.write(u.level_view(0));
    return [p] CARBON_HD(Span s, Idx idx) -> void {
      const auto g = s.global_location(idx);
      p.write(s, idx, 0, field_value(g.x, g.y, g.z));
    };
  }));

  // K1: restriction L0 -> L1 (coarse = mean of 8 children). bind_cross_level declares the
  // child (L0) read so the Skeleton orders this after K0.
  sk.add(mblock.make_kernel(1, [u, R](carbon::KernelLoader& loader) {
    auto p = u.bind_cross_level(loader, 1, carbon::AccessMode::Write, /*read_child=*/true);
    return [p, R] CARBON_HD(Span s, Idx idx) -> void {
      double sum = 0.0;
      int count = 0;
      for (int8_t oz = 0; oz < R; ++oz) {
        for (int8_t oy = 0; oy < R; ++oy) {
          for (int8_t ox = 0; ox < R; ++ox) {
            const auto cv = p.child(s, idx, carbon::int8_3d{ox, oy, oz}, 0);
            if (cv.valid) {
              sum += cv.value;
              ++count;
            }
          }
        }
      }
      p.write(s, idx, 0, count > 0 ? sum / static_cast<double>(count) : 0.0);
    };
  }));

  // K2: same-level 6-neighbour smoothing on L1. The same-level neighbour walk needs the block
  // Partition (the cross-level mgrid::Partition has no neighbour()), so it binds the level's
  // block::Field directly — same field_uid as u.level_view(1). read_stencil records the Stencil
  // pattern + that level's halo factory, so the Skeleton auto-inserts u@L1's HaloBlock right
  // before this kernel.
  sk.add(mblock.make_kernel(1, [u, sm](carbon::KernelLoader& loader) {
    auto in = loader.read_stencil(u.level_field(1));
    auto out = loader.write(sm.level_field(1));
    return [in, out] CARBON_HD(Span s, Idx idx) -> void {
      const carbon::int8_3d offs[6] = {{+1, 0, 0}, {-1, 0, 0}, {0, +1, 0},
                                       {0, -1, 0}, {0, 0, +1}, {0, 0, -1}};
      double sum = 0.0;
      int count = 0;
      for (const auto& o : offs) {
        const auto nv = in.neighbour(s, idx, o, 0);
        if (nv.valid) {
          sum += nv.value;
          ++count;
        }
      }
      out.write(s, idx, 0, count > 0 ? sum / static_cast<double>(count) : 0.0);
    };
  }));

  // K3: prolongation L1 -> L0 (inject parent value). bind_cross_level on u declares the parent
  // (L1) read so the Skeleton orders this after K1; recon@L0 is the write target.
  sk.add(mblock.make_kernel(0, [u, recon](carbon::KernelLoader& loader) {
    auto pin = u.bind_cross_level(loader, 0, carbon::AccessMode::Read, /*read_child=*/false,
                                  /*read_parent=*/true);
    auto pout = loader.write(recon.level_view(0));
    return [pin, pout] CARBON_HD(Span s, Idx idx) -> void {
      const auto pv = pin.parent(s, idx, 0);
      pout.write(s, idx, 0, pv.valid ? pv.value : 0.0);
    };
  }));

  sk.compile();

  // ---- Structural assertions on the inferred schedule. ----
  const auto& eff_to_user = sk.effective_to_user();
  const auto& edges = sk.edges();
  if (sk.effective_kernels().size() != sk.kernels().size() + 1U) {
    std::fprintf(
        stderr, "test_mgrid_skeleton: expected exactly one auto-halo (effective %zu vs user %zu)\n",
        sk.effective_kernels().size(), sk.kernels().size());
    return 1;
  }
  size_t auto_halos = 0;
  size_t halo_eff = carbon::Skeleton::kNoKernel;
  for (size_t e = 0; e < eff_to_user.size(); ++e) {
    if (eff_to_user[e] == carbon::Skeleton::kNoKernel) {
      ++auto_halos;
      halo_eff = e;
    }
  }
  // Map a user-kernel index to its effective index.
  const auto eff_of = [&](size_t user_idx) -> size_t {
    for (size_t e = 0; e < eff_to_user.size(); ++e) {
      if (eff_to_user[e] == user_idx) {
        return e;
      }
    }
    return carbon::Skeleton::kNoKernel;
  };
  const size_t eff_fill = eff_of(0);
  const size_t eff_restrict = eff_of(1);
  const size_t eff_smooth = eff_of(2);
  const size_t eff_prolong = eff_of(3);

  const auto has_edge = [&](size_t from, size_t to) -> bool {
    if (from >= edges.size()) {
      return false;
    }
    return std::find(edges[from].begin(), edges[from].end(), to) != edges[from].end();
  };

  if (auto_halos != 1U || halo_eff + 1U != eff_smooth) {
    std::fprintf(stderr,
                 "test_mgrid_skeleton: auto-halo placement wrong (count=%zu halo_eff=%zu "
                 "smooth_eff=%zu)\n",
                 auto_halos, halo_eff, eff_smooth);
    return 1;
  }
  if (!has_edge(eff_fill, eff_restrict)) {
    std::fprintf(stderr,
                 "test_mgrid_skeleton: missing fill->restrict edge — bind_cross_level read_child "
                 "token not recorded\n");
    return 1;
  }
  if (!has_edge(eff_restrict, eff_prolong)) {
    std::fprintf(stderr,
                 "test_mgrid_skeleton: missing restrict->prolong edge — bind_cross_level "
                 "read_parent token not recorded\n");
    return 1;
  }

  // ---- Run and verify numerics. ----
  sk.run(carbon::DataView::Standard);
  sk.wait();
  carbon::Stream stream = system.stream();

  // u@L1 == restriction mean (coarse field, linear in coarse coords).
  const auto coarse_expected = [&](const carbon::int64_3d& gc) -> double {
    double e = 0.0;
    for (int oz = 0; oz < R; ++oz) {
      for (int oy = 0; oy < R; ++oy) {
        for (int ox = 0; ox < R; ++ox) {
          e += field_value(R * gc.x + ox, R * gc.y + oy, R * gc.z + oz);
        }
      }
    }
    return e / static_cast<double>(R * R * R);
  };

  auto host_u1 = mblock.grid(1).download_to_host<double, 1>(u.level_field(1), stream);
  auto host_sm1 = mblock.grid(1).download_to_host<double, 1>(sm.level_field(1), stream);
  auto host_recon0 = mblock.grid(0).download_to_host<double, 1>(recon.level_field(0), stream);
  stream.wait();

  double max_err = 0.0;
  const bool restrict_ok =
      sweep_active(mblock.grid(1), host_u1, n_gpus, [&](const carbon::int64_3d& gc, double got) {
        max_err = std::max(max_err, std::fabs(got - coarse_expected(gc)));
        return std::fabs(got - coarse_expected(gc)) <= kTol;
      });
  // K2 is the interior 6-neighbour mean of the (linear) coarse field, so sm@L1 == u@L1 on every
  // coarse cell with all 6 neighbours in-domain (incl. cells adjacent to the multi-GPU slab cut,
  // which only holds if the auto-inserted halo delivered the neighbour slab's data).
  const carbon::int64_3d coarse_n{finest.x / R, finest.y / R, finest.z / R};
  int64_t interior_checked = 0;
  const bool smooth_ok =
      sweep_active(mblock.grid(1), host_sm1, n_gpus, [&](const carbon::int64_3d& gc, double got) {
        const bool interior = gc.x > 0 && gc.x < coarse_n.x - 1 && gc.y > 0 &&
                              gc.y < coarse_n.y - 1 && gc.z > 0 && gc.z < coarse_n.z - 1;
        if (!interior) {
          return true;
        }
        ++interior_checked;
        max_err = std::max(max_err, std::fabs(got - coarse_expected(gc)));
        return std::fabs(got - coarse_expected(gc)) <= kTol;
      });
  // recon@L0[v] == coarse value of v's parent (v / R).
  const bool prolong_ok = sweep_active(
      mblock.grid(0), host_recon0, n_gpus, [&](const carbon::int64_3d& gc, double got) {
        const carbon::int64_3d pc{gc.x / R, gc.y / R, gc.z / R};
        max_err = std::max(max_err, std::fabs(got - coarse_expected(pc)));
        return std::fabs(got - coarse_expected(pc)) <= kTol;
      });

  if (!restrict_ok || !smooth_ok || !prolong_ok) {
    std::fprintf(stderr,
                 "test_mgrid_skeleton: numeric mismatch (restrict=%d smooth=%d prolong=%d "
                 "max_err=%.3e)\n",
                 static_cast<int>(restrict_ok), static_cast<int>(smooth_ok),
                 static_cast<int>(prolong_ok), max_err);
    return 1;
  }

  std::printf(
      "test_mgrid_skeleton: OK (%zu GPU(s), %zu streams, effective kernels=%zu incl 1 auto-halo, "
      "interior coarse cells=%ld, max_err=%.3e)\n",
      n_gpus, sk.num_streams(), sk.effective_kernels().size(), static_cast<long>(interior_checked),
      max_err);
  return 0;
}

}  // namespace

int
main() {
  return run_test();
}
