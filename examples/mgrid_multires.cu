// Multi-resolution (mGrid) worked example: restriction, prolongation, reduction, and an
// intra-level halo'd smoothing step on carbon::MultiResBlock — every operation written through
// the standard kernel layer (MultiResBlock::make_kernel / make_reduce_kernel,
// MultiResField::level_view / make_halo_kernel), with no hand-rolled <<<>>> kernels.
//
// It builds a fully-active depth-2 octree over an N×N×N domain across every visible GPU (the
// coarsest level is z-partitioned and the cut is projected to the fine level, so a coarse cell
// and its 8 children are co-resident — cross-level reads are transfer-free), seeds the finest
// level with the linear field f(x,y,z) = x + 2y + 3z, then runs four self-verifying stages:
//
//   A. Restriction  L0 -> L1 : coarse[c] = mean of c's 8 children (cross-level child reads).
//      Verified: for a linear f the cell-mean equals f at the cell centroid (exact).
//   B. Reduction    over L1  : sum of all coarse values via make_reduce_kernel; checked vs host.
//   C. Prolongation L1 -> L0 : recon[v] = value of v's parent (cross-level parent read); checked
//      vs host (== the mean of v's 8 siblings).
//   D. Smoothing    on  L0   : after a per-level halo, out[v] = mean of v's in-domain 6-axis
//      neighbours via block::Partition::neighbour. For a linear f the interior mean == f(v)
//      exactly, INCLUDING cells adjacent to the multi-GPU slab cut — which only holds if the
//      halo delivered the neighbour slab's data, so this doubles as a halo correctness check.
//   E. Skeleton     whole    : the same restriction / stencil-smoothing / prolongation assembled
//      into ONE carbon::Skeleton and replayed. Cross-level reads are declared with
//      MultiResField::bind_cross_level (so the scheduler orders restrict after the fill and
//      prolong after restrict); the L0 smoothing uses read_stencil, so the Skeleton AUTO-INSERTS
//      u's per-level halo — no manual make_halo_kernel call. Verified to reproduce stages A/C/D.
//
// Usage: mgrid_multires [N]   (N defaults to 32; must be positive and, for the multi-GPU case,
// a multiple of MemZ*R = 16 so the multi-slab z-alignment precondition holds).

#include <carbon/carbon.h>
#include <carbon/cpu.h>
#include <carbon/devices.h>
#include <carbon/kernel.h>
#include <carbon/mgrid.h>
#include <carbon/skeleton.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using SBlock = carbon::DefaultMultiResBlock;  // 8^3 memory brick / 2^3 user block (octree).
using Span = carbon::detail::block::Span<SBlock>;
using Idx = carbon::detail::block::Idx;

constexpr int VoxelCount = SBlock::VoxelCount;
constexpr double kTol = 1e-9;

// Linear field — its cell-mean is exact under restriction and its 6-neighbour interior mean
// equals the centre value, which makes every stage host-verifiable to floating-point tolerance.
[[nodiscard]] CARBON_HD constexpr auto
field_value(int64_t x,
            int64_t y,
            int64_t z) noexcept -> double {
  return static_cast<double>(x) + 2.0 * static_cast<double>(y) + 3.0 * static_cast<double>(z);
}

// Map a downloaded (owned-only) per-GPU host buffer entry back to its global voxel coordinate,
// iterating the level's retained bricks. Returns false on any mismatch found by `check`.
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

}  // namespace

int
main(int argc,
     char** argv) {
  carbon::init();

  int64_t domain_n = 32;
  if (argc > 1) {
    const long parsed = std::strtol(argv[1], nullptr, 10);  // example CLI: a domain size only.
    if (parsed <= 0) {
      std::fprintf(stderr, "mgrid_multires: N must be a positive integer (got '%s')\n", argv[1]);
      return 2;
    }
    domain_n = static_cast<int64_t>(parsed);
  }

  auto devices = carbon::get_all_devices();
  if (devices.empty()) {
    std::printf("mgrid_multires: no CUDA devices visible — skipping\n");
    return 0;
  }
  const size_t n_gpus = devices.size();
  const int ref_factor = SBlock::UserX;  // octree branching = 2.
  if (n_gpus > 1U && domain_n % (SBlock::MemZ * ref_factor) != 0) {
    std::fprintf(stderr, "mgrid_multires: multi-GPU run needs N (%ld) a multiple of MemZ*R (%d)\n",
                 static_cast<long>(domain_n), SBlock::MemZ * ref_factor);
    return 2;
  }
  carbon::System system(carbon::detail::GpuGroup{std::move(devices)});

  const carbon::int64_3d finest{domain_n, domain_n, domain_n};
  const auto is_active = [](int /*level*/, const carbon::int64_3d& /*v*/) -> bool {
    return true;
  };
  carbon::MultiResBlock<SBlock> mblock(system, finest, /*depth=*/2, is_active, ref_factor,
                                       "mgrid_multires");
  const int R = mblock.ref_factor();
  std::printf("mgrid_multires: N=%ld depth=%d R=%d gpus=%zu\n", static_cast<long>(domain_n),
              mblock.depth(), R, n_gpus);

  auto u = mblock.newField<double, 1>(carbon::MemorySide::Device, "u");
  auto recon = mblock.newField<double, 1>(carbon::MemorySide::Device, "recon");
  auto smoothed = mblock.newField<double, 1>(carbon::MemorySide::Device, "smoothed");
  carbon::Stream stream = system.stream();

  // Seed the finest level with f(x,y,z).
  mblock
      .make_kernel(0,
                   [u](carbon::KernelLoader& loader) {
                     auto p = loader.write(u.level_view(0));
                     return [p] CARBON_HD(Span s, Idx idx) -> void {
                       const auto g = s.global_location(idx);
                       p.write(s, idx, 0, field_value(g.x, g.y, g.z));
                     };
                   })
      .launch(stream);

  // ---- Stage A: restriction L0 -> L1 (mean of 8 children). ----
  mblock
      .make_kernel(1,
                   [u, R](carbon::KernelLoader& loader) {
                     auto p = loader.read(u.level_view(1));  // own=L1, child=L0.
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
                   })
      .launch(stream);
  stream.wait();

  auto host_l1 = mblock.grid(1).download_to_host<double, 1>(u.level_field(1), stream);
  stream.wait();
  double max_err_restrict = 0.0;
  const bool restrict_ok = sweep_active(
      mblock.grid(1), host_l1, n_gpus, [&](const carbon::int64_3d& gc, double got) -> bool {
        // Mean of the R^3 fine children of coarse cell gc (fine coords R*gc + o).
        double expect = 0.0;
        for (int oz = 0; oz < R; ++oz) {
          for (int oy = 0; oy < R; ++oy) {
            for (int ox = 0; ox < R; ++ox) {
              expect += field_value(R * gc.x + ox, R * gc.y + oy, R * gc.z + oz);
            }
          }
        }
        expect /= static_cast<double>(R * R * R);
        max_err_restrict = std::max(max_err_restrict, std::fabs(got - expect));
        return std::fabs(got - expect) <= kTol;
      });
  if (!restrict_ok) {
    std::fprintf(stderr, "mgrid_multires: restriction mismatch (max_err=%.3e)\n", max_err_restrict);
    return 1;
  }
  std::printf("mgrid_multires: [A] restriction L0->L1 OK (max_err=%.3e)\n", max_err_restrict);

  // ---- Stage B: reduction over L1 (sum of coarse values). ----
  carbon::ReductionResult<double> coarse_sum(system);
  mblock
      .make_reduce_kernel<double>(
          1,
          [u](carbon::KernelLoader& loader) {
            auto p = loader.read(u.level_view(1));
            return [p] CARBON_HD(Span s, Idx idx) -> double {
              return p.read(s, idx, 0);
            };
          },
          coarse_sum)
      .launch(stream, carbon::DataView::Standard);
  coarse_sum.sync(stream);
  stream.wait();
  double host_sum = 0.0;
  (void)sweep_active(mblock.grid(1), host_l1, n_gpus,
                     [&](const carbon::int64_3d& /*gc*/, double got) -> bool {
                       host_sum += got;
                       return true;
                     });
  const double reduced = coarse_sum.host_value(carbon::DataView::Standard);
  if (std::fabs(reduced - host_sum) > kTol * std::max(1.0, std::fabs(host_sum))) {
    std::fprintf(stderr, "mgrid_multires: reduce mismatch (device=%.6f host=%.6f)\n", reduced,
                 host_sum);
    return 1;
  }
  std::printf("mgrid_multires: [B] reduce(sum L1) OK (%.6f)\n", reduced);

  // ---- Stage C: prolongation L1 -> L0 (inject parent value). ----
  mblock
      .make_kernel(0,
                   [u, recon](carbon::KernelLoader& loader) {
                     auto pin = loader.read(u.level_view(0));  // own=L0, parent=L1.
                     auto pout = loader.write(recon.level_view(0));
                     return [pin, pout] CARBON_HD(Span s, Idx idx) -> void {
                       const auto pv = pin.parent(s, idx, 0);
                       pout.write(s, idx, 0, pv.valid ? pv.value : 0.0);
                     };
                   })
      .launch(stream);
  stream.wait();
  auto host_recon = mblock.grid(0).download_to_host<double, 1>(recon.level_field(0), stream);
  stream.wait();
  double max_err_prolong = 0.0;
  const bool prolong_ok = sweep_active(
      mblock.grid(0), host_recon, n_gpus, [&](const carbon::int64_3d& gc, double got) -> bool {
        // Parent of fine voxel gc is the coarse cell gc / R; its value is the mean of
        // that cell's R^3 children.
        const carbon::int64_3d pc{gc.x / R, gc.y / R, gc.z / R};
        double expect = 0.0;
        for (int oz = 0; oz < R; ++oz) {
          for (int oy = 0; oy < R; ++oy) {
            for (int ox = 0; ox < R; ++ox) {
              expect += field_value(R * pc.x + ox, R * pc.y + oy, R * pc.z + oz);
            }
          }
        }
        expect /= static_cast<double>(R * R * R);
        max_err_prolong = std::max(max_err_prolong, std::fabs(got - expect));
        return std::fabs(got - expect) <= kTol;
      });
  if (!prolong_ok) {
    std::fprintf(stderr, "mgrid_multires: prolongation mismatch (max_err=%.3e)\n", max_err_prolong);
    return 1;
  }
  std::printf("mgrid_multires: [C] prolongation L1->L0 OK (max_err=%.3e)\n", max_err_prolong);

  // ---- Stage D: intra-level halo + 6-neighbour smoothing on L0. ----
  // Re-seed L0 with f (Stage A only read it; it is untouched, but be explicit), then exchange the
  // per-level halo so cross-slab neighbours are available, and smooth.
  u.make_halo_kernel(0).launch(stream);
  stream.wait();
  mblock.grid(0)
      .make_kernel([u, smoothed](carbon::KernelLoader& loader) {
        auto in = loader.read(u.level_field(0));  // block::Partition (has neighbour()).
        auto out = loader.write(smoothed.level_field(0));
        return [in, out] CARBON_HD(Span s, Idx idx) -> void {
          double sum = 0.0;
          int count = 0;
          const carbon::int8_3d offs[6] = {{+1, 0, 0}, {-1, 0, 0}, {0, +1, 0},
                                           {0, -1, 0}, {0, 0, +1}, {0, 0, -1}};
          for (const auto& o : offs) {
            const auto nv = in.neighbour(s, idx, o, 0);
            if (nv.valid) {
              sum += nv.value;
              ++count;
            }
          }
          out.write(s, idx, 0, count > 0 ? sum / static_cast<double>(count) : 0.0);
        };
      })
      .launch(stream);
  stream.wait();
  auto host_smooth = mblock.grid(0).download_to_host<double, 1>(smoothed.level_field(0), stream);
  stream.wait();
  double max_err_smooth = 0.0;
  int64_t interior_checked = 0;
  const bool smooth_ok = sweep_active(
      mblock.grid(0), host_smooth, n_gpus, [&](const carbon::int64_3d& gc, double got) -> bool {
        // Only interior cells have all 6 neighbours in-domain; for a linear f their mean == f(gc).
        const bool interior = gc.x > 0 && gc.x < domain_n - 1 && gc.y > 0 && gc.y < domain_n - 1 &&
                              gc.z > 0 && gc.z < domain_n - 1;
        if (!interior) {
          return true;
        }
        ++interior_checked;
        const double expect = field_value(gc.x, gc.y, gc.z);
        max_err_smooth = std::max(max_err_smooth, std::fabs(got - expect));
        return std::fabs(got - expect) <= kTol;
      });
  if (!smooth_ok) {
    std::fprintf(stderr, "mgrid_multires: smoothing mismatch (max_err=%.3e)\n", max_err_smooth);
    return 1;
  }
  std::printf(
      "mgrid_multires: [D] halo + 6-neighbour smoothing OK (interior cells=%ld, max_err=%.3e)\n",
      static_cast<long>(interior_checked), max_err_smooth);

  // ---- Stage E: the same multi-level pipeline assembled into one carbon::Skeleton. ----
  // fill u@L0 -> restrict L0->L1 -> {stencil-smooth L0 (auto-halo), prolong L1->L0}. The
  // cross-level reads are declared via bind_cross_level; the stencil smooth's read_stencil makes
  // the Skeleton auto-insert u@L0's halo. The compiled graph reproduces stages A/C/D exactly.
  auto u_sk = mblock.newField<double, 1>(carbon::MemorySide::Device, "u_sk");
  auto sk_smooth = mblock.newField<double, 1>(carbon::MemorySide::Device, "sk_smooth");
  auto sk_recon = mblock.newField<double, 1>(carbon::MemorySide::Device, "sk_recon");

  carbon::Skeleton sk(system);
  sk.add(mblock.make_kernel(0, [u_sk](carbon::KernelLoader& loader) {
    auto p = loader.write(u_sk.level_view(0));
    return [p] CARBON_HD(Span s, Idx idx) -> void {
      const auto g = s.global_location(idx);
      p.write(s, idx, 0, field_value(g.x, g.y, g.z));
    };
  }));
  sk.add(mblock.make_kernel(1, [u_sk, R](carbon::KernelLoader& loader) {
    auto p = u_sk.bind_cross_level(loader, 1, carbon::AccessMode::Write, /*read_child=*/true);
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
  sk.add(mblock.make_kernel(0, [u_sk, sk_smooth](carbon::KernelLoader& loader) {
    auto in = loader.read_stencil(u_sk.level_field(0));  // auto-halo on u_sk@L0.
    auto out = loader.write(sk_smooth.level_field(0));
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
  sk.add(mblock.make_kernel(0, [u_sk, sk_recon](carbon::KernelLoader& loader) {
    auto pin = u_sk.bind_cross_level(loader, 0, carbon::AccessMode::Read, /*read_child=*/false,
                                     /*read_parent=*/true);
    auto pout = loader.write(sk_recon.level_view(0));
    return [pin, pout] CARBON_HD(Span s, Idx idx) -> void {
      const auto pv = pin.parent(s, idx, 0);
      pout.write(s, idx, 0, pv.valid ? pv.value : 0.0);
    };
  }));
  sk.compile();
  sk.run(carbon::DataView::Standard);
  sk.wait();

  auto sk_host_l1 = mblock.grid(1).download_to_host<double, 1>(u_sk.level_field(1), stream);
  auto sk_host_smooth =
      mblock.grid(0).download_to_host<double, 1>(sk_smooth.level_field(0), stream);
  auto sk_host_recon = mblock.grid(0).download_to_host<double, 1>(sk_recon.level_field(0), stream);
  stream.wait();

  double max_err_sk = 0.0;
  const bool sk_restrict_ok = sweep_active(
      mblock.grid(1), sk_host_l1, n_gpus, [&](const carbon::int64_3d& gc, double got) -> bool {
        double expect = 0.0;
        for (int oz = 0; oz < R; ++oz) {
          for (int oy = 0; oy < R; ++oy) {
            for (int ox = 0; ox < R; ++ox) {
              expect += field_value(R * gc.x + ox, R * gc.y + oy, R * gc.z + oz);
            }
          }
        }
        expect /= static_cast<double>(R * R * R);
        max_err_sk = std::max(max_err_sk, std::fabs(got - expect));
        return std::fabs(got - expect) <= kTol;
      });
  const bool sk_smooth_ok = sweep_active(
      mblock.grid(0), sk_host_smooth, n_gpus, [&](const carbon::int64_3d& gc, double got) -> bool {
        const bool interior = gc.x > 0 && gc.x < domain_n - 1 && gc.y > 0 && gc.y < domain_n - 1 &&
                              gc.z > 0 && gc.z < domain_n - 1;
        if (!interior) {
          return true;
        }
        const double expect = field_value(gc.x, gc.y, gc.z);
        max_err_sk = std::max(max_err_sk, std::fabs(got - expect));
        return std::fabs(got - expect) <= kTol;
      });
  const bool sk_prolong_ok = sweep_active(
      mblock.grid(0), sk_host_recon, n_gpus, [&](const carbon::int64_3d& gc, double got) -> bool {
        const carbon::int64_3d pc{gc.x / R, gc.y / R, gc.z / R};
        double expect = 0.0;
        for (int oz = 0; oz < R; ++oz) {
          for (int oy = 0; oy < R; ++oy) {
            for (int ox = 0; ox < R; ++ox) {
              expect += field_value(R * pc.x + ox, R * pc.y + oy, R * pc.z + oz);
            }
          }
        }
        expect /= static_cast<double>(R * R * R);
        max_err_sk = std::max(max_err_sk, std::fabs(got - expect));
        return std::fabs(got - expect) <= kTol;
      });
  if (!sk_restrict_ok || !sk_smooth_ok || !sk_prolong_ok) {
    std::fprintf(stderr,
                 "mgrid_multires: Skeleton stage mismatch (restrict=%d smooth=%d prolong=%d "
                 "max_err=%.3e)\n",
                 static_cast<int>(sk_restrict_ok), static_cast<int>(sk_smooth_ok),
                 static_cast<int>(sk_prolong_ok), max_err_sk);
    return 1;
  }
  std::printf(
      "mgrid_multires: [E] Skeleton (restrict+smooth+prolong, %zu effective kernels incl auto-halo,"
      " %zu streams) OK (max_err=%.3e)\n",
      sk.effective_kernels().size(), sk.num_streams(), max_err_sk);

  std::printf("mgrid_multires: PASS (all stages verified across %zu GPU(s))\n", n_gpus);
  return 0;
}
