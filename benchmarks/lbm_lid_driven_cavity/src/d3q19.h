#pragma once

// D3Q19 lattice constants for the carbon LBM benchmark.
//
// Indexing: populations are bucketed by sign(c.z) so the disg lattice halo
// (carbon::HaloDisg with StencilSemantic::Lattice) collapses to exactly two
// fused cudaMemcpyAsync calls per inter-slab adjacency pair, regardless of Q.
//
//   q in [0, 4]  :  c.z = -1   (5 populations, sent downward)
//   q in [5, 13] :  c.z =  0   (9 populations including center, no z-halo)
//   q in [14, 18]:  c.z = +1   (5 populations, sent upward)
//
// Within each bucket the assignment is palindromic: c(q) = -c(Q - 1 - q),
// which gives the closed-form
//
//   opp(q) = 18 - q
//
// and lets the GO/BK split fall out of "q < 9 / q == 9 / q > 9" rather than
// the old +10 offset trick. The center sits at q = 9 (its own opposite).
//
// This ordering matches Neon's `Neon::set::StencilSemantic::lattice`
// recommendation and Carbon's halo_scalability benchmark — see the
// 2026-05 CHANGELOG entry for the design rationale and 86% bandwidth-
// efficiency numbers vs the unfused ordering.

#include <carbon/cuda_annotation.h>
#include <carbon/integer3d.h>

#include <stdint.h>

namespace carbon::bench::lbm {

struct D3Q19 {
  static constexpr int Q = 19;
  static constexpr int kCenter = 9;
  // Z-bucket boundaries (half-open intervals), exposed for callers that want
  // to assert lattice-friendly population ordering at compile time.
  static constexpr int kZNegBegin = 0;
  static constexpr int kZNegEnd = 5;
  static constexpr int kZZeroBegin = 5;
  static constexpr int kZZeroEnd = 14;
  static constexpr int kZPosBegin = 14;
  static constexpr int kZPosEnd = 19;

  // Discrete velocities c_q. Bucketed by sign(c.z) so all c.z<0 entries are
  // contiguous in [0, 5), c.z=0 in [5, 14), c.z>0 in [14, 19); palindromic
  // within the lattice so c(18 - q) = -c(q).
  [[nodiscard]] CARBON_HD static constexpr auto
  c(int q) noexcept -> ::carbon::int8_3d {
    switch (q) {
      // ---- c.z = -1 (5 entries; sent downward by the lattice halo) ----
      case 0:
        return {0, 0, -1};
      case 1:
        return {-1, 0, -1};
      case 2:
        return {1, 0, -1};
      case 3:
        return {0, -1, -1};
      case 4:
        return {0, 1, -1};
      // ---- c.z = 0 (9 entries; not exchanged across z-slabs) ----
      case 5:
        return {-1, 0, 0};
      case 6:
        return {0, -1, 0};
      case 7:
        return {-1, -1, 0};
      case 8:
        return {-1, 1, 0};
      case 9:
        return {0, 0, 0};  // center
      case 10:
        return {1, -1, 0};
      case 11:
        return {1, 1, 0};
      case 12:
        return {0, 1, 0};
      case 13:
        return {1, 0, 0};
      // ---- c.z = +1 (5 entries; sent upward by the lattice halo) ----
      case 14:
        return {0, -1, 1};
      case 15:
        return {0, 1, 1};
      case 16:
        return {-1, 0, 1};
      case 17:
        return {1, 0, 1};
      case 18:
        return {0, 0, 1};
      default:
        return {0, 0, 0};
    }
  }

  // Closed-form opposite-direction map: c(18 - q) = -c(q). Verified by
  // construction in c() above and locked by tests/test_d3q19_lattice.cpp.
  [[nodiscard]] CARBON_HD static constexpr auto
  opp(int q) noexcept -> int {
    return Q - 1 - q;
  }

  // Lattice weights w_q. Standard D3Q19: 1/3 (center), 1/18 (axial),
  // 1/36 (face-diagonal). Returned as float so the BGK arithmetic stays on a
  // single precision and does not promote.
  //
  // Axial = exactly one component nonzero; face-diagonal = exactly two
  // nonzero components (D3Q19 has no corner / 3-component directions).
  [[nodiscard]] CARBON_HD static constexpr auto
  w(int q) noexcept -> float {
    if (q == kCenter) {
      return 1.0F / 3.0F;
    }
    const auto cq = c(q);
    const int nz = (cq.x != 0 ? 1 : 0) + (cq.y != 0 ? 1 : 0) + (cq.z != 0 ? 1 : 0);
    return (nz == 1) ? (1.0F / 18.0F) : (1.0F / 36.0F);
  }

  // KBC is not supported on D3Q19 in this benchmark — the standard KBC
  // formulation (Karlin et al.) targets D3Q27 and Neon's reference only
  // wires KBC against D3Q27. main() rejects (q19, kbc) at the CLI parse
  // stage so the moment / shear-decompose helpers are not needed here.
};

}  // namespace carbon::bench::lbm
