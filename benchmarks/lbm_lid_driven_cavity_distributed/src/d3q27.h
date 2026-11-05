#pragma once

// D3Q27 lattice (full 3D 27-velocity stencil) for the carbon LBM benchmark.
//
// Indexing: populations are bucketed by sign(c.z) so the disg lattice halo
// collapses to exactly two fused transfers per inter-slab adjacency pair:
//
//   q in [0, 9)   :  c.z = -1   (9 populations, sent downward)
//   q in [9, 18)  :  c.z =  0   (9 populations including center, no z-halo)
//   q in [18, 27) :  c.z = +1   (9 populations, sent upward)
//
// Palindromic within the lattice: c(q) = -c(26 - q), which gives the closed
// form
//
//   opp(q) = 26 - q
//
// and places the center at q = 13 (its own opposite). This differs from the
// Neon lbmMultiRes ordering (which placed center at q=0) but matches the
// Neon::set::StencilSemantic::lattice fusion contract. The kernels in
// lbm_kernels.h are written generically over q in [0, Q) so the iteration
// kernel and macroscopic-extraction kernel pick up the new ordering for free.
//
// The KBC moment and shear-decomposition helpers used to be hand-tabulated
// per q-index against the legacy ordering. They are now derived directly
// from c(q) (moment is the c⊗c outer product's six unique entries; shear
// projection is a closed form of the active components and their signs),
// so the table stays correct under any future re-ordering.

#include <carbon/cuda_annotation.h>
#include <carbon/integer3d.h>

#include <stdint.h>

namespace carbon::bench::lbm {

struct D3Q27 {
  static constexpr int Q = 27;
  static constexpr int kCenter = 13;
  static constexpr int kZNegBegin = 0;
  static constexpr int kZNegEnd = 9;
  static constexpr int kZZeroBegin = 9;
  static constexpr int kZZeroEnd = 18;
  static constexpr int kZPosBegin = 18;
  static constexpr int kZPosEnd = 27;

  // Discrete velocities c_q. Bucketed by sign(c.z); palindromic so
  // c(26 - q) = -c(q).
  [[nodiscard]] CARBON_HD static constexpr auto
  c(int q) noexcept -> ::carbon::int8_3d {
    switch (q) {
      // ---- c.z = -1 (9 entries) ----
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
      case 5:
        return {-1, -1, -1};
      case 6:
        return {-1, 1, -1};
      case 7:
        return {1, -1, -1};
      case 8:
        return {1, 1, -1};
      // ---- c.z = 0 (9 entries) ----
      case 9:
        return {-1, 0, 0};
      case 10:
        return {0, -1, 0};
      case 11:
        return {-1, -1, 0};
      case 12:
        return {-1, 1, 0};
      case 13:
        return {0, 0, 0};  // center
      case 14:
        return {1, -1, 0};
      case 15:
        return {1, 1, 0};
      case 16:
        return {0, 1, 0};
      case 17:
        return {1, 0, 0};
      // ---- c.z = +1 (9 entries) ----
      case 18:
        return {-1, -1, 1};
      case 19:
        return {-1, 1, 1};
      case 20:
        return {1, -1, 1};
      case 21:
        return {1, 1, 1};
      case 22:
        return {0, -1, 1};
      case 23:
        return {0, 1, 1};
      case 24:
        return {-1, 0, 1};
      case 25:
        return {1, 0, 1};
      case 26:
        return {0, 0, 1};
      default:
        return {0, 0, 0};
    }
  }

  // Closed-form opposite: c(26 - q) = -c(q) by construction.
  [[nodiscard]] CARBON_HD static constexpr auto
  opp(int q) noexcept -> int {
    return Q - 1 - q;
  }

  // Lattice weights w_q. Standard D3Q27:
  //   8/27 (center), 2/27 (axial, 1 nonzero), 1/54 (face, 2 nonzero),
  //   1/216 (corner, 3 nonzero).
  // Derived from |c(q)|_0 (the count of nonzero components) so the table
  // stays correct under any reordering.
  [[nodiscard]] CARBON_HD static constexpr auto
  w(int q) noexcept -> float {
    if (q == kCenter) {
      return 8.0F / 27.0F;
    }
    const auto cq = c(q);
    const int nz = (cq.x != 0 ? 1 : 0) + (cq.y != 0 ? 1 : 0) + (cq.z != 0 ? 1 : 0);
    if (nz == 1) {
      return 2.0F / 27.0F;
    }
    if (nz == 2) {
      return 1.0F / 54.0F;
    }
    return 1.0F / 216.0F;
  }

  // Moment-projection matrix Mq[6] for the second-order non-equilibrium
  // stress tensor used by KBC. Layout matches Neon's lbmMultiRes/lattice.h
  // latticeMoment but is now derived from c(q) directly:
  //
  //   m = (c.x*c.x, c.x*c.y, c.x*c.z, c.y*c.y, c.y*c.z, c.z*c.z)
  //
  // i.e. the 6 unique entries of the outer product c ⊗ c. All entries fit
  // int8_t since |c.i| <= 1.
  struct Moment {
    int8_t v[6];
  };

  [[nodiscard]] CARBON_HD static constexpr auto
  moment(int q) noexcept -> Moment {
    const auto cq = c(q);
    return {{static_cast<int8_t>(cq.x * cq.x), static_cast<int8_t>(cq.x * cq.y),
             static_cast<int8_t>(cq.x * cq.z), static_cast<int8_t>(cq.y * cq.y),
             static_cast<int8_t>(cq.y * cq.z), static_cast<int8_t>(cq.z * cq.z)}};
  }

  // KBC shear decomposition — direction-dependent linear combination of the
  // off-diagonal stress moments Pi[1] (xy), Pi[2] (xz), Pi[4] (yz) and the
  // axial-deviator moments Nxz = Pi[0] - Pi[5], Nyz = Pi[3] - Pi[5].
  //
  // Closed form derived from c(q):
  //
  //   - cardinal x  (|c.x|=1, others 0)        :  (2*Nxz - Nyz) / 6
  //   - cardinal y  (|c.y|=1, others 0)        :  (-Nxz + 2*Nyz) / 6
  //   - cardinal z  (|c.z|=1, others 0)        :  (-Nxz - Nyz) / 6
  //   - face xy     (|c.x|=1, |c.y|=1, c.z=0)  :  sign(c.x*c.y) * Pi[1] / 4
  //   - face xz     (|c.x|=1, c.y=0, |c.z|=1)  :  sign(c.x*c.z) * Pi[2] / 4
  //   - face yz     (c.x=0, |c.y|=1, |c.z|=1)  :  sign(c.y*c.z) * Pi[4] / 4
  //   - corner / center                         :  0
  //
  // Mirrors Neon's fdecompose_shear lambda in lbmMultiRes/collide.h byte-for-
  // byte under the legacy ordering; expressed via c(q) here so it stays
  // correct under any reordering.
  [[nodiscard]] CARBON_HD static constexpr auto
  shear_decompose(int q,
                  const float Pi[6]) noexcept -> float {
    const auto cq = c(q);
    const int nz = (cq.x != 0 ? 1 : 0) + (cq.y != 0 ? 1 : 0) + (cq.z != 0 ? 1 : 0);
    if (nz == 1) {
      const float Nxz = Pi[0] - Pi[5];
      const float Nyz = Pi[3] - Pi[5];
      if (cq.x != 0) {
        return (2.0F * Nxz - Nyz) / 6.0F;
      }
      if (cq.y != 0) {
        return (-Nxz + 2.0F * Nyz) / 6.0F;
      }
      return (-Nxz - Nyz) / 6.0F;  // cq.z != 0
    }
    if (nz == 2) {
      if (cq.z == 0) {
        return static_cast<float>(cq.x * cq.y) * Pi[1] / 4.0F;
      }
      if (cq.y == 0) {
        return static_cast<float>(cq.x * cq.z) * Pi[2] / 4.0F;
      }
      return static_cast<float>(cq.y * cq.z) * Pi[4] / 4.0F;  // cq.x == 0
    }
    return 0.0F;  // corners (3 nonzero) and center.
  }
};

}  // namespace carbon::bench::lbm
