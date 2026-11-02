#pragma once

// LBM kernel builders for the carbon lid-driven cavity benchmark.
//
// All kernels are templated over a @c Lattice trait struct (@ref D3Q19 in
// d3q19.h, @ref D3Q27 in d3q27.h) that exposes:
//   - static constexpr int Q
//   - static constexpr int kCenter
//   - constexpr static auto c(int q)   -> ::carbon::int8_3d
//   - constexpr static auto opp(int q) -> int
//   - constexpr static auto w(int q)   -> float
// and additionally, for KBC support:
//   - struct Moment { int8_t v[6]; };  static constexpr auto moment(int q) -> Moment;
//   - constexpr static auto shear_decompose(int q, const float Pi[6]) -> float
//
// Three carbon::Kernel factories live here, all templated on Lattice (and
// CollisionPolicy for the iteration kernel):
//
//   make_compute_wall_mask_kernel<Lattice>(...) — one-shot kernel that
//       walks every direction q != kCenter around every Bulk cell and sets
//       bit q in wall_neighbour_bits when the source neighbour at -c[q] is
//       a non-Bulk cell. Lattice-generic — same body works for D3Q19 (19
//       bits) and D3Q27 (27 bits).
//
//   make_iteration_kernel<Lattice, CollisionPolicy>(...) — the hot kernel.
//       For each Bulk cell:
//         1. Pull-stream the Q populations from neighbours, applying
//            half-way bounce-back from this cell's opposite-direction
//            population whenever the wall_neighbour_bits flag is set.
//         2. Compute macroscopic rho and u from the pulled populations.
//         3. Apply CollisionPolicy (BGK or KBC) in-place on the
//            populations.
//         4. Write the post-collision populations back to fOut.
//       Loaded via loader.read_stencil(fIn, lattice_stencil, Lattice) so
//       carbon::Skeleton auto-inserts the z-filtered population halo.
//
//   make_macroscopic_kernel<Lattice>(...) — extracts rho + u into separate
//       fields (one component for rho, three for u). Off the hot path; used
//       by callers that want to dump VTK for visualization.
//
// CollisionPolicy is one of:
//   - BgkCollision: standard single-relaxation-time BGK with
//     omega = 1 / (3 * nu_lb + 0.5). Works for any lattice.
//   - KbcCollision: Karlin-Bösch-Chikatamarla entropic-stabilised collision
//     with per-cell gamma derived from the non-equilibrium moment ratio
//     e0 / e1 (Karlin et al., 2014). The standard formulation targets
//     D3Q27 — the moment matrix and shear-decomposition table on the
//     lattice traits are mandatory.

#include "cell_type.h"

#include <carbon/cuda_annotation.h>
#include <carbon/dense.h>
#include <carbon/kernel.h>

#include <stdint.h>

#include <string>

namespace carbon::bench::lbm {

// Running set of physical knobs derived from the user's CLI: ulb is the
// dimensionless lid velocity, omega the BGK relaxation parameter (also used
// by KBC: beta = omega/2). Carried through both the iteration kernel and
// the macroscopic-extraction kernel as a single value-typed struct so the
// loading lambda only captures one POD.
struct LbmParameters {
  float ulb;
  float omega;
};

// ---------------------------------------------------------------------------
// Collision policies
// ---------------------------------------------------------------------------

// Standard BGK single-relaxation-time collision. f_q^post = (1-omega)*f_q +
// omega * f_q^eq, where the equilibrium is the standard 2nd-order Hermite
// expansion: f_q^eq = w_q * rho * (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 |u|^2).
// Lattice-agnostic — works for D3Q19, D3Q27, or any future Q.
struct BgkCollision {
  template <typename Lattice>
  CARBON_HD static auto
  apply(float omega,
        float rho,
        float ux,
        float uy,
        float uz,
        float pop[Lattice::Q]) noexcept -> void {
    const float usqr = 1.5F * (ux * ux + uy * uy + uz * uz);
    const float one_minus_omega = 1.0F - omega;
#pragma unroll
    for (int q = 0; q < Lattice::Q; ++q) {
      const auto cq = Lattice::c(q);
      const float cu = 3.0F * (static_cast<float>(cq.x) * ux + static_cast<float>(cq.y) * uy +
                               static_cast<float>(cq.z) * uz);
      const float feq = rho * Lattice::w(q) * (1.0F + cu + 0.5F * cu * cu - usqr);
      pop[q] = one_minus_omega * pop[q] + omega * feq;
    }
  }
};

// KBC entropic-stabilised collision (Karlin-Bösch-Chikatamarla, 2014).
// Splits the non-equilibrium part fneq = f - feq into a "shear" component
// (deltaS, governed by the second-moment stress) and a "rest" component
// (deltaH = fneq - deltaS, the higher-order moments). The rest component
// is relaxed with a per-cell, per-iteration multiplier gamma chosen to
// minimise the H-functional along the collision step:
//
//   gamma = 1/beta - (2 - 1/beta) * <S, H>_feq / <H, H>_feq
//
// where the inner product is sum_q (X_q * Y_q / feq_q). beta = omega/2 is
// the relaxation timescale shared with BGK (when gamma == 1/beta the model
// reduces to BGK exactly). The stabilisation gives KBC much better
// behaviour at low viscosity / high Re where BGK becomes oscillatory.
//
// Mirrors Neon/apps/lbmMultiRes/collide.h::collideKBC. Requires the
// Lattice traits to expose @c moment(q) and @c shear_decompose — D3Q27
// in this benchmark.
struct KbcCollision {
  template <typename Lattice>
  CARBON_HD static auto
  apply(float omega,
        float rho,
        float ux,
        float uy,
        float uz,
        float pop[Lattice::Q]) noexcept -> void {
    constexpr int Q = Lattice::Q;
    const float beta = omega * 0.5F;
    const float inv_beta = 1.0F / beta;
    const float usqr = 1.5F * (ux * ux + uy * uy + uz * uz);

    // ---- equilibrium + non-equilibrium ----
    float feq[Q];
    float fneq[Q];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const auto cq = Lattice::c(q);
      const float cu = 3.0F * (static_cast<float>(cq.x) * ux + static_cast<float>(cq.y) * uy +
                               static_cast<float>(cq.z) * uz);
      feq[q] = rho * Lattice::w(q) * (1.0F + cu + 0.5F * cu * cu - usqr);
      fneq[q] = pop[q] - feq[q];
    }

    // ---- second-moment stress Pi[6] = sum_q fneq_q * Mq[i] ----
    float Pi[6] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const auto m = Lattice::moment(q);
#pragma unroll
      for (int i = 0; i < 6; ++i) {
        Pi[i] += fneq[q] * static_cast<float>(m.v[i]);
      }
    }

    // ---- shear component deltaS_q + entropic ratio e0 / e1 ----
    float deltaS[Q];
    float e0 = 0.0F;
    float e1 = 0.0F;
    constexpr float tiny = 1.0e-7F;  // guards 1/feq when feq -> 0 (rare).
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      deltaS[q] = rho * Lattice::shear_decompose(q, Pi);
      const float deltaH = fneq[q] - deltaS[q];
      // Entropic inner products <S, H>_feq and <H, H>_feq.
      const float inv_feq = 1.0F / (feq[q] + tiny);
      e0 += deltaS[q] * deltaH * inv_feq;
      e1 += deltaH * deltaH * inv_feq;
    }

    // ---- per-cell gamma + collision write-back ----
    const float gamma = inv_beta - (2.0F - inv_beta) * e0 / (tiny + e1);
#pragma unroll
    for (int q = 0; q < Q; ++q) {
      const float deltaH = fneq[q] - deltaS[q];
      pop[q] = pop[q] - beta * (2.0F * deltaS[q] + gamma * deltaH);
    }
  }
};

// ---------------------------------------------------------------------------
// Wall-mask precomputation
// ---------------------------------------------------------------------------

// One-shot precomputation of wall_neighbour_bits per Bulk cell. For each
// direction q != kCenter, sets bit q if the source neighbour at -c[q] is
// out of the global domain or holds a non-Bulk cell. Halo / global edge
// cells (!valid) are treated as walls so the cavity boundary cells fall
// straight into bounce-back.
template <typename Grid,
          typename Lattice>
[[nodiscard]] inline auto
make_compute_wall_mask_kernel(Grid& grid,
                              typename Grid::template Field<int32_t,
                                                            1>& cell_in,
                              typename Grid::template Field<int32_t,
                                                            1>& cell_out) -> ::carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return ::carbon::Kernel::compute(
      grid,
      [&cell_in, &cell_out](::carbon::KernelLoader& loader) {
        const auto in = loader.read_stencil(cell_in);
        const auto out = loader.write(cell_out);
        return [in, out] CARBON_HOST_DEVICE(Span s, Idx idx) {
          const int32_t self_packed = in.read(s, idx, 0);
          const Classification self_cls = classification_of(self_packed);
          if (self_cls != Classification::Bulk) {
            out.write(s, idx, 0, self_packed);
            return;
          }
          uint32_t wall_bits = 0U;
#pragma unroll
          for (int q = 0; q < Lattice::Q; ++q) {
            if (q == Lattice::kCenter) {
              continue;
            }
            // Source for direction q under pull streaming sits at -c[q].
            const auto cq = Lattice::c(q);
            const ::carbon::int8_3d src_off{static_cast<int8_t>(-cq.x), static_cast<int8_t>(-cq.y),
                                            static_cast<int8_t>(-cq.z)};
            const auto ngh = in.neighbour(s, idx, src_off, 0);
            const bool is_wall = !ngh.valid || classification_of(ngh.value) != Classification::Bulk;
            if (is_wall) {
              wall_bits |= (1U << static_cast<uint32_t>(q));
            }
          }
          out.write(s, idx, 0, encode_cell(Classification::Bulk, wall_bits));
        };
      },
      "lbm_compute_wall_mask",
      /*shared_mem=*/0U);
}

// ---------------------------------------------------------------------------
// Hot iteration kernel (collide-stream)
// ---------------------------------------------------------------------------

// Q-aligned velocity stencil for lattice-semantic halo: population q pairs with
// stencil.points()[q] == Lattice::c(q). Center at kCenter has c.z == 0 and is
// never sent across z-slab boundaries.
template <typename Lattice>
[[nodiscard]] inline auto
make_lattice_population_stencil() -> ::carbon::Stencil {
  std::vector<::carbon::int32_3d> pts;
  pts.reserve(Lattice::Q);
  for (int q = 0; q < Lattice::Q; ++q) {
    const auto c = Lattice::c(q);
    pts.push_back({c.x, c.y, c.z});
  }
  return ::carbon::Stencil(pts);
}

// Pull-stream source offset for direction q: under pull streaming the cell
// reads its q population from the neighbour at -c[q]. Returned as an
// int8_3d so it can feed Partition::read_pull, the grid-generic @b trusted
// neighbour read (no per-direction validity branch).
//
// The iteration kernel only follows the non-wall branch when wall_bits
// guarantees the source neighbour is an in-domain Bulk cell (set by
// make_compute_wall_mask_kernel: bit q is 0 iff the neighbour at -c[q] is
// both .valid and Bulk), so a bounds check would always succeed there —
// skipping it removes one branch per direction per cell update and is what
// closes the bandwidth gap with hand-rolled LBM kernels. read_pull is the
// primitive that lets Dense and DenseDisg share this hot kernel verbatim:
// each family builds the shifted Idx through its own pitch math (aggregated
// cell shift vs. five-region reclassification) behind the same call, mirroring
// Neon's dPartition::pull.
template <typename Lattice>
[[nodiscard]] CARBON_HD inline auto
pull_offset(int q) noexcept -> ::carbon::int8_3d {
  const auto cq = Lattice::c(q);
  return ::carbon::int8_3d{static_cast<int8_t>(-cq.x), static_cast<int8_t>(-cq.y),
                           static_cast<int8_t>(-cq.z)};
}

// Per-Bulk-cell pipeline: pull-stream Q populations into a register array,
// compute macroscopic rho/u, apply the CollisionPolicy in-place, write Q
// post-collision populations back to fOut. Non-Bulk cells leave fOut alone
// (their populations are kept at the initial equilibrium, matching Neon's
// behaviour).
//
// loader.read_stencil(fIn) records an AccessPattern::Stencil token, which
// causes carbon::Skeleton::compile to insert an auto-halo on fIn before this
// kernel — multi-GPU correctness without manual halo plumbing.
template <typename Grid,
          typename Lattice,
          typename Collision>
[[nodiscard]] inline auto
make_iteration_kernel(Grid& grid,
                      typename Grid::template Field<float,
                                                    Lattice::Q>& f_in,
                      typename Grid::template Field<float,
                                                    Lattice::Q>& f_out,
                      typename Grid::template Field<int32_t,
                                                    1>& cell_field,
                      LbmParameters params,
                      std::string name) -> ::carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return ::carbon::Kernel::compute(
      grid,
      [&f_in, &f_out, &cell_field, params](::carbon::KernelLoader& loader) {
        const auto fin = loader.read_stencil(f_in, make_lattice_population_stencil<Lattice>(),
                                             ::carbon::StencilSemantic::Lattice);
        const auto fout = loader.write(f_out);
        const auto cell = loader.read(cell_field);
        const float omega = params.omega;
        return [fin, fout, cell, omega] CARBON_HOST_DEVICE(Span s, Idx idx) {
          const int32_t packed = cell.read(s, idx, 0);
          if (classification_of(packed) != Classification::Bulk) {
            return;
          }
          const uint32_t wall_bits = wall_bits_of(packed);

          // ---- 1. Pull-stream Q populations into pop[Q] ----
          // For q != kCenter: pop[q] either pulls from the cell at -c[q]
          // (normal stream) or, when wall_bits bit q is set, applies the
          // half-way bounce-back boundary condition. The bounce-back read
          // pulls (a) self.pop[opp(q)] — the standard reflected population —
          // PLUS (b) wall_neighbour.pop[opp(q)]. The wall cell at -c[q] has
          // its opp(q) population pre-seeded by the host-side init: 0 for
          // stationary walls (collapses to plain bounce-back) and
          // -6 * w_opp(q) * ulb * (c_opp(q) . u_lid) for the moving lid
          // (the half-way moving-wall momentum-correction term that drives
          // the cavity flow). Same shape as Neon's
          // lbm-flow-over-sphere PULL_STREAM macro. Without (b) the lid is
          // indistinguishable from a stationary wall and u stays ~0
          // everywhere — the bug that prompted this fix.
          //
          // The wall mask precomputation already filtered out-of-domain
          // and non-Bulk neighbours into wall_bits, so the non-wall branch
          // is provably safe to read through raw offset arithmetic — no
          // per-direction validity check needed.
          float pop[Lattice::Q];
#pragma unroll
          for (int q = 0; q < Lattice::Q; ++q) {
            if (q == Lattice::kCenter) {
              pop[q] = fin.read(s, idx, q);
              continue;
            }
            const ::carbon::int8_3d off = pull_offset<Lattice>(q);
            if (wall_bits & (1U << static_cast<uint32_t>(q))) {
              const int oq = Lattice::opp(q);
              pop[q] = fin.read(s, idx, oq) + fin.read_pull(s, idx, off, oq);
            } else {
              pop[q] = fin.read_pull(s, idx, off, q);
            }
          }

          // ---- 2. Macroscopic rho, u (sum over q with c[q] weighting) ----
          float rho = 0.0F;
          float mx = 0.0F;
          float my = 0.0F;
          float mz = 0.0F;
#pragma unroll
          for (int q = 0; q < Lattice::Q; ++q) {
            const auto cq = Lattice::c(q);
            rho += pop[q];
            mx += static_cast<float>(cq.x) * pop[q];
            my += static_cast<float>(cq.y) * pop[q];
            mz += static_cast<float>(cq.z) * pop[q];
          }
          const float inv_rho = 1.0F / rho;
          const float ux = mx * inv_rho;
          const float uy = my * inv_rho;
          const float uz = mz * inv_rho;

          // ---- 3. Collision (BGK or KBC) in-place on pop[] ----
          Collision::template apply<Lattice>(omega, rho, ux, uy, uz, pop);

      // ---- 4. Write back ----
#pragma unroll
          for (int q = 0; q < Lattice::Q; ++q) {
            fout.write(s, idx, q, pop[q]);
          }
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

// ---------------------------------------------------------------------------
// D3Q19 + BGK hand-unrolled fast-path
// ---------------------------------------------------------------------------
//
// Specialised iteration kernel that exploits two BGK-D3Q19 symmetries that
// the generic make_iteration_kernel above can't: the post-collision write
// pairs (q, opp(q)) share most of their equilibrium (eqopp = eq + rho * w *
// 6 * c.u), and the macroscopic moments factor through sub-sums grouped by
// the sign of each component of c. This brings the kernel onto Neon's
// `lbm-flow-over-sphere` SOTA arithmetic profile — ~2x fewer collision FLOPs
// and ~3x fewer macroscopic FLOPs than the lattice-generic body.
//
// The structural layout (pull-stream + half-way bounce-back + two-pop ping-
// pong + auto-halo via loader.read_stencil) is identical to the generic
// kernel; only the math inside the per-cell body differs. Used only for the
// (D3Q19, BGK) combo — D3Q27 and KBC stay on the lattice-generic path
// because their symmetry tables are different (D3Q27) or the collision step
// itself is not symmetry-factorisable (KBC).
template <typename Grid>
[[nodiscard]] inline auto
make_iteration_kernel_d3q19_bgk_fast(Grid& grid,
                                     typename Grid::template Field<float,
                                                                   D3Q19::Q>& f_in,
                                     typename Grid::template Field<float,
                                                                   D3Q19::Q>& f_out,
                                     typename Grid::template Field<int32_t,
                                                                   1>& cell_field,
                                     LbmParameters params,
                                     std::string name) -> ::carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return ::carbon::Kernel::compute(
      grid,
      [&f_in, &f_out, &cell_field, params](::carbon::KernelLoader& loader) {
        const auto fin = loader.read_stencil(f_in, make_lattice_population_stencil<D3Q19>(),
                                             ::carbon::StencilSemantic::Lattice);
        const auto fout = loader.write(f_out);
        const auto cell = loader.read(cell_field);
        const float omega = params.omega;
        return [fin, fout, cell, omega] CARBON_HOST_DEVICE(Span s, Idx idx) {
          const int32_t packed = cell.read(s, idx, 0);
          if (classification_of(packed) != Classification::Bulk) {
            return;
          }
          const uint32_t wall_bits = wall_bits_of(packed);

          // ---- 1. Pull-stream — same logic as the generic body ----
          // See the matching comment in make_iteration_kernel: bounce-back
          // adds the wall neighbour's pre-seeded opp(q) population so the
          // moving lid's -6*w*ulb*(c.u_lid) correction actually feeds in.
          float pop[D3Q19::Q];
#pragma unroll
          for (int q = 0; q < D3Q19::Q; ++q) {
            if (q == D3Q19::kCenter) {
              pop[q] = fin.read(s, idx, q);
              continue;
            }
            const ::carbon::int8_3d off = pull_offset<D3Q19>(q);
            if (wall_bits & (1U << static_cast<uint32_t>(q))) {
              const int oq = D3Q19::opp(q);
              pop[q] = fin.read(s, idx, oq) + fin.read_pull(s, idx, off, oq);
            } else {
              pop[q] = fin.read_pull(s, idx, off, q);
            }
          }

          // ---- 2. Factored macroscopic ----
          // Sub-sums grouped by the sign of each component of c. Indices
          // pair with the new c.z-bucketed q-id table in d3q19.h:
          //   c.x = -1 : {1, 5, 7, 8, 16}
          //   c.x = +1 : {2, 10, 11, 13, 17}
          //   c.y = -1 : {3, 6, 7, 10, 14}
          //   c.y = +1 : {4, 8, 11, 12, 15}
          //   c.z = -1 : {0, 1, 2, 3, 4}     (entire bucket [0, 5))
          //   c.z = +1 : {14, 15, 16, 17, 18} (entire bucket [14, 19))
          // The c.z sums are now full contiguous-bucket scans, mirroring
          // the disg lattice halo's per-bucket fused transfers.
          const float X_M1 = pop[1] + pop[5] + pop[7] + pop[8] + pop[16];
          const float X_P1 = pop[2] + pop[10] + pop[11] + pop[13] + pop[17];
          const float Y_M1 = pop[3] + pop[6] + pop[7] + pop[10] + pop[14];
          const float Y_P1 = pop[4] + pop[8] + pop[11] + pop[12] + pop[15];
          const float Z_M1 = pop[0] + pop[1] + pop[2] + pop[3] + pop[4];
          const float Z_P1 = pop[14] + pop[15] + pop[16] + pop[17] + pop[18];

          // rho = sum over all q. Cheaper to add up the three z-buckets.
          const float Z_0 =
              pop[5] + pop[6] + pop[7] + pop[8] + pop[9] + pop[10] + pop[11] + pop[12] + pop[13];
          const float rho = Z_M1 + Z_0 + Z_P1;
          const float inv_rho = 1.0F / rho;
          const float ux = (X_P1 - X_M1) * inv_rho;
          const float uy = (Y_P1 - Y_M1) * inv_rho;
          const float uz = (Z_P1 - Z_M1) * inv_rho;
          const float usqr = 1.5F * (ux * ux + uy * uy + uz * uz);

          // ---- 3. Symmetric BGK collision ----
          // Compute equilibria for q in [0, 9] (the "first half" + center)
          // and derive q in [10, 18] via the BGK symmetry
          //   feq(c) - feq(-c) = 6 * rho * w * c.u .
          // Under the new palindromic ordering c(18 - q) = -c(q), so
          //   eq_(18 - q) = eq_q + 6 * rho * w_q * c(18 - q).u .
          // (c.u for the BK velocity of each pair is filled in below.)
          constexpr float c1over18 = 1.0F / 18.0F;
          constexpr float c1over36 = 1.0F / 36.0F;
          constexpr float c1over3 = 1.0F / 3.0F;

          // Off-axial projections used by the face-diagonal pairs.
          const float ck_xy = ux + uy;  // (1, 1, 0) / (-1, -1, 0)
          const float ck_xz = ux + uz;  // (1, 0, 1) / (-1, 0, -1)
          const float ck_yz = uy + uz;  // (0, 1, 1) / (0, -1, -1)
          const float dk_xy = ux - uy;  // (1, -1, 0) / (-1, 1, 0)
          const float dk_xz = ux - uz;  // (1, 0, -1) / (-1, 0, 1)
          const float dk_yz = uy - uz;  // (0, 1, -1) / (0, -1, 1)

          // First-half equilibria. c.u for q in [0, 9]:
          //   q=0  c=(0,0,-1)   c.u = -uz
          //   q=1  c=(-1,0,-1)  c.u = -ck_xz
          //   q=2  c=(1,0,-1)   c.u = +dk_xz
          //   q=3  c=(0,-1,-1)  c.u = -ck_yz
          //   q=4  c=(0,1,-1)   c.u = +dk_yz
          //   q=5  c=(-1,0,0)   c.u = -ux
          //   q=6  c=(0,-1,0)   c.u = -uy
          //   q=7  c=(-1,-1,0)  c.u = -ck_xy
          //   q=8  c=(-1,1,0)   c.u = -dk_xy
          //   q=9  center       c.u = 0
          const float eq_0 = rho * c1over18 * (1.0F - 3.0F * uz + 4.5F * uz * uz - usqr);
          const float eq_1 = rho * c1over36 * (1.0F - 3.0F * ck_xz + 4.5F * ck_xz * ck_xz - usqr);
          const float eq_2 = rho * c1over36 * (1.0F + 3.0F * dk_xz + 4.5F * dk_xz * dk_xz - usqr);
          const float eq_3 = rho * c1over36 * (1.0F - 3.0F * ck_yz + 4.5F * ck_yz * ck_yz - usqr);
          const float eq_4 = rho * c1over36 * (1.0F + 3.0F * dk_yz + 4.5F * dk_yz * dk_yz - usqr);
          const float eq_5 = rho * c1over18 * (1.0F - 3.0F * ux + 4.5F * ux * ux - usqr);
          const float eq_6 = rho * c1over18 * (1.0F - 3.0F * uy + 4.5F * uy * uy - usqr);
          const float eq_7 = rho * c1over36 * (1.0F - 3.0F * ck_xy + 4.5F * ck_xy * ck_xy - usqr);
          const float eq_8 = rho * c1over36 * (1.0F - 3.0F * dk_xy + 4.5F * dk_xy * dk_xy - usqr);
          const float eq_9 = rho * c1over3 * (1.0F - usqr);

          // Second-half equilibria via the symmetry. c.u for q in [10, 18]:
          //   q=10 c=(1,-1,0)   c.u = +dk_xy
          //   q=11 c=(1,1,0)    c.u = +ck_xy
          //   q=12 c=(0,1,0)    c.u = +uy
          //   q=13 c=(1,0,0)    c.u = +ux
          //   q=14 c=(0,-1,1)   c.u = -dk_yz
          //   q=15 c=(0,1,1)    c.u = +ck_yz
          //   q=16 c=(-1,0,1)   c.u = -dk_xz
          //   q=17 c=(1,0,1)    c.u = +ck_xz
          //   q=18 c=(0,0,1)    c.u = +uz
          const float eq_10 = eq_8 + rho * c1over36 * 6.0F * dk_xy;
          const float eq_11 = eq_7 + rho * c1over36 * 6.0F * ck_xy;
          const float eq_12 = eq_6 + rho * c1over18 * 6.0F * uy;
          const float eq_13 = eq_5 + rho * c1over18 * 6.0F * ux;
          const float eq_14 = eq_4 - rho * c1over36 * 6.0F * dk_yz;
          const float eq_15 = eq_3 + rho * c1over36 * 6.0F * ck_yz;
          const float eq_16 = eq_2 - rho * c1over36 * 6.0F * dk_xz;
          const float eq_17 = eq_1 + rho * c1over36 * 6.0F * ck_xz;
          const float eq_18 = eq_0 + rho * c1over18 * 6.0F * uz;

          const float one_minus_omega = 1.0F - omega;

          // ---- 4. Write back ----
          fout.write(s, idx, 0, one_minus_omega * pop[0] + omega * eq_0);
          fout.write(s, idx, 1, one_minus_omega * pop[1] + omega * eq_1);
          fout.write(s, idx, 2, one_minus_omega * pop[2] + omega * eq_2);
          fout.write(s, idx, 3, one_minus_omega * pop[3] + omega * eq_3);
          fout.write(s, idx, 4, one_minus_omega * pop[4] + omega * eq_4);
          fout.write(s, idx, 5, one_minus_omega * pop[5] + omega * eq_5);
          fout.write(s, idx, 6, one_minus_omega * pop[6] + omega * eq_6);
          fout.write(s, idx, 7, one_minus_omega * pop[7] + omega * eq_7);
          fout.write(s, idx, 8, one_minus_omega * pop[8] + omega * eq_8);
          fout.write(s, idx, 9, one_minus_omega * pop[9] + omega * eq_9);
          fout.write(s, idx, 10, one_minus_omega * pop[10] + omega * eq_10);
          fout.write(s, idx, 11, one_minus_omega * pop[11] + omega * eq_11);
          fout.write(s, idx, 12, one_minus_omega * pop[12] + omega * eq_12);
          fout.write(s, idx, 13, one_minus_omega * pop[13] + omega * eq_13);
          fout.write(s, idx, 14, one_minus_omega * pop[14] + omega * eq_14);
          fout.write(s, idx, 15, one_minus_omega * pop[15] + omega * eq_15);
          fout.write(s, idx, 16, one_minus_omega * pop[16] + omega * eq_16);
          fout.write(s, idx, 17, one_minus_omega * pop[17] + omega * eq_17);
          fout.write(s, idx, 18, one_minus_omega * pop[18] + omega * eq_18);
        };
      },
      std::move(name),
      /*shared_mem=*/0U);
}

// ---------------------------------------------------------------------------
// Optional macroscopic-extraction kernel (rho + u out)
// ---------------------------------------------------------------------------

// Off the hot path. Projects rho + u out of the current population field
// into separate (1, 3)-component fields. Used when callers want to dump
// VTK / probe profiles between benchmark runs. Same generic Q-loop the
// iteration kernel uses for the moment computation.
template <typename Grid,
          typename Lattice>
[[nodiscard]] inline auto
make_macroscopic_kernel(Grid& grid,
                        typename Grid::template Field<float,
                                                      Lattice::Q>& f_in,
                        typename Grid::template Field<int32_t,
                                                      1>& cell_field,
                        typename Grid::template Field<float,
                                                      1>& rho_field,
                        typename Grid::template Field<float,
                                                      3>& u_field,
                        LbmParameters params) -> ::carbon::Kernel {
  using Span = typename Grid::Span;
  using Idx = typename Grid::Idx;
  return ::carbon::Kernel::compute(
      grid,
      [&f_in, &cell_field, &rho_field, &u_field, params](::carbon::KernelLoader& loader) {
        const auto fin = loader.read(f_in);
        const auto cell = loader.read(cell_field);
        const auto rho_w = loader.write(rho_field);
        const auto u_w = loader.write(u_field);
        const float ulb = params.ulb;
        return [fin, cell, rho_w, u_w, ulb] CARBON_HOST_DEVICE(Span s, Idx idx) {
          const int32_t packed = cell.read(s, idx, 0);
          const Classification cls = classification_of(packed);
          if (cls == Classification::Bulk) {
            float rho = 0.0F;
            float mx = 0.0F;
            float my = 0.0F;
            float mz = 0.0F;
#pragma unroll
            for (int q = 0; q < Lattice::Q; ++q) {
              const auto cq = Lattice::c(q);
              const float fq = fin.read(s, idx, q);
              rho += fq;
              mx += static_cast<float>(cq.x) * fq;
              my += static_cast<float>(cq.y) * fq;
              mz += static_cast<float>(cq.z) * fq;
            }
            const float inv_rho = 1.0F / rho;
            rho_w.write(s, idx, 0, rho);
            u_w.write(s, idx, 0, mx * inv_rho);
            u_w.write(s, idx, 1, my * inv_rho);
            u_w.write(s, idx, 2, mz * inv_rho);
          } else if (cls == Classification::MovingWall) {
            // Lid moves with u = (ulb, 0, 0); rho = 1 by construction.
            rho_w.write(s, idx, 0, 1.0F);
            u_w.write(s, idx, 0, ulb);
            u_w.write(s, idx, 1, 0.0F);
            u_w.write(s, idx, 2, 0.0F);
          } else {
            rho_w.write(s, idx, 0, 1.0F);
            u_w.write(s, idx, 0, 0.0F);
            u_w.write(s, idx, 1, 0.0F);
            u_w.write(s, idx, 2, 0.0F);
          }
        };
      },
      "lbm_macroscopic",
      /*shared_mem=*/0U);
}

}  // namespace carbon::bench::lbm
