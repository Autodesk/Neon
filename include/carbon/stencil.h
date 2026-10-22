#pragma once

#include <carbon/export.h>
#include <carbon/integer3d.h>

#include <cstdint>
#include <string_view>
#include <vector>

namespace carbon {

/**
 * Halo-update semantic flag, mirroring @c Neon::set::StencilSemantic .
 *
 * @c Standard treats every cell component as opaque scalar data and copies the full
 * boundary plane: each adjacency pair moves @c Components · X · Y elements per direction
 * regardless of stencil shape.
 *
 * @c Lattice is the LBM-shaped variant: the field's @c Components are interpreted as
 * lattice populations whose @ref Stencil offsets are their discrete velocities @c c_q .
 * Only populations whose @c c_q.z matches the transfer direction are sent across each
 * z-slab boundary — for D3Q19 that's 5/19, for D3Q27 9/27. Carries an additional
 * pre-condition: @c stencil.size() must equal the field's @c Components and entries are
 * paired by index @c (population q ↔ stencil.points()[q]) .
 */
enum struct StencilSemantic : int {
  Standard = 0,
  Lattice = 1
};

[[nodiscard]] CARBON_API auto
to_string(StencilSemantic semantic) noexcept -> std::string_view;

/**
 * Result of @ref Stencil::lattice_profile : counts and longest-run statistics over each
 * @c c_q.z bucket. Used by lattice-halo descriptors to decide how many fused transfers
 * the halo issues per direction (one per maximal run of consecutive same-sign-z
 * populations) and by tests / benchmarks to validate that a population ordering is
 * "lattice-friendly" (i.e. all @c c_q.z>0 entries are contiguous, same for @c c_q.z<0 ).
 */
struct LatticeProfile {
  int positive_z{};         /**< Number of stencil points with @c c.z > 0 . */
  int negative_z{};         /**< Number of stencil points with @c c.z < 0 . */
  int zero_z{};             /**< Number of stencil points with @c c.z == 0 . */
  int run_count_positive{}; /**< Maximal-run count over @c c.z > 0 entries (0 if none). */
  int run_count_negative{}; /**< Maximal-run count over @c c.z < 0 entries (0 if none). */
  int max_run_positive{};   /**< Longest contiguous run of @c c.z > 0 entries. */
  int max_run_negative{};   /**< Longest contiguous run of @c c.z < 0 entries. */
  /**
   * @c true iff every @c c.z > 0 entry is in a single contiguous run AND every @c c.z < 0
   * entry is in a single contiguous run. When this holds, the lattice halo on the disg
   * layout collapses to exactly two transfers per adjacency pair (regardless of the
   * number of active populations) — same call count as the @c Standard halo with
   * @c positive_z + negative_z bytes total per direction instead of @c Components.
   */
  bool fully_groupable{};
};

/**
 * A stencil is a set of 3D integer offsets from a cell — e.g. the classic 7- / 19- / 27-point
 * neighbourhoods. Offsets are stored as @ref int8_3d so they pair directly with
 * @c carbon::detail::dense::Partition::neighbour and stay cheap to keep in registers; this
 * caps each axis at @c [-127, +127], which covers every realistic discretisation (sub-block
 * halos, wide-kernel stencils up to ~256 cells across).
 *
 * The constructor takes @ref int32_3d offsets for ergonomics (small literals like
 * @c {1, 0, 0} are int-typed) and validates each axis fits @c int8_t before storing; out-of
 * -range offsets throw rather than silently truncate.
 *
 * Purely a host-side descriptor: `std::vector` is host-only, so this type is not `CARBON_HD`
 * and is not readable from device code. Use it to classify partitions and size halos; pass
 * the resulting offsets / radius to device code by value.
 *
 * Mirrors @c Neon::domain::Stencil, stripped to the parts Carbon currently needs (no
 * @c find / @c addPoint / named factories yet).
 */
class CARBON_API Stencil {
public:
  /**
   * Empty stencil.
   */
  Stencil() = default;

  /**
   * Build a stencil from @p points. Each axis of every offset must fit in @c int8_t
   * (@c [-127, +127]); offsets outside that range throw @c std::out_of_range with the
   * offending entry logged. The check happens on the raw input @b before
   * @p remove_center is applied.
   *
   * @param points          Offsets relative to a cell; copied and converted to @ref int8_3d.
   * @param remove_center   If true, any @c (0, 0, 0) entries in @p points are dropped before
   *                        storing. Defaults to @c false (points are kept as provided).
   */
  explicit Stencil(const std::vector<int32_3d>& points,
                   bool remove_center = false);

  /**
   * Stored offsets, in insertion order (after optional center filtering). Always @ref int8_3d
   * — the stencil format pairs directly with @c Partition::neighbour without further
   * conversion.
   */
  [[nodiscard]] auto
  points() const noexcept -> const std::vector<int8_3d>&;

  /** Number of offsets stored. */
  [[nodiscard]] auto
  size() const noexcept -> size_t;

  /** True iff @ref size() == 0. */
  [[nodiscard]] auto
  empty() const noexcept -> bool;

  /**
   * Chebyshev radius of the stencil: @c max(|p.x|, |p.y|, |p.z|) over all stored points,
   * or 0 if the stencil is empty. Including a @c (0, 0, 0) center does not change the result.
   *
   * Computed once at construction (and after the optional center filter) and cached, so this
   * accessor is a constant-time read. Matches @c Neon::domain::Stencil::getRadius semantics and
   * the `int32_t` return type (a @c [0, 127] @c int8_t value widens cleanly).
   */
  [[nodiscard]] auto
  radius() const noexcept -> std::int32_t;

  /**
   * Bucket the stored points by @c c.z sign and report run-length statistics — the
   * input the lattice-halo descriptor uses to count fused transfers (one per maximal
   * run of consecutive same-sign-z entries on a disg layout, one per kept entry on
   * the aggregated layout).
   *
   * Cheap (linear in @ref size ) and not cached: cardinality counts are small and the
   * call only happens at halo-descriptor construction (or in tests / benchmarks).
   */
  [[nodiscard]] auto
  lattice_profile() const noexcept -> LatticeProfile;

  /**
   * Result of @ref reorder_for_lattice_fusion : a permuted offset list that's
   * @c LatticeProfile::fully_groupable @b and the index permutation that produced
   * it.
   *
   * Conventions: @c permutation[new_index] @c == @c old_index . User code that has
   * a parallel weight / population table @c old_w of size @c offsets.size() builds
   * the matching reordered table as
   * @code
   * std::vector<float> new_w(reordered.size());
   * for (size_t i = 0; i < new_w.size(); ++i) new_w[i] = old_w[permutation[i]];
   * @endcode
   *
   * @c reordered.size() @c == @c permutation.size() @c == @c offsets.size() (no
   * entries are dropped; this is a pure rearrangement).
   */
  struct LatticeReorderResult {
    /** Permuted offset list, all @c c.z<0 first, then @c c.z==0 , then @c c.z>0 . */
    std::vector<int32_3d> reordered{};
    /** @c permutation[new_idx] @c == @c old_idx . Same length as @ref reordered . */
    std::vector<size_t> permutation{};
  };

  /**
   * Permute a user's velocity / offset list into a lattice-halo-friendly ordering:
   * all entries with @c c.z<0 first (in their original relative order), then
   * @c c.z==0 entries, then @c c.z>0 entries. The result is guaranteed to be
   * @c LatticeProfile::fully_groupable -- the lattice halo on a @c DenseDisg layout
   * collapses to exactly two transfers per adjacency pair (one fused upward, one
   * fused downward) when paired with this ordering.
   *
   * Stable within each @c c.z bucket: the relative order of two entries with the
   * same @c c.z sign in @p offsets is preserved in @c reordered . This matters when
   * an LBM lattice has additional internal structure (e.g. D3Q19's symmetry under
   * @c opp(q) = Q-1-q ) that the user wants to preserve through the reordering.
   *
   * @param offsets  User's velocity / offset list. Same shape as the @c Stencil
   *                 constructor input; @c int32_3d for ergonomic small-literal init.
   *
   * @return @ref LatticeReorderResult with the permuted offsets and the matching
   *         index permutation. Empty input returns empty result.
   *
   * @note Pure host-side helper: no validation against @c int8_t bounds happens
   *       here (that runs in the @c Stencil ctor when the user feeds the result
   *       back in).
   */
  [[nodiscard]] static auto
  reorder_for_lattice_fusion(const std::vector<int32_3d>& offsets) -> LatticeReorderResult;

private:
  /** Computes Chebyshev radius from @p points (used by the constructor to seed @c radius_). */
  static auto
  compute_radius_(const std::vector<int8_3d>& points) noexcept -> std::int32_t;

  std::vector<int8_3d> points_{};
  std::int32_t radius_{0};
};

}  // namespace carbon
