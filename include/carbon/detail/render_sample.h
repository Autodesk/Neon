#pragma once

/**
 * Grid-generic point sampling: "what is the value of this field at this world position",
 * answered uniformly for all four grid families (Dense, DenseDisg, Block, MultiResBlock).
 *
 * Ray marching, volumetric advection, surface probes and particle tracing all reduce to the
 * same question, and each of them would otherwise have to re-derive the world-to-cell map
 * per family. A @c Sampler is that map, packaged as a small device-capturable value:
 *
 * @code
 *   const auto s = render::make_sampler(span, [] CARBON_HD (const Span& sp, const Idx& i) {
 *     return static_cast<float>(part.read(sp, i, 0));
 *   });
 *   const render::FieldSample v = s.sample(world);         // trilinear
 *   const render::FieldGradientSample g = s.gradient(world);
 * @endcode
 *
 * Every sampler exposes the same surface (@ref DenseSampler documents it once):
 * @c sample / @c sample_nearest / @c gradient / @c spacing / @c world_bounds_min /
 * @c world_bounds_max / @c owned_z_lo / @c owned_z_hi .
 *
 * @par Reading the value
 *
 * The cell value comes from a caller-supplied @c value_fn rather than a @c Partition , so
 * one sampler serves scalar fields, single components of vector fields, and derived
 * quantities computed on the fly (speed, Q-criterion, a free-stream-subtracted wake scalar)
 * with no extra types. The Cartesian families call @c value_fn(span, idx) ; the
 * multi-resolution tower calls @c value_fn(level, span, idx) because the level is part of
 * the answer there.
 *
 * @par Ownership, and why @c sample can legitimately return invalid
 *
 * A sampler sees exactly one GPU's slab. Outside that slab's addressable storage the sample
 * is reported invalid rather than silently clamped to a far-away cell, and the caller is
 * expected to composite per-GPU results (this is what the ray-march segment merge already
 * does, and @c owned_z_lo / @c owned_z_hi give a caller the band to clip against). What
 * counts as addressable differs by family and is documented per sampler; the sparse
 * families are the interesting case, since a missing @b corner is normal there and is
 * handled by falling back to the centre cell instead of bleeding a zero into the result.
 */

#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/detail/block/locator.h>
#include <carbon/detail/block/span.h>
#include <carbon/detail/dense/span.h>
#include <carbon/detail/dense_disg/span.h>
#include <carbon/detail/mgrid/locator.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>

#include <stdint.h>
#include <cmath>

namespace carbon::render {

/** Scalar sample plus a validity flag (see the ownership note in the file header). */
struct FieldSample {
  float value{0.0F};
  bool valid{false};
};

/** Gradient sample plus a validity flag; invalid when any stencil arm was unaddressable. */
struct FieldGradientSample {
  float32_3d value{};
  bool valid{false};
};

namespace trilinear_detail {

[[nodiscard]] CARBON_HD inline auto
clamp_grid_index(int c,
                 int64_t extent) noexcept -> int {
  if (c < 0) {
    return 0;
  }
  if (c >= extent) {
    return static_cast<int>(extent - 1);
  }
  return c;
}

/**
 * Trilinear blend of eight corners on a Cartesian lattice, clamping x/y into
 * @c [0, ge_*) and z into the caller's addressable storage band @c [z_storage_lo,
 * z_storage_hi) . Rejects the sample outright when the base corner is outside x/y or when
 * the z pair does not overlap the storage band at all — a clamped-from-far-away value is
 * worse than no value, since the caller can composite another GPU's answer instead.
 */
template <typename CornerReadFn>
[[nodiscard]] CARBON_HD inline auto
sample_clamped(int ix0,
               int iy0,
               int iz0,
               float tx,
               float ty,
               float tz,
               int64_t ge_x,
               int64_t ge_y,
               int64_t z_storage_lo,
               int64_t z_storage_hi,
               CornerReadFn read_corner) -> FieldSample {
  if (ix0 < 0 || iy0 < 0 || iz0 < 0 || ix0 >= ge_x || iy0 >= ge_y) {
    return {};
  }
  if (iz0 + 1 < z_storage_lo || iz0 >= z_storage_hi) {
    return {};
  }

  auto corner = [&](int gx, int gy, int gz) -> float {
    gx = clamp_grid_index(gx, ge_x);
    gy = clamp_grid_index(gy, ge_y);
    if (gz < static_cast<int>(z_storage_lo)) {
      gz = static_cast<int>(z_storage_lo);
    } else if (gz >= static_cast<int>(z_storage_hi)) {
      gz = static_cast<int>(z_storage_hi) - 1;
    }
    return read_corner(gx, gy, gz);
  };

  const float c000 = corner(ix0, iy0, iz0);
  const float c100 = corner(ix0 + 1, iy0, iz0);
  const float c010 = corner(ix0, iy0 + 1, iz0);
  const float c110 = corner(ix0 + 1, iy0 + 1, iz0);
  const float c001 = corner(ix0, iy0, iz0 + 1);
  const float c101 = corner(ix0 + 1, iy0, iz0 + 1);
  const float c011 = corner(ix0, iy0 + 1, iz0 + 1);
  const float c111 = corner(ix0 + 1, iy0 + 1, iz0 + 1);

  const float c00 = c000 * (1.0F - tx) + c100 * tx;
  const float c10 = c010 * (1.0F - tx) + c110 * tx;
  const float c01 = c001 * (1.0F - tx) + c101 * tx;
  const float c11 = c011 * (1.0F - tx) + c111 * tx;
  const float c0 = c00 * (1.0F - ty) + c10 * ty;
  const float c1 = c01 * (1.0F - ty) + c11 * ty;
  return {c0 * (1.0F - tz) + c1 * tz, true};
}

/** Trilinear blend of eight already-fetched corner values. */
[[nodiscard]] CARBON_HD inline auto
blend(const float c[8],
      float tx,
      float ty,
      float tz) noexcept -> float {
  const float c00 = c[0] * (1.0F - tx) + c[1] * tx;
  const float c10 = c[2] * (1.0F - tx) + c[3] * tx;
  const float c01 = c[4] * (1.0F - tx) + c[5] * tx;
  const float c11 = c[6] * (1.0F - tx) + c[7] * tx;
  const float c0 = c00 * (1.0F - ty) + c10 * ty;
  const float c1 = c01 * (1.0F - ty) + c11 * ty;
  return c0 * (1.0F - tz) + c1 * tz;
}

}  // namespace trilinear_detail

/**
 * Central-difference gradient built from a sampler's own @c sample . Shared by every
 * family so the stencil, the step and the all-arms-must-be-valid rule stay identical
 * across them; each sampler's @c gradient member is a one-line call into this.
 */
template <typename Sampler>
[[nodiscard]] CARBON_HD inline auto
central_difference_gradient(const Sampler& sampler,
                            float32_3d world,
                            float h) -> FieldGradientSample {
  const float inv_2h = 1.0F / (2.0F * h);
  const FieldSample xp = sampler.sample({world.x + h, world.y, world.z});
  const FieldSample xm = sampler.sample({world.x - h, world.y, world.z});
  const FieldSample yp = sampler.sample({world.x, world.y + h, world.z});
  const FieldSample ym = sampler.sample({world.x, world.y - h, world.z});
  const FieldSample zp = sampler.sample({world.x, world.y, world.z + h});
  const FieldSample zm = sampler.sample({world.x, world.y, world.z - h});
  if (!xp.valid || !xm.valid || !yp.valid || !ym.valid || !zp.valid || !zm.valid) {
    return {};
  }
  return {
      {(xp.value - xm.value) * inv_2h, (yp.value - ym.value) * inv_2h,
       (zp.value - zm.value) * inv_2h},
      true,
  };
}

/**
 * Sampler over an aggregated dense slab (@c carbon::Dense ).
 *
 * Node-centred, matching @c dense::Span 's own convention: cell @c (i,j,k) sits at
 * @c origin + (i,j,k) · spacing . Corners are clamped in x/y to the global extent and in z
 * to the slab's haloed storage band, so a sample one cell outside the owned band still
 * reads the halo this GPU holds rather than being rejected. The z band is additionally
 * intersected with the global extent: the first slab's lower halo and the last slab's upper
 * halo are in-buffer but hold nothing meaningful under @c BoundaryMode::Open , and clamping
 * into them would feed uninitialised cells into the blend.
 *
 * @tparam Span    @c carbon::detail::dense::Span .
 * @tparam ValueFn @c (const Span&, const Idx&) -> convertible-to-float .
 */
template <typename Span, typename ValueFn>
struct DenseSampler {
  Span span{};
  ValueFn value_fn{};

  [[nodiscard]] CARBON_HD auto
  spacing() const noexcept -> float {
    return span.spacing();
  }

  [[nodiscard]] CARBON_HD auto
  world_bounds_min() const noexcept -> float32_3d {
    return span.origin();
  }

  [[nodiscard]] CARBON_HD auto
  world_bounds_max() const noexcept -> float32_3d {
    const float32_3d o = span.origin();
    const int64_3d ge = span.global_extent();
    const float sp = span.spacing();
    return {
        o.x + static_cast<float>(ge.x) * sp,
        o.y + static_cast<float>(ge.y) * sp,
        o.z + static_cast<float>(ge.z) * sp,
    };
  }

  /** Global z of this slab's first owned cell. Buffer z 0 is the lower halo, not owned. */
  [[nodiscard]] CARBON_HD auto
  owned_z_begin() const noexcept -> int64_t {
    return span.z_offset() + span.halo_z();
  }

  /** Lower world-z of this slab's @b owned band (halo excluded). */
  [[nodiscard]] CARBON_HD auto
  owned_z_lo() const noexcept -> float {
    return span.origin().z + static_cast<float>(owned_z_begin()) * span.spacing();
  }

  [[nodiscard]] CARBON_HD auto
  owned_z_hi() const noexcept -> float {
    return span.origin().z +
           static_cast<float>(owned_z_begin() + span.logic_extent().z) * span.spacing();
  }

  /** Global-z band this slab can address: its whole storage buffer, clipped to the domain. */
  [[nodiscard]] CARBON_HD auto
  addressable_z_lo() const noexcept -> int64_t {
    const int64_t lo = span.z_offset();
    return lo > 0 ? lo : 0;
  }

  [[nodiscard]] CARBON_HD auto
  addressable_z_hi() const noexcept -> int64_t {
    const int64_t hi = span.z_offset() + span.haloed_extent().z;
    const int64_t ge_z = span.global_extent().z;
    return hi < ge_z ? hi : ge_z;
  }

  [[nodiscard]] CARBON_HD auto
  sample(float32_3d world) const -> FieldSample {
    const float sp = span.spacing();
    const float32_3d origin = span.origin();
    const int64_3d ge = span.global_extent();

    const float fx = (world.x - origin.x) / sp;
    const float fy = (world.y - origin.y) / sp;
    const float fz = (world.z - origin.z) / sp;

    const int ix0 = static_cast<int>(std::floor(fx));
    const int iy0 = static_cast<int>(std::floor(fy));
    const int iz0 = static_cast<int>(std::floor(fz));

    return trilinear_detail::sample_clamped(
        ix0, iy0, iz0, fx - static_cast<float>(ix0), fy - static_cast<float>(iy0),
        fz - static_cast<float>(iz0), ge.x, ge.y, addressable_z_lo(), addressable_z_hi(),
        [&](int gx, int gy, int gz) -> float {
          const auto idx = span.make_idx(static_cast<int64_t>(gx), static_cast<int64_t>(gy),
                                         static_cast<int64_t>(gz) - span.z_offset());
          return static_cast<float>(value_fn(span, idx));
        });
  }

  [[nodiscard]] CARBON_HD auto
  sample_nearest(float32_3d world) const -> FieldSample {
    const float sp = span.spacing();
    const float32_3d origin = span.origin();
    const int64_3d ge = span.global_extent();

    const int64_t gx = static_cast<int64_t>(std::lround((world.x - origin.x) / sp));
    const int64_t gy = static_cast<int64_t>(std::lround((world.y - origin.y) / sp));
    const int64_t gz = static_cast<int64_t>(std::lround((world.z - origin.z) / sp));
    if (gx < 0 || gy < 0 || gx >= ge.x || gy >= ge.y) {
      return {};
    }
    if (gz < addressable_z_lo() || gz >= addressable_z_hi()) {
      return {};
    }
    const auto idx = span.make_idx(gx, gy, gz - span.z_offset());
    return {static_cast<float>(value_fn(span, idx)), true};
  }

  [[nodiscard]] CARBON_HD auto
  gradient(float32_3d world) const -> FieldGradientSample {
    return central_difference_gradient(*this, world, 0.5F * span.spacing());
  }
};

/**
 * Sampler over a disaggregated dense slab (@c carbon::DenseDisg ).
 *
 * Same Cartesian domain and same node-centred convention as @ref DenseSampler ; what
 * differs is cell addressing, which goes through @c Span::set_and_validate because the disg
 * layout splits z into five physically disjoint storage regions and the region fields have
 * to be recomputed per cell.
 *
 * The addressable z band is the @b owned band only. Unlike Dense, the disg
 * @c set_and_validate does not expose the ghost planes through the @c Standard view, so a
 * sample in the outermost half-cell of a slab clamps to the owned edge instead of reading
 * this GPU's halo copy. The seam is still correct in a multi-GPU render because the
 * neighbouring slab owns that region and its segment is composited in.
 *
 * @tparam Span    @c carbon::detail::dense_disg::Span .
 * @tparam ValueFn @c (const Span&, const Idx&) -> convertible-to-float .
 */
template <typename Span, typename ValueFn>
struct DenseDisgSampler {
  Span span{};
  ValueFn value_fn{};

  [[nodiscard]] CARBON_HD auto
  spacing() const noexcept -> float {
    return span.spacing();
  }

  [[nodiscard]] CARBON_HD auto
  world_bounds_min() const noexcept -> float32_3d {
    return span.origin();
  }

  [[nodiscard]] CARBON_HD auto
  world_bounds_max() const noexcept -> float32_3d {
    const float32_3d o = span.origin();
    const int64_3d ge = span.global_extent();
    const float sp = span.spacing();
    return {
        o.x + static_cast<float>(ge.x) * sp,
        o.y + static_cast<float>(ge.y) * sp,
        o.z + static_cast<float>(ge.z) * sp,
    };
  }

  /** Global z of this slab's first owned cell (see @c DenseDisg::z_offset_global ). */
  [[nodiscard]] CARBON_HD auto
  owned_z_begin() const noexcept -> int64_t {
    return span.z_offset() + span.halo_z();
  }

  [[nodiscard]] CARBON_HD auto
  owned_z_lo() const noexcept -> float {
    return span.origin().z + static_cast<float>(owned_z_begin()) * span.spacing();
  }

  [[nodiscard]] CARBON_HD auto
  owned_z_hi() const noexcept -> float {
    return span.origin().z + static_cast<float>(addressable_z_hi()) * span.spacing();
  }

  /** Global-z band this slab can address. Owned only — the disg @c Standard view has no halo. */
  [[nodiscard]] CARBON_HD auto
  addressable_z_lo() const noexcept -> int64_t {
    return owned_z_begin();
  }

  [[nodiscard]] CARBON_HD auto
  addressable_z_hi() const noexcept -> int64_t {
    return owned_z_begin() + span.iter_extent(::carbon::DataView::Standard).z;
  }

  [[nodiscard]] CARBON_HD auto
  sample(float32_3d world) const -> FieldSample {
    const float sp = span.spacing();
    const float32_3d origin = span.origin();
    const int64_3d ge = span.global_extent();

    const float fx = (world.x - origin.x) / sp;
    const float fy = (world.y - origin.y) / sp;
    const float fz = (world.z - origin.z) / sp;

    const int ix0 = static_cast<int>(std::floor(fx));
    const int iy0 = static_cast<int>(std::floor(fy));
    const int iz0 = static_cast<int>(std::floor(fz));

    const int64_t z_lo = addressable_z_lo();
    const int64_t z_hi = addressable_z_hi();

    return trilinear_detail::sample_clamped(
        ix0, iy0, iz0, fx - static_cast<float>(ix0), fy - static_cast<float>(iy0),
        fz - static_cast<float>(iz0), ge.x, ge.y, z_lo, z_hi, [&](int gx, int gy, int gz) -> float {
          const auto vi =
              span.set_and_validate(::carbon::DataView::Standard, static_cast<int64_t>(gx),
                                    static_cast<int64_t>(gy), static_cast<int64_t>(gz) - z_lo);
          return vi.valid ? static_cast<float>(value_fn(span, vi.idx)) : 0.0F;
        });
  }

  [[nodiscard]] CARBON_HD auto
  sample_nearest(float32_3d world) const -> FieldSample {
    const float sp = span.spacing();
    const float32_3d origin = span.origin();

    const int64_t gx = static_cast<int64_t>(std::lround((world.x - origin.x) / sp));
    const int64_t gy = static_cast<int64_t>(std::lround((world.y - origin.y) / sp));
    const int64_t gz = static_cast<int64_t>(std::lround((world.z - origin.z) / sp));

    const auto vi =
        span.set_and_validate(::carbon::DataView::Standard, gx, gy, gz - owned_z_begin());
    if (!vi.valid) {
      return {};
    }
    return {static_cast<float>(value_fn(span, vi.idx)), true};
  }

  [[nodiscard]] CARBON_HD auto
  gradient(float32_3d world) const -> FieldGradientSample {
    return central_difference_gradient(*this, world, 0.5F * span.spacing());
  }
};

/**
 * Sampler over one block-sparse slab (@c carbon::Block ), driven by the brick-index volume
 * from @c detail/block/locator.h .
 *
 * Cell-centred, matching @c Locator::voxel_centre : voxel @c v sits at
 * @c world_origin + (v + 0.5) · spacing . (This differs from the Cartesian samplers' node-
 * centred convention because the locator's world frame is defined by the domain corner, not
 * by cell 0's centre.)
 *
 * @par Missing corners
 *
 * Holes are the normal state of a sparse grid, so a corner that lands in an unretained
 * brick, an inactive voxel, or another GPU's slab is replaced by the value at the cell
 * containing the query point rather than by zero — zeros would paint a dark rind around
 * every sparse boundary. The sample is rejected only when that centre cell is itself
 * inactive, which is the honest "there is no data here" answer.
 *
 * @tparam SBlock  Brick policy.
 * @tparam ValueFn @c (const Span&, const Idx&) -> convertible-to-float .
 */
template <typename SBlock, typename ValueFn>
struct BlockSampler {
  using SpanT = ::carbon::detail::block::Span<SBlock>;
  using LocatorT = ::carbon::detail::block::Locator<SBlock>;
  using Idx = ::carbon::detail::block::Idx;

  LocatorT locator{};
  SpanT span{};
  ValueFn value_fn{};

  [[nodiscard]] CARBON_HD auto
  spacing() const noexcept -> float {
    return locator.world_spacing;
  }

  [[nodiscard]] CARBON_HD auto
  world_bounds_min() const noexcept -> float32_3d {
    return locator.world_bounds_min();
  }

  [[nodiscard]] CARBON_HD auto
  world_bounds_max() const noexcept -> float32_3d {
    return locator.world_bounds_max();
  }

  [[nodiscard]] CARBON_HD auto
  owned_z_lo() const noexcept -> float {
    return locator.owned_z_lo;
  }

  [[nodiscard]] CARBON_HD auto
  owned_z_hi() const noexcept -> float {
    return locator.owned_z_hi;
  }

  [[nodiscard]] CARBON_HD auto
  sample_nearest(float32_3d world) const -> FieldSample {
    const auto hit = locator.locate_world(span, world);
    if (!hit.valid) {
      return {};
    }
    return {static_cast<float>(value_fn(span, hit.idx)), true};
  }

  [[nodiscard]] CARBON_HD auto
  sample(float32_3d world) const -> FieldSample {
    const auto centre = locator.locate_world(span, world);
    if (!centre.valid) {
      return {};
    }
    const float fallback = static_cast<float>(value_fn(span, centre.idx));

    const float32_3d v = locator.to_voxel_space(world);
    const float fx = v.x - 0.5F;
    const float fy = v.y - 0.5F;
    const float fz = v.z - 0.5F;
    const int64_t ix0 = static_cast<int64_t>(std::floor(fx));
    const int64_t iy0 = static_cast<int64_t>(std::floor(fy));
    const int64_t iz0 = static_cast<int64_t>(std::floor(fz));

    float c[8];
    for (int k = 0; k < 8; ++k) {
      const int64_3d voxel{
          ix0 + (k & 1),
          iy0 + ((k >> 1) & 1),
          iz0 + ((k >> 2) & 1),
      };
      const auto hit = locator.locate(span, voxel);
      c[k] = hit.valid ? static_cast<float>(value_fn(span, hit.idx)) : fallback;
    }

    return {
        trilinear_detail::blend(c, fx - static_cast<float>(ix0), fy - static_cast<float>(iy0),
                                fz - static_cast<float>(iz0)),
        true,
    };
  }

  [[nodiscard]] CARBON_HD auto
  gradient(float32_3d world) const -> FieldGradientSample {
    return central_difference_gradient(*this, world, 0.5F * locator.world_spacing);
  }
};

/**
 * Sampler over a multi-resolution tower (@c carbon::MultiResBlock ), driven by the per-level
 * brick-index tower from @c detail/mgrid/locator.h .
 *
 * Resolves the query point to the finest level that holds it (@c MultiResLocator::find_finest )
 * and then samples that level exactly as @ref BlockSampler does. So a point in a refined
 * region reads fine data, a point in the coarse far field reads coarse data, and neither
 * caller nor kernel has to know where the refinement boundary is.
 *
 * The level is chosen per @b query, not per ray, which means a sample straddling a
 * refinement boundary blends corners from one level only (the finest one covering its
 * centre) and falls back to the centre value for corners the level does not hold. Blending
 * @b across levels would need a resampling rule this pass does not attempt.
 *
 * @tparam SBlock  Brick policy shared by every level.
 * @tparam ValueFn @c (int level, const Span&, const Idx&) -> convertible-to-float .
 */
template <typename SBlock, typename ValueFn>
struct MultiResSampler {
  using TowerT = ::carbon::detail::mgrid::MultiResLocator<SBlock>;
  using SpanT = ::carbon::detail::block::Span<SBlock>;
  using Idx = ::carbon::detail::block::Idx;

  TowerT tower{};
  ValueFn value_fn{};

  /** Finest level's spacing; the level actually sampled may be coarser. */
  [[nodiscard]] CARBON_HD auto
  spacing() const noexcept -> float {
    return tower.depth > 0 ? tower.level_locator[0].world_spacing : 1.0F;
  }

  [[nodiscard]] CARBON_HD auto
  world_bounds_min() const noexcept -> float32_3d {
    return tower.depth > 0 ? tower.level_locator[0].world_bounds_min() : float32_3d{};
  }

  [[nodiscard]] CARBON_HD auto
  world_bounds_max() const noexcept -> float32_3d {
    return tower.depth > 0 ? tower.level_locator[0].world_bounds_max() : float32_3d{};
  }

  [[nodiscard]] CARBON_HD auto
  owned_z_lo() const noexcept -> float {
    return tower.owned_z_lo();
  }

  [[nodiscard]] CARBON_HD auto
  owned_z_hi() const noexcept -> float {
    return tower.owned_z_hi();
  }

  [[nodiscard]] CARBON_HD auto
  sample_nearest(float32_3d world) const -> FieldSample {
    int level = -1;
    Idx idx{};
    if (!tower.find_finest(world, level, idx)) {
      return {};
    }
    return {static_cast<float>(value_fn(level, tower.level_span[level], idx)), true};
  }

  [[nodiscard]] CARBON_HD auto
  sample(float32_3d world) const -> FieldSample {
    int level = -1;
    Idx centre{};
    if (!tower.find_finest(world, level, centre)) {
      return {};
    }
    const auto& loc = tower.level_locator[level];
    const SpanT& span = tower.level_span[level];
    const float fallback = static_cast<float>(value_fn(level, span, centre));

    const float32_3d v = loc.to_voxel_space(world);
    const float fx = v.x - 0.5F;
    const float fy = v.y - 0.5F;
    const float fz = v.z - 0.5F;
    const int64_t ix0 = static_cast<int64_t>(std::floor(fx));
    const int64_t iy0 = static_cast<int64_t>(std::floor(fy));
    const int64_t iz0 = static_cast<int64_t>(std::floor(fz));

    float c[8];
    for (int k = 0; k < 8; ++k) {
      const int64_3d voxel{
          ix0 + (k & 1),
          iy0 + ((k >> 1) & 1),
          iz0 + ((k >> 2) & 1),
      };
      const auto hit = loc.locate(span, voxel);
      c[k] = hit.valid ? static_cast<float>(value_fn(level, span, hit.idx)) : fallback;
    }

    return {
        trilinear_detail::blend(c, fx - static_cast<float>(ix0), fy - static_cast<float>(iy0),
                                fz - static_cast<float>(iz0)),
        true,
    };
  }

  [[nodiscard]] CARBON_HD auto
  gradient(float32_3d world) const -> FieldGradientSample {
    return central_difference_gradient(*this, world, 0.5F * spacing());
  }
};

/** Build a sampler over an aggregated dense slab. */
template <typename ValueFn>
[[nodiscard]] CARBON_HD auto
make_sampler(const ::carbon::detail::dense::Span& span,
             ValueFn value_fn) -> DenseSampler<::carbon::detail::dense::Span,
                                               ValueFn> {
  return DenseSampler<::carbon::detail::dense::Span, ValueFn>{span, value_fn};
}

/** Build a sampler over a disaggregated dense slab. */
template <typename ValueFn>
[[nodiscard]] CARBON_HD auto
make_sampler(const ::carbon::detail::dense_disg::Span& span,
             ValueFn value_fn) -> DenseDisgSampler<::carbon::detail::dense_disg::Span,
                                                   ValueFn> {
  return DenseDisgSampler<::carbon::detail::dense_disg::Span, ValueFn>{span, value_fn};
}

/**
 * Build a sampler over a block-sparse slab. The @c Locator carries the world frame, so the
 * caller picks it at @c Block::locator time rather than here.
 */
template <typename SBlock,
          typename ValueFn>
[[nodiscard]] CARBON_HD auto
make_sampler(const ::carbon::detail::block::Locator<SBlock>& locator,
             const ::carbon::detail::block::Span<SBlock>& span,
             ValueFn value_fn) -> BlockSampler<SBlock,
                                               ValueFn> {
  return BlockSampler<SBlock, ValueFn>{locator, span, value_fn};
}

/** Build a sampler over a multi-resolution tower. */
template <typename SBlock,
          typename ValueFn>
[[nodiscard]] CARBON_HD auto
make_sampler(const ::carbon::detail::mgrid::MultiResLocator<SBlock>& tower,
             ValueFn value_fn) -> MultiResSampler<SBlock,
                                                  ValueFn> {
  return MultiResSampler<SBlock, ValueFn>{tower, value_fn};
}

}  // namespace carbon::render
