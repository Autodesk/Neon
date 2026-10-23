#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense_disg/idx.h>
#include <carbon/detail/dense_disg/span.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>
#include <carbon/neighbour_value.h>

#include <stdint.h>

namespace carbon::detail::dense_disg {

template <typename T, int Components>
class Cursor;

/**
 * Lightweight view of a @b disaggregated partitioned slab. Mirrors carbon's aggregated
 * @c carbon::detail::dense::Partition but drives its offset math through the disg pitch
 * formula on @ref Span::linear_offset (i.e. through Neon's @c dPartition::getPitch ),
 * which consumes the per-region precomputed fields populated by @ref Span::set_and_validate .
 *
 * Lifetime: does not own @c ptr ; the pointed-to storage must outlive the partition.
 *
 * @tparam T          Element type referenced by @ref ptr .
 * @tparam Components Per-cell component count (1 = scalar, 3 = vector, ...). Same value
 *                    carried by the @c Field this partition was built from. Must match
 *                    the @c components argument the @ref Span uses in its pitch math.
 */
template <typename T, int Components>
class Partition {
  static_assert(Components >= 1,
                "carbon::dense_disg::Partition<T, Components> requires Components >= 1 "
                "(1 = scalar, 3 = vector, ...).");

public:
  static constexpr int static_value = Components;

  constexpr Partition() noexcept = default;

  /**
   * @param p        Raw pointer to slab storage; may be null.
   * @param view     Region this partition targets; defaults to @c DataView::Standard .
   * @param origin   Physical-space origin (copied from the owning grid's @c Span).
   * @param spacing  Isotropic cell spacing (copied from the @c Span).
   */
  constexpr explicit Partition(T* p,
                               carbon::DataView view = carbon::DataView::Standard,
                               float32_3d origin = {},
                               float spacing = 1.0F) noexcept
      : ptr_(p),
        data_view_(view),
        origin_(origin),
        spacing_(spacing) {}

  [[nodiscard]] CARBON_HD constexpr auto
  ptr() const noexcept -> T* {
    return ptr_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  data_view() const noexcept -> carbon::DataView {
    return data_view_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  origin() const noexcept -> float32_3d {
    return origin_;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  spacing() const noexcept -> float {
    return spacing_;
  }

  /**
   * Read the element at cell @p idx for component @p component_id using the disaggregated
   * pitch on @p span . Callable from host and device. No bounds checking — caller ensures
   * the indices are valid (matches Neon's @c dPartition::operator() for performance).
   */
  [[nodiscard]] CARBON_HD constexpr auto
  read(const Span& span,
       const Idx& idx,
       int component_id) const noexcept -> const T& {
    return ptr_[span.linear_offset(idx, Components, component_id)];
  }

  /**
   * Counterpart to @ref read : stores @p value at @p idx for component @p component_id .
   * Marked @c const because the @b view (this object) is unchanged by the call; the
   * pointed-to storage is mutated through the non-const @c T* it holds. Same pitch math
   * as @ref read . No bounds checking.
   */
  CARBON_HD constexpr auto
  write(const Span& span,
        const Idx& idx,
        int component_id,
        T value) const noexcept -> void {
    ptr_[span.linear_offset(idx, Components, component_id)] = value;
  }

  /**
   * Read the cell at @p idx + @p offset for component @p component_id , returning
   * @ref NeighbourValue<T> so the caller can distinguish in-domain reads from
   * out-of-domain accesses without manual boundary checks. Twin of
   * @c dense::Partition::neighbour ; the difference is that the offset math goes
   * through the disaggregated pitch on @ref Span::linear_offset (which consumes the
   * per-region precomputed fields that @c dense_disg::neighbour_idx populates on the
   * returned @ref Idx ).
   *
   * Valid iff the neighbour lies inside the global Cartesian grid (under
   * @c BoundaryMode::Open ) or anywhere the halo exchange has already wrapped it into
   * the buffer (under @c BoundaryMode::PeriodicZ ). Same x/y boundary semantics as
   * the aggregated dense grid (storage is z-haloed only; x/y out-of-domain reads stay
   * invalid).
   */
  [[nodiscard]] CARBON_HD constexpr auto
  neighbour(const Span& span,
            const Idx& idx,
            int8_3d offset,
            int component_id) const noexcept -> NeighbourValue<T> {
    const NeighbourIdx r = neighbour_idx(span, idx, offset);
    if (!r.valid) {
      return {};
    }
    return NeighbourValue<T>{ptr_[span.linear_offset(r.idx, Components, component_id)], true};
  }

  /**
   * @b Trusted neighbour read: return the element at @p idx + @p offset for component
   * @p component_id with @b no validity branch. Grid-generic twin of
   * @c dense::Partition::read_pull — a pull-stream kernel written against @c read_pull runs
   * unchanged on both @c Dense and @c DenseDisg. The caller guarantees the neighbour is
   * in-buffer (e.g. via a precomputed wall mask). No bounds checking.
   *
   * @par Fast path (the reason this isn't just @c linear_offset(pull_idx(...)) )
   *
   * The disaggregated buffer is partitioned on @b z only, so any neighbour that lands in the
   * @b same z-region as the centre shares the centre's per-region pitch fields
   * ( @c region_first_z / @c region_z_dim ) — its element offset is then exactly the centre's
   * offset plus a @b constant delta @c (Δx + Δy·xy_pitch + Δz·slab_pitch) , with @b no
   * five-region reclassification. Two cases hit this cheap arithmetic (identical in cost to the
   * aggregated @c Dense pull):
   *
   * - @b Any offset with @c Δz == 0 (all x / y neighbours): the z-region cannot change.
   * - @b A z-offset whose centre @b and neighbour both sit in the internal region (detected via
   *   the centre's @c region_z_dim > 1 plus a range check on the shifted @c bz ). For an
   *   @c Internal -view launch this is @b every cell except the two region-seam slabs; for a
   *   @c Standard launch it is the whole internal band (≈ all owned cells bar the two
   *   boundary slabs).
   *
   * Only a z-offset that actually crosses the internal↔boundary seam falls back to the full
   * @ref pull_idx reclassification. This is what closes the single-GPU LBM gap with @c Dense:
   * the region math used to run on every one of the ~18 pull directions per cell (inflating
   * register pressure and instruction count); now the bulk pulls are branch-free constant-delta
   * loads and only the thin seam pays the reclassification.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  read_pull(const Span& span,
            const Idx& idx,
            int8_3d offset,
            int component_id) const noexcept -> const T& {
    const int64_t base = span.linear_offset(idx, Components, component_id);
    const int64_t xy_delta =
        static_cast<int64_t>(offset.x) + static_cast<int64_t>(offset.y) * span.haloed_extent().x;
    if (offset.z == 0) {
      // Same z-region as the centre by construction (regions split on z only).
      return ptr_[base + xy_delta];
    }
    const int64_t bz_neighbour = idx.cell().z + static_cast<int64_t>(offset.z);
    const int64_t lz = span.haloed_extent().z - 2 * span.halo_z();
    // Centre internal (region_z_dim == lz - 2 > 1) and neighbour still internal
    // (bz in (1, lz)) => both share the internal region's pitch, so the offset is a
    // constant slab-pitch delta away.
    if (idx.region_z_dim() > 1 && bz_neighbour > 1 && bz_neighbour < lz) {
      return ptr_[base + xy_delta + static_cast<int64_t>(offset.z) * span.xy_slab_elements()];
    }
    // Seam / boundary centre: the neighbour lands in a different region — full pitch math.
    return ptr_[span.linear_offset(pull_idx(span, idx, offset), Components, component_id)];
  }

  [[nodiscard]] CARBON_HD constexpr auto
  cursor(const Span& span,
         const Idx& idx) const noexcept -> Cursor<T,
                                                  Components> {
    return Cursor<T, Components>{*this, span, idx};
  }

  constexpr void
  reset(T* p = nullptr,
        carbon::DataView view = carbon::DataView::Standard,
        float32_3d origin = {},
        float spacing = 1.0F) noexcept {
    ptr_ = p;
    data_view_ = view;
    origin_ = origin;
    spacing_ = spacing;
  }

private:
  T* ptr_{};
  carbon::DataView data_view_{carbon::DataView::Standard};
  float32_3d origin_{};
  float spacing_{1.0F};
};

/** Forwards to @ref Partition::read_pull / read / write ; same trusted contract as dense. */
template <typename T, int Components>
class Cursor {
public:
  CARBON_HD constexpr Cursor(const Partition<T,
                                             Components>& part,
                             Span span,
                             Idx idx) noexcept
      : part_(part),
        span_(span),
        idx_(idx) {}

  [[nodiscard]] CARBON_HD constexpr auto
  value(int component_id = 0) const noexcept -> const T& {
    return part_.read(span_, idx_, component_id);
  }

  CARBON_HD constexpr auto
  write(T value,
        int component_id = 0) const noexcept -> void {
    part_.write(span_, idx_, component_id, value);
  }

  [[nodiscard]] CARBON_HD constexpr auto
  pull(int8_3d offset,
       int component_id = 0) const noexcept -> const T& {
    return part_.read_pull(span_, idx_, offset, component_id);
  }

private:
  Partition<T, Components> part_{};
  Span span_{};
  Idx idx_{};
};

}  // namespace carbon::detail::dense_disg
