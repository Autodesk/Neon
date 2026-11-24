#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense_disg_distributed/idx.h>
#include <carbon/detail/dense_disg_distributed/span.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>
#include <carbon/neighbour_value.h>

#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CARBON_DDD_NOINLINE __noinline__
#else
#define CARBON_DDD_NOINLINE
#endif

namespace carbon::detail::dense_disg_distributed {

/**
 * Pencil/brick seam path of @c Partition::read_pull . Kept out of line so a slab kernel that
 * pulls many offsets does not pay the nested classification's register pressure.
 */
[[nodiscard]] CARBON_HD CARBON_DDD_NOINLINE constexpr auto
pull_offset_nonslab(const Span& span,
                    const Idx& idx,
                    ::carbon::int8_3d offset,
                    int components,
                    int component_id) noexcept -> int64_t {
  return span.linear_offset(pull_idx(span, idx, offset), components, component_id);
}

template <typename T, int Components>
class Cursor;

/**
 * Lightweight view of one slab of a @ref DenseDisgDistributed field. Twin of
 * @c dense_disg::Partition : the offset math goes through the disaggregated pitch on
 * @ref Span::linear_offset , which consumes the per-region fields populated by
 * @ref Span::set_and_validate .
 *
 * Lifetime: does not own @c ptr ; the pointed-to storage must outlive the partition.
 *
 * @tparam T          Element type referenced by @ref ptr .
 * @tparam Components Per-cell component count. Must match the @c components argument the
 *                    @ref Span uses in its pitch math.
 */
template <typename T, int Components>
class Partition {
  static_assert(Components >= 1,
                "carbon::dense_disg_distributed::Partition<T, Components> requires "
                "Components >= 1 (1 = scalar, 3 = vector, ...).");

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

  /** Read component @p component_id of cell @p idx through the disaggregated pitch. */
  template <typename SpanT>
  [[nodiscard]] CARBON_HD constexpr auto
  read(const SpanT& span,
       const Idx& idx,
       int component_id) const noexcept -> const T& {
    return ptr_[linear_offset_(span, idx, component_id)];
  }

  /** Store @p value at @p idx for component @p component_id . Same pitch math as @ref read . */
  template <typename SpanT>
  CARBON_HD constexpr auto
  write(const SpanT& span,
        const Idx& idx,
        int component_id,
        T value) const noexcept -> void {
    ptr_[linear_offset_(span, idx, component_id)] = value;
  }

  /**
   * Read the cell at @p idx + @p offset for component @p component_id , returning
   * @ref NeighbourValue<T> . Valid iff @ref neighbour_idx accepts the neighbour (in the haloed
   * buffer and inside the global grid, or wrapped by a periodic axis).
   */
  template <typename SpanT>
  [[nodiscard]] CARBON_HD constexpr auto
  neighbour(const SpanT& span,
            const Idx& idx,
            int8_3d offset,
            int component_id) const noexcept -> NeighbourValue<T> {
    const NeighbourIdx r = neighbour_idx(span, idx, offset);
    if (!r.valid) {
      return {};
    }
    return NeighbourValue<T>{ptr_[linear_offset_(span, r.idx, component_id)], true};
  }

  /**
   * @b Trusted neighbour read with no validity branch; grid-generic twin of
   * @c dense_disg::Partition::read_pull . The caller guarantees the neighbour is in-buffer.
   *
   * Keeps the same-region fast paths, where the neighbour is a constant delta from the centre's
   * offset:
   *
   * - Slab storage: any @c Δz == 0 offset, or a z-offset whose centre and neighbour both sit in
   *   the internal region.
   * - Pencil storage: centre and neighbour both in @c Internal (the internal-internal path every
   *   interior cell takes), or a @c Δz == 0 offset from a z-boundary plane that stays on owned
   *   rows.
   * - Brick storage: internal-internal (pitch @c Span::internal_pitch ), then the in-face paths
   *   that stay inside one canonical boundary group: a @c Δz == 0 offset inside a z-boundary
   *   plane, a @c Δy == 0 offset inside the canonical rows of a y-boundary group, a @c Δx == 0
   *   offset inside the canonical block of an x-boundary group.
   *
   * Every other offset (crossing a seam between regions) pays the full @ref pull_idx
   * reclassification.
   */
  template <typename SpanT>
  [[nodiscard]] CARBON_HD constexpr auto
  read_pull(const SpanT& span,
            const Idx& idx,
            int8_3d offset,
            int component_id) const noexcept -> const T& {
    // Compile-time slab path (TypedSpan<StorageKind::Slab>): bit-identical to
    // dense_disg::Partition::read_pull — no ghost_y / ghost_x branches in the IR.
    if constexpr (requires { SpanT::kind; }) {
      if constexpr (SpanT::kind == StorageKind::Slab) {
        return read_pull_slab_(span, idx, offset, component_id);
      } else {
        return read_pull_nonslab_(span, idx, offset, component_id);
      }
    } else {
      // Untyped Span fallback (host helpers / older call sites).
      if (span.is_slab()) {
        return read_pull_slab_(span, idx, offset, component_id);
      }
      return read_pull_nonslab_(span, idx, offset, component_id);
    }
  }

  template <typename SpanT>
  [[nodiscard]] CARBON_HD constexpr auto
  cursor(const SpanT& span,
         const Idx& idx) const noexcept -> Cursor<T,
                                                  Components> {
    return Cursor<T, Components>{*this, Span{span}, idx};
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
  template <typename SpanT>
  [[nodiscard]] CARBON_HD static constexpr auto
  linear_offset_(const SpanT& span,
                 const Idx& idx,
                 int component_id) noexcept -> int64_t {
    if constexpr (requires { SpanT::kind; }) {
      if constexpr (SpanT::kind == StorageKind::Slab) {
        return idx.offset_local_no_card() +
               span.xy_slab_elements() *
                   (static_cast<int64_t>(idx.region_first_z()) * Components +
                    static_cast<int64_t>(component_id) * static_cast<int64_t>(idx.region_z_dim()));
      }
    }
    return span.linear_offset(idx, Components, component_id);
  }

  template <typename SpanT>
  [[nodiscard]] CARBON_HD constexpr auto
  read_pull_slab_(const SpanT& span,
                  const Idx& idx,
                  int8_3d offset,
                  int component_id) const noexcept -> const T& {
    const int64_t base = linear_offset_(span, idx, component_id);
    const int64_t xy_delta =
        static_cast<int64_t>(offset.x) + static_cast<int64_t>(offset.y) * span.haloed_extent().x;
    if (offset.z == 0) {
      return ptr_[base + xy_delta];
    }
    const int64_t bz_neighbour = idx.cell().z + static_cast<int64_t>(offset.z);
    const int64_t lz = span.haloed_extent().z - 2 * Span::ghost_radius;
    if (idx.region_z_dim() > 1 && bz_neighbour > 1 && bz_neighbour < lz) {
      return ptr_[base + xy_delta + static_cast<int64_t>(offset.z) * span.xy_slab_elements()];
    }
    return ptr_[linear_offset_(span, slab_pull_idx(span, idx, offset), component_id)];
  }

  [[nodiscard]] CARBON_HD static constexpr auto
  inner_(int64_t v,
         int64_t n) noexcept -> bool {
    return v > 1 && v < n;
  }

  [[nodiscard]] CARBON_HD static constexpr auto
  owned_(int64_t v,
         int64_t n) noexcept -> bool {
    return v >= 1 && v <= n;
  }

  template <typename SpanT>
  [[nodiscard]] CARBON_HD CARBON_DDD_NOINLINE constexpr auto
  read_pull_nonslab_(const SpanT& span,
                     const Idx& idx,
                     int8_3d offset,
                     int component_id) const noexcept -> const T& {
    const int64_t base = span.linear_offset(idx, Components, component_id);
    if (span.ghost_x() != 0) {
      return read_pull_brick_(span, idx, offset, component_id, base);
    }
    const int64_t nx = span.haloed_extent().x;
    const int64_t lz = span.haloed_extent().z - 2 * Span::ghost_radius;
    const int64_t ly = span.haloed_extent().y - 2;
    const int64_t by = idx.cell().y;
    const int64_t bz = idx.cell().z;
    const int64_t by_neighbour = by + static_cast<int64_t>(offset.y);
    const int64_t bz_neighbour = bz + static_cast<int64_t>(offset.z);
    const int64_t xy_delta = static_cast<int64_t>(offset.x) + static_cast<int64_t>(offset.y) * nx;
    if (by > 1 && by < ly && bz > 1 && bz < lz && by_neighbour > 1 && by_neighbour < ly &&
        bz_neighbour > 1 && bz_neighbour < lz) {
      return ptr_[base + static_cast<int64_t>(offset.x) +
                  nx *
                      (static_cast<int64_t>(offset.y) + static_cast<int64_t>(offset.z) * (ly - 2))];
    }
    if (offset.z == 0 && (bz == 1 || bz == lz) && by_neighbour >= 1 && by_neighbour <= ly) {
      return ptr_[base + xy_delta];
    }
    return ptr_[pull_offset_nonslab(span, idx, offset, Components, component_id)];
  }

  [[nodiscard]] CARBON_HD constexpr auto
  read_pull_brick_(const Span& span,
                   const Idx& idx,
                   int8_3d offset,
                   int component_id,
                   int64_t base) const noexcept -> const T& {
    const int64_3d n = span.logic_extent();
    const int64_3d c = idx.cell();
    const int64_t dx = static_cast<int64_t>(offset.x);
    const int64_t dy = static_cast<int64_t>(offset.y);
    const int64_t dz = static_cast<int64_t>(offset.z);
    const int64_t mx = c.x + dx;
    const int64_t my = c.y + dy;
    const int64_t mz = c.z + dz;
    if (inner_(c.x, n.x) && inner_(c.y, n.y) && inner_(c.z, n.z) && inner_(mx, n.x) &&
        inner_(my, n.y) && inner_(mz, n.z)) {
      return ptr_[base + dx + span.internal_pitch() * (dy + dz * (n.y - 2))];
    }
    if (dz == 0 && (c.z == 1 || c.z == n.z) && owned_(mx, n.x) && owned_(my, n.y)) {
      return ptr_[base + dx + dy * n.x];
    }
    if (dy == 0 && (c.y == 1 || c.y == n.y) && inner_(c.z, n.z) && inner_(mz, n.z) &&
        owned_(mx, n.x)) {
      return ptr_[base + dx + dz * n.x];
    }
    if (dx == 0 && (c.x == 1 || c.x == n.x) && inner_(c.y, n.y) && inner_(c.z, n.z) &&
        inner_(my, n.y) && inner_(mz, n.z)) {
      return ptr_[base + dy + dz * n.y];
    }
    return ptr_[pull_offset_nonslab(span, idx, offset, Components, component_id)];
  }

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

}  // namespace carbon::detail::dense_disg_distributed
