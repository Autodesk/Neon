#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>
#include <carbon/detail/dense/span.h>
#include <carbon/floating3d.h>
#include <carbon/integer3d.h>
#include <carbon/neighbour_value.h>

#include <stdint.h>
#include <cassert>

namespace carbon::detail::dense {

/**
 * Trusted cell accessor: one @ref Span::linear_offset for the centre, then neighbour reads via
 * constant strides (same contract as @ref Partition::read_pull ). User code never constructs an
 * @ref Idx for neighbours.
 */
template <typename T, int Components>
class Cursor {
public:
  CARBON_HD constexpr Cursor(T* base,
                             int32_t y_stride,
                             int32_t z_stride,
                             int64_t component_stride
#ifndef NDEBUG
                             ,
                             int64_3d centre,
                             int64_3d haloed
#endif
                             ) noexcept
      : base_(base),
        y_stride_(y_stride),
        z_stride_(z_stride),
        component_stride_(component_stride)
#ifndef NDEBUG
        ,
        centre_(centre),
        haloed_(haloed)
#endif
  {
  }

  [[nodiscard]] CARBON_HD constexpr auto
  value(int component_id = 0) const noexcept -> const T& {
    return base_[component_offset_(component_id)];
  }

  CARBON_HD constexpr auto
  write(T value,
        int component_id = 0) const noexcept -> void {
    base_[component_offset_(component_id)] = value;
  }

  [[nodiscard]] CARBON_HD constexpr auto
  pull(int8_3d offset,
       int component_id = 0) const noexcept -> const T& {
#ifndef NDEBUG
    const int64_t bx = centre_.x + static_cast<int64_t>(offset.x);
    const int64_t by = centre_.y + static_cast<int64_t>(offset.y);
    const int64_t bz = centre_.z + static_cast<int64_t>(offset.z);
    assert(bx >= 0 && bx < haloed_.x && by >= 0 && by < haloed_.y && bz >= 0 && bz < haloed_.z);
#endif
    const int64_t delta = static_cast<int64_t>(offset.x) +
                          static_cast<int64_t>(offset.y) * static_cast<int64_t>(y_stride_) +
                          static_cast<int64_t>(offset.z) * static_cast<int64_t>(z_stride_);
    return base_[component_offset_(component_id) + delta];
  }

private:
  [[nodiscard]] CARBON_HD constexpr auto
  component_offset_(int component_id) const noexcept -> int64_t {
    if constexpr (Components == 1) {
      (void)component_id;
      return 0;
    }
    return static_cast<int64_t>(component_id) * component_stride_;
  }

  T* base_{};
  int32_t y_stride_{};
  int32_t z_stride_{};
  int64_t component_stride_{};
#ifndef NDEBUG
  int64_3d centre_{};
  int64_3d haloed_{};
#endif
};

/**
 * Lightweight view of a partitioned slab: a raw pointer to elements of type @p T tagged with the
 * @ref DataView region this view targets (Standard / Internal / Boundary). The non-type template
 * parameter @p Components is the compile-time number of components stored per cell (1 for a
 * scalar field, 3 for a vector field, etc.); use @c static_value to read it back.
 *
 * Global slab metadata (z-origin, halo radius, owned vs. haloed extent, layout strides) lives on
 * @ref Span and is shared by every field over the same grid/slab. Each partition also carries a
 * copy of the grid's physical @ref origin and @ref spacing (same values as on the matching
 * @c Span) so device code can read world metadata from the partition alone.
 *
 * Does not own @c ptr; lifetime of the pointed-to storage must exceed use of this object.
 *
 * @tparam T          Element or payload type referenced by @ref ptr.
 * @tparam Components Number of components per cell (compile-time). Same value carried by the
 *                    @ref Field this partition was built from.
 */
template <typename T, int Components>
class Partition {
  static_assert(Components >= 1,
                "carbon::Dense::Partition<T, Components> requires Components >= 1 "
                "(1 = scalar, 3 = vector, ...). Components matches the value carried by the "
                "Dense::Field this partition was built from.");

public:
  /** Compile-time component count from the class template parameter (same as @p Components). */
  static constexpr int static_value = Components;

  constexpr Partition() noexcept = default;

  /**
   * Binds @p ptr and the partition's @ref DataView.
   * @param p        Raw pointer to storage; may be null.
   * @param view     Region this partition targets; defaults to DataView::Standard.
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

  /** Physical-space origin at global cell @c (0,0,0) (cell-corner convention). */
  [[nodiscard]] CARBON_HD constexpr auto
  origin() const noexcept -> float32_3d {
    return origin_;
  }

  /** Isotropic cell spacing (same convention as @ref Span::spacing). */
  [[nodiscard]] CARBON_HD constexpr auto
  spacing() const noexcept -> float {
    return spacing_;
  }

  /**
   * Read the element at cell @p idx for component @p component_id using the layout
   * described by @p span (SoA dense, matching Neon's @c dPartition::operator()). Offset math
   * lives on @ref Span::linear_offset so all fields over the same slab share it.
   *
   * Callable from host and device. No bounds checking (matches Neon's operator() for
   * performance); caller is responsible for ensuring the indices are valid.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  read(const Span& span,
       const Idx& idx,
       int component_id) const noexcept -> const T& {
    return ptr_[span.linear_offset(idx, component_id)];
  }

  /**
   * Counterpart to @ref read: stores @p value at cell @p idx for component @p component_id.
   * Same SoA layout (offset math via @ref Span::linear_offset) so a kernel that reads via
   * @ref read writes back through the same coordinate system.
   *
   * Marked @c const because @c Partition is conceptually a non-owning view onto the buffer
   * (parallel to @c std::span<T>::operator[] being @c const): it is the @b view that is
   * unchanged by the call, while the pointed-to storage is mutated. Callable from host and
   * device. No bounds checking; caller ensures the indices are valid.
   */
  CARBON_HD constexpr auto
  write(const Span& span,
        const Idx& idx,
        int component_id,
        T value) const noexcept -> void {
    ptr_[span.linear_offset(idx, component_id)] = value;
  }

  /**
   * Read the cell at @p idx + @p offset for component @p component_id, returning
   * @ref NeighbourValue<T> so the caller can distinguish in-domain reads from out-of-domain
   * accesses without dropping into manual boundary checks.
   *
   * @c valid is @c true iff the neighbour lies inside the @b global Cartesian grid @b and is
   * reachable from this slab's storage. Two checks combine:
   *
   * - @b Buffer (storage) bounds: @c (cell + offset) must be inside @c span.haloed_extent(). For
   *   @b x and @b y the haloed extent equals the global extent (no x/y halos), so this is the
   *   only check needed there. For @b z, this enforces that the offset stays within
   *   @c [-halo_z, +halo_z] of the slab — a stencil that exceeds the halo radius will read
   *   @c valid = false at the slab edge.
   * - @b Global-domain bounds on @b z: even in-buffer halo cells may live outside the global
   *   grid. The upper z-halo of the top-most slab maps to global z = @c global_extent.z, and the
   *   lower z-halo of the bottom-most slab to global z = -1; both are out-of-domain in a
   *   non-toroidal grid (carbon's only mode today). Such reads return @c valid = false even
   *   though the storage exists. For interior slabs the upper halo aliases the first owned
   *   layer of the next slab — that cell @b is in the global domain, and once halos are
   *   exchanged the read returns @c valid = true.
   *
   * Same SoA layout as @ref read; offset math via @ref Span::linear_offset. Mirrors Neon's
   * @c dPartition::getNghData; the offset type is @c int8_3d (radius up to ±127, common
   * stencil sizes) which is small enough to keep stencil parameters in registers.
   *
   * The shape-and-bounds half of the work lives in @c detail::dense::neighbour_idx (internal,
   * not user-facing); this member is purely the data-access half on top of it. That split
   * keeps @c Partition grid-family agnostic — the same template will work for sparse / block
   * grids, with each family providing its own @c neighbour_idx overload.
   *
   * Callable from host and device. @c CARBON_HD + @c constexpr so the compiler can inline
   * neighbour reads inside a stencil kernel without indirection.
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
    return NeighbourValue<T>{ptr_[span.linear_offset(r.idx, component_id)], true};
  }

  /**
   * @b Trusted neighbour read: return the element at @p idx + @p offset for component
   * @p component_id with @b no validity branch. The caller guarantees the neighbour is
   * in-buffer (e.g. it precomputed a wall / active mask so the non-boundary branch can
   * never address out-of-buffer storage). Mirrors Neon's @c dPartition::pull, which
   * likewise reads through raw offset arithmetic once the kernel has proven the source
   * index is in-buffer; skipping the per-neighbour bounds check is what lets a pull-stream
   * LBM kernel hit the same bandwidth as a hand-rolled kernel.
   *
   * For the aggregated dense grid the neighbour @ref Idx is just the cell shifted by
   * @p offset (all buffer metadata lives on @ref Span), so this is a single
   * @ref Span::linear_offset. The @c dense_disg twin does the same call shape but recomputes
   * its per-region pitch fields for the shifted cell — that is exactly why this member is
   * the grid-generic pull primitive: a kernel written against @c read_pull runs unchanged on
   * either dense family. No bounds checking; the result is defined only when the neighbour
   * is in-buffer.
   */
  [[nodiscard]] CARBON_HD constexpr auto
  read_pull(const Span& span,
            const Idx& idx,
            int8_3d offset,
            int component_id) const noexcept -> const T& {
    const int64_3d cell = idx.cell();
    const Idx shifted(cell.x + static_cast<int64_t>(offset.x),
                      cell.y + static_cast<int64_t>(offset.y),
                      cell.z + static_cast<int64_t>(offset.z));
    return ptr_[span.linear_offset(shifted, component_id)];
  }

  /**
   * @brief Trusted cell cursor at @p idx : one linear_offset for the centre, then @ref Cursor::pull
   *        for neighbours. Same in-buffer contract as @ref read_pull .
   */
  [[nodiscard]] CARBON_HD constexpr auto
  cursor(const Span& span,
         const Idx& idx) const noexcept -> Cursor<T,
                                                  Components> {
#ifndef NDEBUG
    return Cursor<T, Components>{ptr_ + span.linear_offset(idx, 0),
                                 static_cast<int32_t>(span.haloed_extent().x),
                                 static_cast<int32_t>(span.xy_slab_elements()),
                                 span.linear_storage_elements(),
                                 idx.cell(),
                                 span.haloed_extent()};
#else
    return Cursor<T, Components>{
        ptr_ + span.linear_offset(idx, 0), static_cast<int32_t>(span.haloed_extent().x),
        static_cast<int32_t>(span.xy_slab_elements()), span.linear_storage_elements()};
#endif
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

}  // namespace carbon::detail::dense
