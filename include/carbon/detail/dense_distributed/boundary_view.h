#pragma once

#include <carbon/boundary_condition_distributed.h>
#include <carbon/cuda_annotation.h>
#include <carbon/detail/dense_distributed/span.h>
#include <carbon/integer3d.h>
#include <carbon/process_grid.h>

#include <cstdint>
#include <type_traits>

namespace carbon::detail::dense_distributed {

/** Axis a face lies on (0 = x, 1 = y, 2 = z). */
[[nodiscard]] constexpr CARBON_HD auto
face_axis(::carbon::Face face) noexcept -> int {
  return static_cast<int>(face) / 2;
}

/** True for the face at the high end of its axis. */
[[nodiscard]] constexpr CARBON_HD auto
face_is_high(::carbon::Face face) noexcept -> bool {
  return (static_cast<int>(face) % 2) == 1;
}

/** The two axes tangential to @p axis, in increasing order. */
[[nodiscard]] constexpr CARBON_HD auto
tangential_axis(int axis,
                int which) noexcept -> int {
  return which == 0 ? (axis == 0 ? 1 : 0) : (axis == 2 ? 1 : 2);
}

/**
 * Pointer-only, trivially copyable view of one global face on one GPU: the condition (kind, value)
 * of every stored face cell, per component. The same struct serves host and device pointers.
 *
 * A face cell is addressed by its global tangential coordinates @c (t0, t1) (the two axes other
 * than the face's own, in increasing order). Only the range @c [lo, hi) is stored: the owned
 * tangential range of the chunk plus a margin of one stencil radius, clipped to the domain, so a
 * diagonal read across the face finds the cell under its target. A GPU that does not touch the face
 * has an empty view (null pointers, zero range) and @ref at reports @c BcKind::None everywhere.
 */
struct BoundaryFaceView {
  const uint8_t* kind{nullptr};
  const double* value{nullptr};
  int64_t lo[2]{0, 0};
  int64_t hi[2]{0, 0};
  int32_t components{0};

  /** Condition at global tangential cell @p t0, @p t1 for @p comp; @c None outside the stored range. */
  [[nodiscard]] constexpr CARBON_HD auto
  at(int64_t t0,
     int64_t t1,
     int comp) const noexcept -> ::carbon::FaceBc {
    if (kind == nullptr || comp < 0 || comp >= components || t0 < lo[0] || t0 >= hi[0] ||
        t1 < lo[1] || t1 >= hi[1]) {
      return {};
    }
    const int64_t e0 = hi[0] - lo[0];
    const int64_t e1 = hi[1] - lo[1];
    const int64_t i = (static_cast<int64_t>(comp) * e1 + (t1 - lo[1])) * e0 + (t0 - lo[0]);
    return ::carbon::FaceBc{static_cast<::carbon::BcKind>(kind[i]), value[i]};
  }
};

/** Result of @ref BoundaryView::resolve: like @ref NeighbourIdxBc plus the condition to apply. */
struct BoundaryResolution {
  NeighbourIdxBc where{};
  ::carbon::FaceBc bc{};
};

/**
 * The six face views of one GPU. Trivially copyable and pointer-only, so it is captured by value
 * into a device lambda next to a @c Partition.
 */
struct BoundaryView {
  BoundaryFaceView faces[::carbon::kFaceCount]{};

  /**
   * Data-driven counterpart of @ref neighbour_idx_bc: the condition is looked up in the face data at
   * the face cell under the target (its global tangential coordinates) instead of coming from
   * @c Span. Interior reads behave exactly as in @ref neighbour_idx. A read out of domain across a
   * face with no stored condition at that cell, across two or more faces, or whose mirror cell is
   * outside the domain or buffer, is @c Invalid.
   */
  [[nodiscard]] constexpr CARBON_HD auto
  resolve(const Span& span,
          const Idx& idx,
          ::carbon::int8_3d offset,
          int comp) const noexcept -> BoundaryResolution {
    const int64_3d cell = idx.cell();
    const int64_3d haloed = span.haloed_extent();
    const int64_3d global_extent = span.global_extent();
    const int64_3d off = span.offset();
    const BoundaryModeDistributed mode = span.boundary_mode();

    const int64_t b[3] = {cell.x + static_cast<int64_t>(offset.x),
                          cell.y + static_cast<int64_t>(offset.y),
                          cell.z + static_cast<int64_t>(offset.z)};
    const int64_t haloed_a[3] = {haloed.x, haloed.y, haloed.z};
    const int64_t global_a[3] = {global_extent.x, global_extent.y, global_extent.z};
    const int64_t off_a[3] = {off.x, off.y, off.z};
    const Axis axes[3] = {Axis::X, Axis::Y, Axis::Z};

    int bc_axis = -1;
    int bc_axis_count = 0;
    for (int a = 0; a < 3; ++a) {
      const int64_t g = b[a] + off_a[a];
      if ((g < 0 || g >= global_a[a]) && !is_periodic(mode, axes[a])) {
        bc_axis = a;
        ++bc_axis_count;
      }
    }

    BoundaryResolution r;
    if (bc_axis_count == 0) {
      if (b[0] < 0 || b[0] >= haloed_a[0] || b[1] < 0 || b[1] >= haloed_a[1] || b[2] < 0 ||
          b[2] >= haloed_a[2]) {
        return r;
      }
      r.where.idx = Idx(b[0], b[1], b[2]);
      r.where.kind = NeighbourKindBc::Interior;
      return r;
    }
    if (bc_axis_count > 1) {
      return r;
    }

    const int64_t g_bc = b[bc_axis] + off_a[bc_axis];
    const int64_t extent = global_a[bc_axis];
    const bool low = g_bc < 0;
    const int64_t k = low ? -g_bc : g_bc - (extent - 1);
    const int64_t mirror_global = low ? k : (extent - 1) - k;
    if (mirror_global < 0 || mirror_global >= extent) {
      return r;
    }
    int64_t m[3] = {b[0], b[1], b[2]};
    m[bc_axis] = mirror_global - off_a[bc_axis];
    if (m[0] < 0 || m[0] >= haloed_a[0] || m[1] < 0 || m[1] >= haloed_a[1] || m[2] < 0 ||
        m[2] >= haloed_a[2]) {
      return r;
    }

    const ::carbon::Face face = static_cast<::carbon::Face>(2 * bc_axis + (low ? 0 : 1));
    int64_t t[2];
    for (int w = 0; w < 2; ++w) {
      const int ta = tangential_axis(bc_axis, w);
      int64_t g = b[ta] + off_a[ta];
      if (is_periodic(mode, axes[ta])) {
        g %= global_a[ta];
        g = g < 0 ? g + global_a[ta] : g;
      }
      t[w] = g;
    }
    const ::carbon::FaceBc bc = faces[static_cast<int>(face)].at(t[0], t[1], comp);
    if (bc.kind == ::carbon::BcKind::None) {
      return r;
    }
    r.where.idx = Idx(m[0], m[1], m[2]);
    r.where.kind = NeighbourKindBc::Boundary;
    r.where.face = face;
    r.where.k = static_cast<int32_t>(k);
    r.bc = bc;
    return r;
  }

  /**
   * Pinning query for solvers. True when the cell at @p idx lies on a face whose stored condition
   * at that cell is Dirichlet for @p comp; @p value is then the Dirichlet value. A cell on several
   * faces (an edge or corner) takes the first Dirichlet face in the order XLow, XHigh, YLow, YHigh,
   * ZLow, ZHigh.
   */
  [[nodiscard]] constexpr CARBON_HD auto
  dirichlet_at(const Span& span,
               const Idx& idx,
               int comp,
               double& value) const noexcept -> bool {
    const int64_3d cell = idx.cell();
    const int64_3d off = span.offset();
    const int64_3d ge = span.global_extent();
    const int64_t g[3] = {cell.x + off.x, cell.y + off.y, cell.z + off.z};
    const int64_t ext[3] = {ge.x, ge.y, ge.z};
    for (int f = 0; f < ::carbon::kFaceCount; ++f) {
      const int a = f / 2;
      const bool high = (f % 2) == 1;
      if (g[a] != (high ? ext[a] - 1 : 0)) {
        continue;
      }
      const ::carbon::FaceBc bc =
          faces[f].at(g[tangential_axis(a, 0)], g[tangential_axis(a, 1)], comp);
      if (bc.kind == ::carbon::BcKind::Dirichlet) {
        value = bc.value;
        return true;
      }
    }
    return false;
  }
};

static_assert(std::is_trivially_copyable_v<BoundaryFaceView>,
              "BoundaryFaceView rides in a by-value kernel closure");
static_assert(std::is_trivially_copyable_v<BoundaryView>,
              "BoundaryView rides in a by-value kernel closure");

}  // namespace carbon::detail::dense_distributed
