#pragma once

#include <carbon/boundary_condition_distributed.h>
#include <carbon/detail/dense_distributed/boundary_view.h>
#include <carbon/detail/dense_distributed/span.h>
#include <carbon/integer3d.h>
#include <carbon/process_grid.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace carbon::detail::dense_distributed {

/**
 * Range of one face stored on one GPU. Empty (@c count == 0) when the GPU does not touch the face,
 * owns no cells, or the face's axis is periodic.
 */
struct FaceRange {
  int64_t lo[2]{0, 0};
  int64_t hi[2]{0, 0};

  [[nodiscard]] auto
  cells() const noexcept -> int64_t {
    return (hi[0] > lo[0] && hi[1] > lo[1]) ? (hi[0] - lo[0]) * (hi[1] - lo[1]) : 0;
  }
};

/**
 * Which part of global face @p face the chunk described by @p span stores. A pure function of the
 * span geometry and the stencil radius @p radius3, so the device and host paths share it.
 *
 * The chunk touches the low face of an axis iff its owned range starts at 0, and the high face iff
 * it ends at the global extent (owned ranges, never haloed ones: a halo reaches outside the domain).
 * The stored tangential range is the owned range widened by @p radius3 on each tangential axis and
 * clipped to the domain, so a diagonal read across the face finds the cell under its target. On a
 * periodic tangential axis the whole axis is stored, because reads there wrap. Faces on a periodic
 * axis are never stored (a condition on one is rejected at construction).
 */
[[nodiscard]] inline auto
face_range(const Span& span,
           ::carbon::Face face,
           const int32_3d& radius3) -> FaceRange {
  const int64_3d halo = span.halo();
  const int64_3d off = span.offset();
  const int64_3d own = span.logic_extent();
  const int64_3d ge = span.global_extent();
  const int64_t own_lo[3] = {off.x + halo.x, off.y + halo.y, off.z + halo.z};
  const int64_t own_n[3] = {own.x, own.y, own.z};
  const int64_t ext[3] = {ge.x, ge.y, ge.z};
  const int64_t rad[3] = {radius3.x, radius3.y, radius3.z};
  const Axis axes[3] = {Axis::X, Axis::Y, Axis::Z};
  const BoundaryModeDistributed mode = span.boundary_mode();

  FaceRange r;
  if (own_n[0] <= 0 || own_n[1] <= 0 || own_n[2] <= 0) {
    return r;
  }
  const int a = face_axis(face);
  if (is_periodic(mode, axes[a])) {
    return r;
  }
  const int64_t own_hi_a = own_lo[a] + own_n[a];
  const bool touches = face_is_high(face) ? (own_hi_a == ext[a]) : (own_lo[a] == 0);
  if (!touches) {
    return r;
  }
  for (int w = 0; w < 2; ++w) {
    const int t = tangential_axis(a, w);
    if (is_periodic(mode, axes[t])) {
      r.lo[w] = 0;
      r.hi[w] = ext[t];
    } else {
      r.lo[w] = std::max<int64_t>(0, own_lo[t] - rad[t]);
      r.hi[w] = std::min<int64_t>(ext[t], own_lo[t] + own_n[t] + rad[t]);
    }
  }
  return r;
}

/**
 * Host-side storage of the face data of one GPU: the condition of every stored face cell, per
 * component. Plain @c std::vector storage, so it can be built and tested without a device; the
 * device path copies these arrays and builds a @ref BoundaryView over the device pointers.
 */
class HostBoundaryStore {
public:
  HostBoundaryStore() = default;

  HostBoundaryStore(const Span& span,
                    const int32_3d& radius3,
                    int components)
      : components_(components),
        radius3_(radius3) {
    for (int f = 0; f < ::carbon::kFaceCount; ++f) {
      ranges_[f] = face_range(span, static_cast<::carbon::Face>(f), radius3);
      const size_t n = static_cast<size_t>(ranges_[f].cells()) * static_cast<size_t>(components);
      kind_[f].assign(n, static_cast<uint8_t>(::carbon::BcKind::None));
      value_[f].assign(n, 0.0);
    }
  }

  [[nodiscard]] auto
  range(::carbon::Face face) const noexcept -> const FaceRange& {
    return ranges_[static_cast<int>(face)];
  }

  [[nodiscard]] auto
  components() const noexcept -> int {
    return components_;
  }

  /**
   * Visit every stored cell of every face: @p fn(face, global_node, comp) returns the condition.
   * @c global_node is the global coordinate of the face cell (the face's own axis at 0 or at
   * extent - 1). @p fn must be a deterministic function of its arguments, because a cell stored on
   * several GPUs (margins overlap neighbouring chunks) is filled independently on each.
   *
   * @throws std::invalid_argument if a stored condition has a non-finite value, or sits on a face
   *         whose axis is not larger than the stencil radius (the mirror cell would be outside the
   *         domain).
   */
  template <typename Fn>
  void
  fill(const int64_3d& global_extent,
       Fn&& fn) {
    const int64_t ext[3] = {global_extent.x, global_extent.y, global_extent.z};
    for (int f = 0; f < ::carbon::kFaceCount; ++f) {
      const FaceRange& r = ranges_[f];
      if (r.cells() == 0) {
        continue;
      }
      const int a = f / 2;
      const int t0 = tangential_axis(a, 0);
      const int t1 = tangential_axis(a, 1);
      const int64_t e0 = r.hi[0] - r.lo[0];
      const int64_t e1 = r.hi[1] - r.lo[1];
      for (int c = 0; c < components_; ++c) {
        for (int64_t j = 0; j < e1; ++j) {
          for (int64_t i = 0; i < e0; ++i) {
            int64_t g[3] = {0, 0, 0};
            g[a] = (f % 2) == 1 ? ext[a] - 1 : 0;
            g[t0] = r.lo[0] + i;
            g[t1] = r.lo[1] + j;
            const ::carbon::FaceBc bc =
                fn(static_cast<::carbon::Face>(f), int64_3d{g[0], g[1], g[2]}, c);
            if (bc.kind != ::carbon::BcKind::None) {
              const int64_t rad[3] = {radius3_.x, radius3_.y, radius3_.z};
              if (!std::isfinite(bc.value)) {
                throw std::invalid_argument("boundary data: non-finite value on face " +
                                            std::to_string(f));
              }
              if (ext[a] <= rad[a]) {
                throw std::invalid_argument(
                    "boundary data: condition on face " + std::to_string(f) +
                    " needs a global extent larger than the stencil radius on that axis");
              }
            }
            const size_t idx = static_cast<size_t>((c * e1 + j) * e0 + i);
            kind_[f][idx] = static_cast<uint8_t>(bc.kind);
            value_[f][idx] = bc.value;
          }
        }
      }
    }
  }

  /** Raw access for the device upload. */
  [[nodiscard]] auto
  kind_data(::carbon::Face face) const noexcept -> const std::vector<uint8_t>& {
    return kind_[static_cast<int>(face)];
  }

  [[nodiscard]] auto
  value_data(::carbon::Face face) const noexcept -> const std::vector<double>& {
    return value_[static_cast<int>(face)];
  }

  /** View over this store's own host arrays (host kernels and tests). */
  [[nodiscard]] auto
  host_view() const noexcept -> BoundaryView {
    BoundaryView v;
    for (int f = 0; f < ::carbon::kFaceCount; ++f) {
      if (ranges_[f].cells() == 0) {
        continue;
      }
      v.faces[f] = make_face_view(ranges_[f], kind_[f].data(), value_[f].data(), components_);
    }
    return v;
  }

  /** Face view for arbitrary (for example device) pointers over this store's layout. */
  [[nodiscard]] static auto
  make_face_view(const FaceRange& r,
                 const uint8_t* kind,
                 const double* value,
                 int components) noexcept -> BoundaryFaceView {
    BoundaryFaceView fv;
    fv.kind = kind;
    fv.value = value;
    fv.lo[0] = r.lo[0];
    fv.lo[1] = r.lo[1];
    fv.hi[0] = r.hi[0];
    fv.hi[1] = r.hi[1];
    fv.components = components;
    return fv;
  }

private:
  int components_{0};
  int32_3d radius3_{};
  FaceRange ranges_[::carbon::kFaceCount]{};
  std::vector<uint8_t> kind_[::carbon::kFaceCount]{};
  std::vector<double> value_[::carbon::kFaceCount]{};
};

}  // namespace carbon::detail::dense_distributed
