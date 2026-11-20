#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/detail/sub_grid/idx.h>
#include <carbon/detail/sub_grid/span.h>

#include <cassert>
#include <cstdint>

namespace carbon::detail::sub_grid {

/**
 * Per-GPU view of a @ref carbon::SubGrid field: a raw pointer to packed SoA storage of
 * @c Components planes, each @c span.sub().count() elements long, indexed by lane.
 *
 * Same-cell access only. There is deliberately no @c neighbour , @c read_pull or cursor:
 * SubGrid fields have no halo, and neighbouring lanes are not neighbouring cells. Stencils
 * belong on parent fields (an S kernel may stencil-read those through the parent's halo).
 * For the same reason this class must not derive from the parent grid's @c Partition .
 *
 * Takes the SubGrid @c Span and @c Idx — not the parent's — so a parent-grid kernel, which
 * has no lane, cannot address a SubGrid field. Does not own the storage.
 */
template <typename ParentSpan, typename ParentIdx, typename T, int Components>
class Partition {
  static_assert(Components >= 1,
                "carbon::SubGrid::Partition<T, Components> requires Components >= 1.");

public:
  using Span = sub_grid::Span<ParentSpan, ParentIdx>;
  using Idx = sub_grid::Idx<ParentIdx>;

  /** Compile-time component count (same as @p Components ). */
  static constexpr int static_value = Components;

  constexpr Partition() noexcept = default;

  /**
   * @param ptr   Packed storage for this GPU: @c Components * count elements.
   * @param owner Identity of the SubGrid the storage is laid out for.
   */
  constexpr Partition(T* ptr,
                      std::uint64_t owner) noexcept
      : ptr_(ptr),
        owner_(owner) {}

  [[nodiscard]] CARBON_HD constexpr auto
  ptr() const noexcept -> T* {
    return ptr_;
  }

  /** Identity of the SubGrid this storage is laid out for. */
  [[nodiscard]] CARBON_HD constexpr auto
  owner() const noexcept -> std::uint64_t {
    return owner_;
  }

  /** Value of component @p component_id at the cell of @p idx . No bounds checking. */
  [[nodiscard]] CARBON_HD constexpr auto
  read(const Span& span,
       const Idx& idx,
       int component_id) const noexcept -> const T& {
    return ptr_[offset_(span, idx, component_id)];
  }

  /** Stores @p value as component @p component_id of the cell of @p idx . */
  CARBON_HD constexpr auto
  write(const Span& span,
        const Idx& idx,
        int component_id,
        T value) const noexcept -> void {
    ptr_[offset_(span, idx, component_id)] = value;
  }

private:
  [[nodiscard]] CARBON_HD constexpr auto
  offset_(const Span& span,
          const Idx& idx,
          int component_id) const noexcept -> int64_t {
    const int64_t lane = idx.sub_lane();
#ifndef NDEBUG
    assert(span.sub().owner() == owner_ && "SubGrid field used with a different SubGrid");
    assert(lane >= 0 && lane < span.sub().count());
    assert(component_id >= 0 && component_id < Components);
    if constexpr (requires { idx.cell(); }) {
      assert(span.sub().parent_idx(lane).cell() == idx.cell() &&
             "SubGrid Idx lane does not match its parent cell");
    }
#endif
    if constexpr (Components == 1) {
      (void)component_id;
      return lane;
    }
    return static_cast<int64_t>(component_id) * span.sub().count() + lane;
  }

  T* ptr_{};
  std::uint64_t owner_{};
};

}  // namespace carbon::detail::sub_grid
