#pragma once

#include <carbon/cuda_annotation.h>
#include <carbon/data_view.h>

#include <cstdint>

namespace carbon::detail::sub_grid {

/**
 * Per-GPU lane table of a @ref carbon::SubGrid : which parent cells the SubGrid selected, in
 * packed order, plus the counts that split them by @ref DataView .
 *
 * Lanes are ordered internal cells first, then boundary cells:
 *
 * @code
 *   lanes:  [ 0 .. internal_count() )  [ internal_count() .. count() )
 *             parent Internal cells       parent Boundary cells
 * @endcode
 *
 * so every view is one contiguous lane range (@ref lane_offset , @ref lane_count ) and
 * @c Internal + @c Boundary covers @c Standard exactly once — the same guarantee the parent
 * grid gives through @c iter_extent . Within each range lanes follow the parent's own
 * iteration order for that view.
 *
 * A plain view: @c table points to device memory on a device span and to host memory on a
 * host span; the owning @ref carbon::SubGrid keeps both alive.
 */
template <typename ParentIdx>
class Lanes {
public:
  constexpr Lanes() noexcept = default;

  constexpr Lanes(const ParentIdx* table,
                  int64_t count,
                  int64_t internal_count,
                  std::uint64_t owner) noexcept
      : table_(table),
        count_(count),
        internal_count_(internal_count),
        owner_(owner) {}

  /** Number of selected cells on this GPU (the S field length per component). */
  [[nodiscard]] CARBON_HD constexpr auto
  count() const noexcept -> int64_t {
    return count_;
  }

  /** Selected cells that lie in the parent's Internal region. */
  [[nodiscard]] CARBON_HD constexpr auto
  internal_count() const noexcept -> int64_t {
    return internal_count_;
  }

  /** Selected cells that lie in the parent's Boundary region. */
  [[nodiscard]] CARBON_HD constexpr auto
  boundary_count() const noexcept -> int64_t {
    return count_ - internal_count_;
  }

  /** First lane of @p view . */
  [[nodiscard]] CARBON_HD constexpr auto
  lane_offset(::carbon::DataView view) const noexcept -> int64_t {
    return view == ::carbon::DataView::Boundary ? internal_count_ : 0;
  }

  /** Number of lanes in @p view . */
  [[nodiscard]] CARBON_HD constexpr auto
  lane_count(::carbon::DataView view) const noexcept -> int64_t {
    switch (view) {
      case ::carbon::DataView::Internal:
        return internal_count_;
      case ::carbon::DataView::Boundary:
        return count_ - internal_count_;
      case ::carbon::DataView::Standard:
        break;
    }
    return count_;
  }

  /** Parent cell of @p lane . No bounds checking. */
  [[nodiscard]] CARBON_HD constexpr auto
  parent_idx(int64_t lane) const noexcept -> const ParentIdx& {
    return table_[lane];
  }

  /** Raw lane table (@ref count entries). */
  [[nodiscard]] CARBON_HD constexpr auto
  table() const noexcept -> const ParentIdx* {
    return table_;
  }

  /** Identity of the owning SubGrid; SubGrid partitions check it in debug builds. */
  [[nodiscard]] CARBON_HD constexpr auto
  owner() const noexcept -> std::uint64_t {
    return owner_;
  }

private:
  const ParentIdx* table_{};
  int64_t count_{};
  int64_t internal_count_{};
  std::uint64_t owner_{};
};

/**
 * Per-GPU span of a @ref carbon::SubGrid .
 *
 * Derives from the parent grid's @c Span and its base subobject @b is the parent span of the
 * same GPU, so an S kernel passes it straight to parent-field partitions
 * (@c uG.read(s, i, 0) ) and inherited queries such as @c global_location , @c origin and
 * @c spacing describe the parent cell. SubGrid-specific data lives behind @ref sub .
 *
 * The inherited 3D iteration members (@c iter_extent , @c set_and_validate , ...) still
 * describe the parent slab; a SubGrid launch is 1D over the lanes of @ref sub .
 *
 * @par Name rule
 * A member of a derived class hides every base member with the same name, which would give one
 * object two answers to the same query depending on the static type at the call site. This
 * class therefore adds exactly one name, @ref sub , and checks that the parent span does not
 * already use it.
 */
template <typename ParentSpan, typename ParentIdx>
class Span : public ParentSpan {
  static_assert(
      !requires(const ParentSpan& p) { p.sub(); },
      "carbon::SubGrid: the parent grid's Span already has a sub() member; "
      "SubGrid's Span would hide it.");

  Lanes<ParentIdx> sub_{};

public:
  constexpr Span() noexcept = default;

  constexpr Span(const ParentSpan& parent,
                 const Lanes<ParentIdx>& lanes) noexcept
      : ParentSpan(parent),
        sub_(lanes) {}

  /** SubGrid lane table and counts for this GPU. */
  [[nodiscard]] CARBON_HD constexpr auto
  sub() const noexcept -> const Lanes<ParentIdx>& {
    return sub_;
  }
};

}  // namespace carbon::detail::sub_grid
