#pragma once

#include <carbon/cuda_annotation.h>

#include <cstdint>

namespace carbon::detail::sub_grid {

/**
 * Cell index of a @ref carbon::SubGrid launch.
 *
 * Derives from the parent grid's @c Idx and carries one extra value, the cell's @b lane: its
 * slot in the packed SubGrid storage. Because the base subobject is the parent cell, an S
 * kernel passes an @c Idx straight to a parent-field partition (it binds to
 * @c const ParentIdx& ), while SubGrid partitions address their packed storage through
 * @ref sub_lane . The conversion only goes one way: a plain @c ParentIdx does not convert to
 * an @c Idx , so a parent-grid kernel cannot address a SubGrid field.
 *
 * Only the launcher builds an @c Idx (from the lane table); user code receives it.
 *
 * @par Name rule
 * A member of a derived class hides every base member with the same name, so this class adds
 * exactly one name, @ref sub_lane , and checks that the parent index does not already use it.
 */
template <typename ParentIdx>
class Idx : public ParentIdx {
  static_assert(
      !requires(const ParentIdx& p) { p.sub_lane(); },
      "carbon::SubGrid: the parent grid's Idx already has a sub_lane() member; "
      "SubGrid's Idx would hide it.");

  int64_t lane_{};

public:
  constexpr Idx() noexcept = default;

  CARBON_HD constexpr Idx(const ParentIdx& parent,
                          int64_t lane) noexcept
      : ParentIdx(parent),
        lane_(lane) {}

  /** Slot of this cell in the packed SubGrid storage (same index space as the lane table). */
  [[nodiscard]] CARBON_HD constexpr auto
  sub_lane() const noexcept -> int64_t {
    return lane_;
  }
};

}  // namespace carbon::detail::sub_grid
