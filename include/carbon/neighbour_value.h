#pragma once

#include <carbon/cuda_annotation.h>

namespace carbon {

/**
 * Result of a neighbour-read API (e.g. @ref carbon::detail::dense::Partition::neighbour).
 *
 * @ref value carries the neighbour's data when @ref valid is @c true. When @ref valid is
 * @c false the neighbour is outside the buffer (no halo on this axis, or beyond the halo
 * thickness on z) and @ref value is left default-constructed; callers should treat it as
 * out-of-domain. The @ref value_or helper is the idiomatic single-line accessor when
 * boundary conditions can be expressed as a default value:
 *
 * @code
 *   const T east = part.neighbour(s, idx, {1, 0, 0}, 0).value_or(T{0});
 * @endcode
 *
 * Mirrors Neon's @c NghData<T> with carbon's snake_case field names.
 *
 * @tparam T Element type read out of the underlying partition.
 */
template <typename T>
struct NeighbourValue {
  T value{};
  bool valid{false};

  /** Returns @ref value when @ref valid is @c true, otherwise @p alternative. */
  [[nodiscard]] CARBON_HD constexpr auto
  value_or(T alternative) const noexcept -> T {
    return valid ? value : alternative;
  }
};

}  // namespace carbon
