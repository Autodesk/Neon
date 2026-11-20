#pragma once

/** Forward declaration of @c carbon::SubGrid and its type trait, for headers that dispatch on it. */

namespace carbon {

template <typename G>
class SubGrid;

namespace detail::sub_grid {

template <typename T>
inline constexpr bool is_sub_grid_v = false;

template <typename G>
inline constexpr bool is_sub_grid_v<::carbon::SubGrid<G>> = true;

}  // namespace detail::sub_grid
}  // namespace carbon
