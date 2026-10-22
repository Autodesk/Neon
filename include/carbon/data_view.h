#pragma once

#include <carbon/export.h>

#include <array>
#include <cstdint>
#include <iosfwd>
#include <string_view>

namespace carbon {

/**
 * Which region of a partition a kernel / container operates on.
 *
 * Standard covers the whole partition (interior + boundary slabs).
 * Internal covers only the inner region that does NOT depend on data from neighbour partitions.
 * Boundary covers only the thin slabs that need up-to-date halos from neighbours; used to
 * overlap compute with halo exchange.
 *
 * Mirrors Neon::DataView (Neon/libNeonCore/include/Neon/core/types/DataView.h) but follows
 * the carbon naming style (PascalCase enumerators, snake_case free functions).
 */
enum class DataView : std::uint8_t {
  Standard = 0,
  Internal = 1,
  Boundary = 2,
};

/** Number of valid DataView values. */
inline constexpr size_t data_view_count = 3;

/**
 * Lower-case canonical name for @p v: "standard" | "internal" | "boundary".
 * Returned view points to a static string literal; safe to store.
 */
[[nodiscard]] CARBON_API auto
to_string(DataView v) -> std::string_view;

/**
 * Parse a DataView from its canonical name. Case-insensitive: accepts "standard",
 * "Standard", "STANDARD", etc.
 * @throws std::out_of_range if @p s is not one of the three canonical names.
 */
[[nodiscard]] CARBON_API auto
from_string(std::string_view s) -> DataView;

/** Integer encoding: 0 = Standard, 1 = Internal, 2 = Boundary. */
[[nodiscard]] CARBON_API auto
to_int(DataView v) noexcept -> int;

/**
 * Build a DataView from an integer encoding.
 * @throws std::out_of_range if @p v is outside [0, data_view_count).
 */
[[nodiscard]] CARBON_API auto
from_int(int v) -> DataView;

/** All DataView values in enum order: {Standard, Internal, Boundary}. */
[[nodiscard]] CARBON_API auto
data_view_options() noexcept -> std::array<DataView,
                                           data_view_count>;

/** Stream operator: writes the lower-case canonical name (same as to_string). */
CARBON_API auto
operator<<(std::ostream& os,
           DataView v) -> std::ostream&;

}  // namespace carbon
