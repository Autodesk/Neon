#pragma once

#include <concepts>
#include <cstdint>

namespace carbon {

/** Types that expose a stable numeric id via get_uid() returning std::uint64_t. */
template <typename U>
concept DataUid = requires(const U& u) {
  {u.get_uid()}->std::same_as<std::uint64_t>;
};

}  // namespace carbon
