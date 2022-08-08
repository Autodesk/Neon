#pragma once

#include "Neon/set/DevSet.h"
#include "Neon/set/dependencyTools/DataParsing.h"
#include "functional"
#include "type_traits"

/**
 * Abstract interface to hide
 */

namespace Neon::set::internal {

enum struct ContainerPatternType
{
    map = 0,
    stencil = 1,
    reduction = 2,
};


struct ContainerPatternTypeUtils
{
    static constexpr int nOptions = 4;

    static auto toString(ContainerPatternType option) -> std::string;
    static auto fromString(const std::string& option) -> ContainerPatternType;
    static auto getOptions() -> std::array<ContainerPatternType, nOptions>;
    static auto isExpandable(ContainerPatternType option) -> bool;
};


}  // namespace Neon::set::internal
/**
 * operator<<
 */
std::ostream& operator<<(std::ostream& os, Neon::set::internal::ContainerPatternType const& m);