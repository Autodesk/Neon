#include "Neon/set/ContainerTools/ContainerPatternType.h"

/**
 * Abstract interface to hide
 */

namespace Neon::set {

auto ContainerPatternTypeUtils::toString(ContainerPatternType option) -> std::string
{
    switch (option) {
        case ContainerPatternType::map: {
            return "compute";
        }
        case ContainerPatternType::stencil: {
            return "halo";
        }
        case ContainerPatternType::reduction: {
            return "sync";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto ContainerPatternTypeUtils::fromString(const std::string& option)
    -> ContainerPatternType
{
    auto const options = getOptions();

    for (auto a : options) {
        if (toString(a) == option) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto ContainerPatternTypeUtils::getOptions() -> std::array<ContainerPatternType, nOptions>
{
    std::array<ContainerPatternType, nOptions> opts = {ContainerPatternType::map,
                                                         ContainerPatternType::stencil,
                                                         ContainerPatternType::reduction};
    return opts;
}


}  // namespace Neon::set::internal


std::ostream& operator<<(std::ostream& os, Neon::set::ContainerPatternType const& m)
{
    return os << std::string(Neon::set::ContainerPatternTypeUtils::toString(m));
}
