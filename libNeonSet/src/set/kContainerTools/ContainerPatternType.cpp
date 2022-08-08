#include "Neon/set/ContainerTools/ContainerPatternType.h"

/**
 * Abstract interface to hide
 */

namespace Neon::set::internal {

auto ContainerPatternTypeUtils::toString(ContainerPatternType option) -> std::string
{
    switch (option) {
        case ContainerPatternType::compute: {
            return "compute";
        }
        case ContainerPatternType::halo: {
            return "halo";
        }
        case ContainerPatternType::sync: {
            return "sync";
        }
        case ContainerPatternType::anchor: {
            return "anchor";
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
    std::array<ContainerPatternType, nOptions> opts = {ContainerPatternType::compute,
                                                         ContainerPatternType::halo,
                                                         ContainerPatternType::sync,
                                                         ContainerPatternType::anchor};
    return opts;
}

auto ContainerPatternTypeUtils::isExpandable(ContainerPatternType option) -> bool
{
    switch (option) {
        case ContainerPatternType::compute: {
            return false;
        }
        case ContainerPatternType::halo: {
            return false;
        }
        case ContainerPatternType::sync: {
            return false;
        }
        case ContainerPatternType::anchor: {
            return true;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

}  // namespace Neon::set::internal


std::ostream& operator<<(std::ostream& os, Neon::set::internal::ContainerPatternType const& m)
{
    return os << std::string(Neon::set::internal::ContainerPatternTypeUtils::toString(m));
}
