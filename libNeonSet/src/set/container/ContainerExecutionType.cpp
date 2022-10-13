#include "Neon/set/container/ContainerExecutionType.h"

/**
 * Abstract interface to hide
 */

namespace Neon::set {

auto ContainerExecutionTypeUtils::toString(ContainerExecutionType option) -> std::string
{
    switch (option) {
        case ContainerExecutionType::device: {
            return "device";
        }
        case ContainerExecutionType::deviceManaged: {
            return "deviceManaged";
        }
        case ContainerExecutionType::hostManaged: {
            return "hostManaged";
        }
        case ContainerExecutionType::deviceThenHostManaged: {
            return "deviceThenHostManaged";
        }
        case ContainerExecutionType::graph: {
            return "graph";
        }
        case ContainerExecutionType::none: {
            return "none";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto ContainerExecutionTypeUtils::fromString(const std::string& option)
    -> ContainerExecutionType
{
    auto const options = getOptions();

    for (auto a : options) {
        if (toString(a) == option) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto ContainerExecutionTypeUtils::getOptions() -> std::array<ContainerExecutionType, nOptions>
{
    std::array<ContainerExecutionType, nOptions> opts = {ContainerExecutionType::device,
                                                         ContainerExecutionType::deviceManaged,
                                                         ContainerExecutionType::deviceThenHostManaged,
                                                         ContainerExecutionType::hostManaged,
                                                         ContainerExecutionType::graph};
    return opts;
}

auto ContainerExecutionTypeUtils::isExpandable(ContainerExecutionType option) -> bool
{
    if (option == ContainerExecutionType::graph ||
        option == ContainerExecutionType::deviceThenHostManaged) {
        return true;
    }
    return false;
}

std::ostream& operator<<(std::ostream& os, Neon::set::ContainerExecutionType const& m)
{
    return os << Neon::set::ContainerExecutionTypeUtils::toString(m);
}

}  // namespace Neon::set
