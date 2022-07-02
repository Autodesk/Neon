#include "Neon/set/ContainerTools/ContainerType.h"

/**
 * Abstract interface to hide
 */

namespace Neon::set::internal {

auto ContainerTypeUtils::toString(ContainerType option) -> std::string
{
    switch (option) {
        case ContainerType::device: {
            return "device";
        }
        case ContainerType::deviceManaged: {
            return "deviceManaged";
        }
        case ContainerType::hostManaged: {
            return "hostManaged";
        }
        case ContainerType::deviceThenHostManaged: {
            return "deviceThenHostManaged";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto ContainerTypeUtils::fromString(const std::string& option) -> ContainerType
{
    auto const options = getOptions();

    for (auto a : options) {
        if (toString(a) == option) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto ContainerTypeUtils::getOptions() -> std::array<ContainerType, nOptions>
{
    std::array<ContainerType, nOptions> opts = {ContainerType::device,
                                                ContainerType::deviceManaged,
                                                ContainerType::deviceThenHostManaged};
    return opts;
}

}  // namespace Neon