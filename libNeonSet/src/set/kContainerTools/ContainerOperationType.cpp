#include "Neon/set/ContainerTools/ContainerOperationType.h"

/**
 * Abstract interface to hide
 */

namespace Neon::set::internal {

auto ContainerOperationTypeUtils::toString(ContainerOperationType option) -> std::string
{
    switch (option) {
        case ContainerOperationType::compute: {
            return "compute";
        }
        case ContainerOperationType::halo: {
            return "halo";
        }
        case ContainerOperationType::sync: {
            return "sync";
        }
        case ContainerOperationType::anchor: {
            return "anchor";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto ContainerOperationTypeUtils::fromString(const std::string& option)
    -> ContainerOperationType
{
    auto const options = getOptions();

    for (auto a : options) {
        if (toString(a) == option) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto ContainerOperationTypeUtils::getOptions() -> std::array<ContainerOperationType, nOptions>
{
    std::array<ContainerOperationType, nOptions> opts = {ContainerOperationType::compute,
                                                         ContainerOperationType::halo,
                                                         ContainerOperationType::sync,
                                                         ContainerOperationType::anchor};
    return opts;
}

}  // namespace Neon::set::internal


std::ostream& operator<<(std::ostream& os, Neon::set::internal::ContainerOperationType const& m)
{
    return os << std::string(Neon::set::internal::ContainerOperationTypeUtils::toString(m));
}
