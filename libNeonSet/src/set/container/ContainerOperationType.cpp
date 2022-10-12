#include "Neon/set/container/ContainerOperationType.h"

/**
 * Abstract interface to hide
 */

namespace Neon::set {

auto ContainerOperationTypeUtils::toString(ContainerOperationType option) -> std::string
{
    switch (option) {
        case ContainerOperationType::compute: {
            return "compute";
        }
        case ContainerOperationType::graph: {
            return "graph";
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
                                                         ContainerOperationType::graph,
                                                         ContainerOperationType::halo,
                                                         ContainerOperationType::sync,
                                                         ContainerOperationType::anchor};
    return opts;
}

std::ostream& operator<<(std::ostream& os, Neon::set::ContainerOperationType const& m)
{
    return os << std::string(Neon::set::ContainerOperationTypeUtils::toString(m));
}
}  // namespace Neon::set::internal



