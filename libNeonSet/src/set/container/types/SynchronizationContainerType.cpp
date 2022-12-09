#include "Neon/set/container/types/SynchronizationContainerType.h"

/**
 * Abstract interface to hide
 */

namespace Neon::set {

auto SynchronizationContainerTypeUtils::toString(SynchronizationContainerType option) -> std::string
{
    switch (option) {
        case SynchronizationContainerType::hostOmpBarrier: {
            return "hostOmpBarrier";
        }
        case SynchronizationContainerType::none: {
            return "none";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto SynchronizationContainerTypeUtils::fromString(const std::string& option)
    -> SynchronizationContainerType
{
    auto const options = getOptions();

    for (auto a : options) {
        if (toString(a) == option) {
            return a;
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

auto SynchronizationContainerTypeUtils::getOptions() -> std::array<SynchronizationContainerType, nOptions>
{
    std::array<SynchronizationContainerType, nOptions> opts = {SynchronizationContainerType::hostOmpBarrier};
    return opts;
}

std::ostream& operator<<(std::ostream& os, Neon::set::SynchronizationContainerType const& m)
{
    return os << Neon::set::SynchronizationContainerTypeUtils::toString(m);
}

}  // namespace Neon::set
