#include "Neon/core/types/Execution.h"
#include "Neon/core/core.h"
namespace Neon {

auto ExecutionUtils::toString(Neon::Execution option) -> const char*
{
    switch (option) {
        case Neon::Execution::device: {
            return "DeviceExecution";
        }
        case Neon::Execution::host: {
            return "HostExecution";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("ExecutionUtils");
}


auto ExecutionUtils::toInt(Neon::Execution option) -> int
{
    switch (option) {
        case Neon::Execution::device: {
            return  static_cast<int>(Neon::Execution::device);
        }
        case Neon::Execution::host: {
            return  static_cast<int>(Neon::Execution::host);
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("ExecutionUtils");

}

auto ExecutionUtils::getAllOptions() -> const std::array<Execution, ExecutionUtils::numConfigurations>&
{
    return mAllOptions;
}

}  // namespace Neon

std::ostream& operator<<(std::ostream& os, Neon::Execution const& m)
{
    return os << std::string(Neon::ExecutionUtils::toString(m));
}