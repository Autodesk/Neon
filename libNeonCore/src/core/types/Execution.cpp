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
            return static_cast<int>(Neon::Execution::device);
        }
        case Neon::Execution::host: {
            return static_cast<int>(Neon::Execution::host);
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("ExecutionUtils");
}

auto ExecutionUtils::getAllOptions() -> const std::array<Execution, ExecutionUtils::numConfigurations>&
{
    return mAllOptions;
}

auto ExecutionUtils::getCompatibleOptions(Neon::DataUse dataUse)
    -> std::vector<Execution>
{
    switch (dataUse) {
        case DataUse::HOST_DEVICE: {
            return {Execution::device,
                    Execution::host};
        }
        case DataUse::DEVICE: {
            return {Execution::device};
        }
        case DataUse::HOST: {
            return {Execution::host};
        }
    }
    NEON_THROW_UNSUPPORTED_OPERATION("");
}

auto ExecutionUtils::checkCompatibility(Neon::DataUse   dataUse,
                                        Neon::Execution execution)
    -> bool
{
    switch (dataUse) {
        case DataUse::HOST_DEVICE: {
            return true;
        }
        case DataUse::DEVICE: {
            return execution == Execution::device;
        }
        case DataUse::HOST: {
            return execution == Execution::host;
        }
    }
    NEON_THROW_UNSUPPORTED_OPERATION("");
}


auto ExecutionUtils::fromInt(int val) -> Execution
{
    switch (val) {
        case static_cast<int>(Execution::device): {
            return Execution::device;
        }
        case static_cast<int>(Execution::host): {
            return Execution::host;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("DataViewUtil");
        }
    }
}

std::ostream& operator<<(std::ostream& os, Neon::Execution const& m)
{
    return os << std::string(Neon::ExecutionUtils::toString(m));
}
}  // namespace Neon
