#include "Neon/core/core.h"
#include "Neon/core/types/Place.h"
namespace Neon {

auto PlaceUtils::toString(Neon::Place option) -> const char*
{
    switch (option) {
        case Neon::Place::device: {
            return "DeviceExecution";
        }
        case Neon::Place::host: {
            return "HostExecution";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("PlaceUtils");
}


auto PlaceUtils::toInt(Neon::Place option) -> int
{
    switch (option) {
        case Neon::Place::device: {
            return static_cast<int>(Neon::Place::device);
        }
        case Neon::Place::host: {
            return static_cast<int>(Neon::Place::host);
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("PlaceUtils");
}

auto PlaceUtils::getAllOptions() -> const std::array<Neon::Place, PlaceUtils::numConfigurations>&
{
    return mAllOptions;
}

auto PlaceUtils::getCompatibleOptions(Neon::DataUse dataUse)
    -> std::vector<Neon::Place>
{
    switch (dataUse) {
        case DataUse::IO_COMPUTE: {
            return {Neon::Place::device,
                    Neon::Place::host};
        }
        case DataUse::COMPUTE: {
            return {Neon::Place::device};
        }
        case DataUse::IO_POSTPROCESSING: {
            return {Neon::Place::host};
        }
    }
    NEON_THROW_UNSUPPORTED_OPERATION("");
}

auto PlaceUtils::checkCompatibility(Neon::DataUse dataUse,
                                    Neon::Place   execution)
    -> bool
{
    switch (dataUse) {
        case DataUse::IO_COMPUTE: {
            return true;
        }
        case DataUse::COMPUTE: {
            return execution == Neon::Place::device;
        }
        case DataUse::IO_POSTPROCESSING: {
            return execution == Neon::Place::host;
        }
    }
    NEON_THROW_UNSUPPORTED_OPERATION("");
}

std::ostream& operator<<(std::ostream& os, Neon::Place const& m)
{
    return os << std::string(Neon::PlaceUtils::toString(m));
}
}  // namespace Neon
