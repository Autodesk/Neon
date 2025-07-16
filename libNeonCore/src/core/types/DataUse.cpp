//
// Created by max on 2020-11-02.
//
#include "Neon/core/types/DataUse.h"
#include "Neon/core/core.h"
namespace Neon {

class DevSet;


auto DataUseUtils::toString(Neon::DataUse option) -> const char*
{
    switch (option) {
        case Neon::DataUse::HOST_DEVICE: {
            return "HOST_DEVICE";
        }
        case Neon::DataUse::DEVICE: {
            return "DEVICE";
        }
        case Neon::DataUse::HOST: {
            return "HOST";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPERATION("DataUse_e: Option not recognized.")
        }
    }
}


auto DataUseUtils::fromInt(int val) -> DataUse
{
    switch (val) {
        case static_cast<int>(DataUse::HOST_DEVICE): {
            return DataUse::HOST_DEVICE;
        }
        case static_cast<int>(DataUse::DEVICE): {
            return DataUse::DEVICE;

            case static_cast<int>(DataUse::HOST): {
                return DataUse::HOST;
            }
            default: {
                NEON_THROW_UNSUPPORTED_OPTION("DataViewUtil");
            }
        }
    }
}

auto DataUseUtils::toInt(DataUse dataUse) -> int
{
    return static_cast<int>(dataUse);
}


std::ostream& operator<<(std::ostream& os, Neon::DataUse const& m)
{
    return os << std::string(Neon::DataUseUtils::toString(m));
}

}  // namespace Neon
