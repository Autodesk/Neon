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

std::ostream& operator<<(std::ostream& os, Neon::DataUse const& m)
{
    return os << std::string(Neon::DataUseUtils::toString(m));
}

}  // namespace Neon
