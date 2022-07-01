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
        case Neon::DataUse::IO_COMPUTE: {
            return "IO_COMPUTE";
        }
        case Neon::DataUse::COMPUTE: {
            return "COMPUTE";
        }
        case Neon::DataUse::IO_POSTPROCESSING: {
            return "IO_POSTPROCESSING";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPERATION("DataUse_e: Option not recognized.")
        }
    }
}
}  // namespace Neon


std::ostream& operator<<(std::ostream& os, Neon::DataUse const& m)
{
    return os << std::string(Neon::DataUseUtils::toString(m));
}