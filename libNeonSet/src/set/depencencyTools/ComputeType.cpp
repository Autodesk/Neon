#include "Neon/set/dependency/ComputeType.h"

namespace Neon {

auto ComputeUtils::toString(Compute val) -> std::string
{
    switch (val) {
        case Compute::MAP: {
            return "MAP";
        }
        case Compute::STENCIL: {
            return "STENCIL";
        }
        case Compute::REDUCE: {
            return "REDUCE";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION();
}

}  // namespace Neon