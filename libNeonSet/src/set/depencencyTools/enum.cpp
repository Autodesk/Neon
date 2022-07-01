#include "Neon/set/dependencyTools/enum.h"

namespace Neon {
namespace set {
namespace internal {
namespace dependencyTools {


auto Access_et::toString(e val) -> std::string
{
    switch (val) {
        case READ: {
            return "READ";
        }
        case WRITE: {
            return "WRITE";
        }
        case NONE: {
            return "NONE";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION();
}

auto Access_et::merge(e valA, e valB) -> e {
    if(valA == e::WRITE) return e::WRITE;
    if(valB == e::WRITE) return e::WRITE;
    if(valA == e::READ) return e::READ;
    if(valB == e::READ) return e::READ;
    return e::NONE;
}

auto Dependencies_et::toString(e val) -> std::string
{
    switch (val) {
        case RAW: {
            return "RAW";
        }
        case WAR: {
            return "WAR";
        }
        case RAR: {
            return "RAR";
        }
        case WAW: {
            return "WAW";
        }
        case NONE: {
            return "NONE";
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION();
}
}  // namespace dependencyTools
}  // namespace internal
}  // namespace set

auto ComputeUtils::toString(Compute val) -> const char*
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