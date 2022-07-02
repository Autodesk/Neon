#include "Neon/core/types/MemoryLayout.h"
#include <vector>
#include "Neon/core//core.h"


namespace Neon {

auto MemoryLayoutUtils::toString(const MemoryLayout& config) -> const char*
{
    switch (config) {
        case MemoryLayout::structOfArrays: {
            return "structOfArrays";
        }
        case MemoryLayout::arrayOfStructs: {
            return "arrayOfStructs";
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

auto MemoryLayoutUtils::toInt(const MemoryLayout& config) -> int
{
    return static_cast<int>(config);
}

auto MemoryLayoutUtils::fromInt(const int& config) -> MemoryLayout
{
    if (config == toInt(MemoryLayout::structOfArrays)) {
        return MemoryLayout::structOfArrays;
    }

    if (config == toInt(MemoryLayout::arrayOfStructs)) {
        return MemoryLayout::arrayOfStructs;
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}

}  // namespace Neon
