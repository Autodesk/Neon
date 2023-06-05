#pragma once
#include "MemoryLayout.h"
#include <string>

namespace Neon {

struct memPadding_e
{
    enum e
    {
        OFF = 0,
        ON = 1
    };
    static auto toString(int config) -> const char*;
};

struct memAlignment_e
{
    enum e
    {
        SYSTEM = 0,      /** Basic alignment provided by the allocator */
        L1 = 1,          /** Alignment based on L1 cache size */
        L2 = 2,          /** Alignment based on L2 cache size */
        PAGE = 3,        /** Alignment based on memory page size */
    };
    static auto toString(int config) -> const char*;
};


}  // namespace Neon
