#pragma once

#include <string>

namespace Neon {

enum class MemoryLayout
{
    structOfArrays = 0,
    arrayOfStructs = 1
};

class MemoryLayoutUtils
{
   public:
    static auto toString(const MemoryLayout& config) -> const char*;
    static auto toInt(const MemoryLayout& config) -> int;
    static auto fromInt(const int& config) -> MemoryLayout;
};

}  // namespace Neon
