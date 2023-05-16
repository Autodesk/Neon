#pragma once
#include "Neon/set/Backend.h"

namespace Neon {


/**
 * Enumeration for the supported type of computation by the skeleton
 * */
enum struct Pattern
{
    MAP /**< Map operation */,
    STENCIL /**< Stencil operation */,
    REDUCE /**< Reduction operation */
};

struct PatternUtils
{
    /**
     * Returns a string for the selected allocator
     *
     * @param allocator
     * @return
     */
    static auto toString(Pattern val) -> std::string;
};

}  // namespace Neon