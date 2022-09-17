#pragma once
#include "Neon/set/Backend.h"

namespace Neon {


/**
 * Enumeration for the supported type of computation by the skeleton
 * */
enum struct Compute
{
    MAP /**< Map operation */,
    STENCIL /**< Stencil operation */,
    REDUCE /**< Reduction operation */
};

struct ComputeUtils
{
    /**
     * Returns a string for the selected allocator
     *
     * @param allocator
     * @return
     */
    static auto toString(Compute val) -> std::string;
};

}  // namespace Neon