#pragma once
#if !defined(NEON_WARP_COMPILATION)
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include "Neon/Report.h"
#include "Neon/core/core.h"
#include "Neon/set/MemoryOptions.h"
// #include "Neon/core/types/mode.h"
// #include "Neon/core/types/devType.h"
#endif


namespace Neon::set::details {

/*
 * Type of thread grids
 */
enum struct ExecutionThreadSpan
{
    d1 = 0 /**< 1D dense thread grid */,
    d2 = 1 /**< 1D dense thread grid */,
    d3 = 2 /**< 1D dense thread grid */,
    d1b3 = 3 /**< 3D blocks arranged in a 1D */
};


struct ExecutionThreadSpanUtils
{
    static constexpr int nOptions = 4;
    static auto toString(ExecutionThreadSpan runtime) -> std::string;

    static constexpr auto isBlockSpan(ExecutionThreadSpan executionThreadSpan){
        return executionThreadSpan == ExecutionThreadSpan::d1b3;
    }
};

}  // namespace Neon::set::details
