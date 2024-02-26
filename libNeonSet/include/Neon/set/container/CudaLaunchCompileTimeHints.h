#pragma once

#include "Neon/core/core.h"


namespace Neon::set::container {

template <bool inited__ = false,
          int  maxThreadsPerBlock__ = 1024,
          int  minBlocksPerMultiprocessor__ = 1,
          int  maxBlocksPerCluster__ = 0>
struct CudaLaunchCompileTimeHint
{
   public:
    static constexpr bool initialized = inited__;
    static constexpr int  maxThreadsPerBlock = maxThreadsPerBlock__;
    static constexpr int  minBlocksPerMultiprocessor = minBlocksPerMultiprocessor__;
    static constexpr int  maxBlocksPerCluster = maxBlocksPerCluster__;
};

}  // namespace Neon::set::container
