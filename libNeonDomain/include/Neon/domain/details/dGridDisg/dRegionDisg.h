#pragma once
#include "Neon/domain/details/dGrid/dSpan.h"
#include "Neon/set/DevSet.h"
#include "dIndexDisg.h"

namespace Neon::domain::details::dissagragated::dGrid {

/**
 * Abstraction that represents the Cell space of a partition
 * This abstraction is used by the neon lambda executor to
 * run a containers on aGrid
 */
class dRegion
{
   public:
    static constexpr int lower = 0;
    static constexpr int center = 1;
    static constexpr int upper = 2;
    static constexpr int nRetions = 3;
};

}  // namespace Neon::domain::details::dissagragated::dGrid

#include "dSpanDisg_imp.h"