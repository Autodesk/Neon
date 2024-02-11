#pragma once
#include "Neon/domain/details/bGridDisg/bGrid.h"

namespace Neon {
using bGridDisg = Neon::domain::details::disaggregated::bGrid::bGrid<Neon::domain::details::StaticBlock<4, 4, 4>>;
}