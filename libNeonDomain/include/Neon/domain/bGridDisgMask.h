#pragma once
#include "Neon/domain/details/bGridDisgMask/bGrid.h"

namespace Neon {
using bGridMask = Neon::domain::details::disaggregated::bGridMask::bGrid<Neon::domain::details::disaggregated::bGridMask::StaticBlock<4, 4, 4>>;
}