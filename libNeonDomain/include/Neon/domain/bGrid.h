#pragma once
#include "Neon/domain/details/bGrid/bGrid.h"

namespace Neon {
using bGrid = Neon::domain::details::bGrid::bGrid<Neon::domain::details::bGrid::StaticBlock<8,8,8>>;
}