#pragma once
#include "Neon/domain/details/bGrid/bGrid.h"

namespace Neon {

template <typename SBlock>
using bGridGenericBlock = Neon::domain::details::bGrid::bGrid<SBlock>;
using bGrid = bGridGenericBlock<Neon::domain::details::bGrid::BlockDefault>;

}  // namespace Neon