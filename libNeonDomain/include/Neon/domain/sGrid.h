#pragma once
#include "Neon/domain/details/sGrid/sGrid.h"

namespace Neon::domain {
template <typename OuterGrid>
using sGrid = Neon::domain::internal::sGrid::sGrid<OuterGrid>;
}