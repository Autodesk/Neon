#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/details/staggeredGrid/StaggeredGrid.h"

namespace Neon::domain::experimental {
template <typename BuildingBlockGridT>
using StaggeredGrid = Neon::domain::details::experimental::staggeredGrid::StaggeredGrid<BuildingBlockGridT>;
}