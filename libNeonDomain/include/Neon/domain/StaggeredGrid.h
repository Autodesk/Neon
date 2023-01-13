#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/internal/experimental/staggeredGrid/StaggeredGrid.h"

namespace Neon::domain::experimental {
template <typename BuildingBlockGridT>
using StaggeredGrid = Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<BuildingBlockGridT>;
}