#pragma once

#include "NodeGeneric.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT>
NodeGeneric<BuildingBlockGridT>::NodeGeneric(const typename BuildingBlocks::Cell::Location& location)
{
    mBuildingBlockCell = BuildingBlocks::Cell(location);
}

template <typename BuildingBlockGridT>
inline auto
NodeGeneric<BuildingBlockGridT>::getBuildingBlockCell()
    -> typename BuildingBlocks::Cell&
{
    return mBuildingBlockCell;
}


template <typename BuildingBlockGridT>
NEON_CUDA_HOST_DEVICE
inline auto
NodeGeneric<BuildingBlockGridT>::getBuildingBlockCell() const
    -> const typename BuildingBlocks::Cell&
{
    return mBuildingBlockCell;
}

}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
