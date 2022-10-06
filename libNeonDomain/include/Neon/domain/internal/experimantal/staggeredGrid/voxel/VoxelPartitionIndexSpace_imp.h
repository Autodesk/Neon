#pragma once

#include "Neon/core/core.h"
#include "VoxelGeneric.h"
#include "VoxelPartitionIndexSpace.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {


template <typename BuildingBlockGridT>
inline auto VoxelPartitionIndexSpace<BuildingBlockGridT>::setAndValidate(Cell&                          cell,
                                                                         const size_t&                  x,
                                                                         [[maybe_unused]] const size_t& y,
                                                                         [[maybe_unused]] const size_t& z)
    const
    -> bool
{
    mBuildingBlocksPIdxSpace.setAndValidate(cell.getBuildingBlockCell(), x, y, z);
    return mActiveFlag(cell.getBuildingBlockCell(), 0) == 1;
}

template <typename BuildingBlockGridT>
VoxelPartitionIndexSpace<BuildingBlockGridT>::VoxelPartitionIndexSpace(const typename BuildingBlocks::PartitionIndexSpace&                       pixs,
                                                                       const typename BuildingBlockGridT::template Field<uint8_t, 1>::Partition& flag)
{
    mBuildingBlocksPIdxSpace = pixs;
    mActiveFlag = flag;
}
}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
