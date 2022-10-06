#pragma once

#include "Neon/core/core.h"
#include "NodeGeneric.h"
#include "NodePartitionIndexSpace.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {


template <typename BuildingBlockGridT>
inline auto NodePartitionIndexSpace<BuildingBlockGridT>::setAndValidate(Cell&                          cell,
                                                                        const size_t&                  x,
                                                                        [[maybe_unused]] const size_t& y,
                                                                        [[maybe_unused]] const size_t& z)
    const
    -> bool
{
    return mBuildingBlocksPIdxSpace.setAndValidate(cell.getBuildingBlockCell(), x, y, z);
}

template <typename BuildingBlockGridT>
NodePartitionIndexSpace<BuildingBlockGridT>::NodePartitionIndexSpace(const typename BuildingBlocks::PartitionIndexSpace& pixs)
{
    mBuildingBlocksPIdxSpace = pixs;
}
}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
