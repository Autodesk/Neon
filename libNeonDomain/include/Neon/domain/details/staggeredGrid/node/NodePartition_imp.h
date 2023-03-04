#pragma once

#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/NghData.h"
#include "Neon/domain/details/staggeredGrid/voxel/VoxelGeneric.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
#include "Neon/sys/memory/mem3d.h"
#include "NodeGeneric.h"
#include "NodeToVoxelMask.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT,
          typename TypeT,
          int CardinalityT>
NodePartition<BuildingBlockGridT, TypeT, CardinalityT>::
    NodePartition(const typename BuildingBlocks::Partition& partition)
{
    mBuildingBlockPartition = partition;
}

template <typename BuildingBlockGridT,
          typename TypeT,
          int CardinalityT>
auto NodePartition<BuildingBlockGridT, TypeT, CardinalityT>::
     operator()(const Self::Node& node,
           int               cardinalityIdx)
    -> TypeT&
{
    return mBuildingBlockPartition(node.getBuildingBlockCell(), cardinalityIdx);
}

template <typename BuildingBlockGridT,
          typename TypeT,
          int CardinalityT>
auto NodePartition<BuildingBlockGridT, TypeT, CardinalityT>::
     operator()(const Self::Node& node,
           int               cardinalityIdx)
    const -> const TypeT&
{
    return mBuildingBlockPartition(node.getBuildingBlockCell(), cardinalityIdx);
}


template <typename BuildingBlockGridT,
          typename TypeT,
          int CardinalityT>
auto NodePartition<BuildingBlockGridT, TypeT, CardinalityT>::
    cardinality()
        const -> int
{
    return mBuildingBlockPartition.cardinality();
}

template <typename BuildingBlockGridT,
          typename TypeT,
          int CardinalityT>
template <int8_t sx,
          int8_t sy,
          int8_t sz>
auto NodePartition<BuildingBlockGridT, TypeT, CardinalityT>::
    getNghNodeValue(const Voxel& voxel,
                    int          cardinalityIdx)
        const -> TypeT
{
    constexpr int8_t x = sx == -1 ? 0 : 1;
    constexpr int8_t y = sy == -1 ? 0 : 1;
    constexpr int8_t z = sz == -1 ? 0 : 1;
    TypeT            alternative;
    return mBuildingBlockPartition.template getNghData<x, y, z>(voxel.getBuildingBlockCell(),
                                                                cardinalityIdx,
                                                                alternative)
        .value;
}

template <typename BuildingBlockGridT,
          typename TypeT,
          int CardinalityT>
auto NodePartition<BuildingBlockGridT, TypeT, CardinalityT>::
    getNghNodeValue(const Voxel&                 voxel,
                    const std::array<int8_t, 3>& offset3D,
                    int                          cardinalityIdx)
        const -> TypeT
{
    const Neon::int8_3d xyz(offset3D[0] == -1 ? 0 : 1,
                            offset3D[1] == -1 ? 0 : 1,
                            offset3D[2] == -1 ? 0 : 1);

    TypeT alternative;
    return mBuildingBlockPartition.getNghData(voxel.getBuildingBlockCell(),
                                              xyz,
                                              cardinalityIdx,
                                              alternative)
        .value;
}

}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
