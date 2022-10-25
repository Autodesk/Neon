#pragma once

namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT, typename TypeT, int CardinalityT>
auto VoxelPartition<BuildingBlockGridT, TypeT, CardinalityT>::
     operator()(const Voxel& voxelHandle /**! Handle to a voxel */,
           int          cardinalityIdx /**! Target cardinality */)
    -> TypeT&
{
    return mBuildingBlockPartition(voxelHandle.getBuildingBlockCell(), cardinalityIdx);
}

template <typename BuildingBlockGridT, typename TypeT, int CardinalityT>
auto VoxelPartition<BuildingBlockGridT, TypeT, CardinalityT>::
     operator()(const Voxel& voxelHandle /**! Handle to a voxel */,
           int          cardinalityIdx /**! Target cardinality */)
    const -> const TypeT&
{
    return mBuildingBlockPartition(voxelHandle.getBuildingBlockCell(), cardinalityIdx);
}

template <typename BuildingBlockGridT, typename TypeT, int CardinalityT>
auto VoxelPartition<BuildingBlockGridT, TypeT, CardinalityT>::
    cardinality()
        const -> int
{
    return mBuildingBlockPartition.cardinality();
}

template <typename BuildingBlockGridT, typename TypeT, int CardinalityT>
template <int8_t sx,
          int8_t sy,
          int8_t sz>
auto VoxelPartition<BuildingBlockGridT, TypeT, CardinalityT>::
    getNghVoxelValue(const Node&  node,
                     int          cardinalityIdx,
                     const TypeT& alternativeVal)
        const -> NghInfo<TypeT>
{
    // STEPS
    // 1. check locally if the neighbour node exists. if it does not, return 'alternativeVal'
    // 2. read the neighbour value
    {  // STEP 1
        const NodeToVoxelMask& actvieVoxMask = mNodeToVoxelMaskPartition(node.getBuildingBlockCell(), 0);
        const bool             isActive = actvieVoxMask.isNeighbourVoxelValid<sx, sy, sz>();
        if (!isActive) {
            return NghInfo(alternativeVal, false);
        }
    }
    return mBuildingBlockPartition.template nghVal < sx == 1 ? 0 : sx,
           sy == 1 ? 0 : sy,
           sz == 1 ? 0 : sz > (node.getBuildingBlockCell(), cardinalityIdx, alternativeVal);
}


template <typename BuildingBlockGridT, typename TypeT, int CardinalityT>
auto VoxelPartition<BuildingBlockGridT, TypeT, CardinalityT>::
    getNghVoxelValue(const Node&                  node,
                     const std::array<int8_t, 3>& offset3D,
                     int                          cardinalityIdx,
                     const TypeT&                 alternativeVal)
        const -> NghInfo<TypeT>
{
    const Neon::int8_3d xyz(offset3D[0] == 1 ? 0 : -1,
                            offset3D[1] == 1 ? 0 : -1,
                            offset3D[2] == 1 ? 0 : -1);
    // STEPS
    // 1. check locally if the neighbour node exists. if it does not, return 'alternativeVal'
    // 2. read the neighbour value
    {  // STEP 1
        const NodeToVoxelMask& actvieVoxMask = mNodeToVoxelMaskPartition(node.getBuildingBlockCell(), 0);
        const bool             isActive = actvieVoxMask.isNeighbourVoxelValid(offset3D[0], offset3D[1], offset3D[2]);
        if (!isActive) {
            return NghInfo(alternativeVal, false);
        }
    }
    return mBuildingBlockPartition.nghVal(node.getBuildingBlockCell(),
                                          xyz,
                                          cardinalityIdx,
                                          alternativeVal);
}

template <typename BuildingBlockGridT, typename TypeT, int CardinalityT>
VoxelPartition<BuildingBlockGridT, TypeT, CardinalityT>::VoxelPartition(const typename BuildingBlocks::Partition&                partition,
                                                                        const typename BuildingBlocks::PartitionNodeToVoxelMask& partitionNodeToVoxelMask)
{
    mBuildingBlockPartition = partition;
    mNodeToVoxelMaskPartition = partitionNodeToVoxelMask;
}
}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
