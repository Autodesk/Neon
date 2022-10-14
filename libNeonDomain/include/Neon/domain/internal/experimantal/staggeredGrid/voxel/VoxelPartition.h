#pragma once
#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/NghInfo.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/voxel/VoxelGeneric.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
#include "Neon/sys/memory/mem3d.h"
#include "VoxelGeneric.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT, typename T, int C>
struct VoxelStorage;


template <typename BuildingBlockGridT,
          typename T_ta,
          int cardinality_ta = 0>
struct VoxelPartition
{
   public:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Partition = typename BuildingBlockGridT::template Partition<T_ta, cardinality_ta>;
        using PartitionNodeToVoxelMask = typename BuildingBlockGridT::template Partition<NodeToVoxelMask, 1>;
    };

    using Self = NodePartition<BuildingBlockGridT, T_ta, cardinality_ta>;
    using PartitionIndexSpace = typename BuildingBlockGridT::PartitionIndexSpace;
    using Voxel = VoxelGeneric<BuildingBlockGridT>;
    using Node = NodeGeneric<BuildingBlockGridT>;
    using Type = T_ta;

    friend VoxelStorage<BuildingBlockGridT, T_ta, cardinality_ta>;

   public:
    VoxelPartition() = default;

    ~VoxelPartition() = default;

    /**
     * To access metadata associated to a voxel by this field partition.
     */
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Voxel& voxelHandle /**! Handle to a voxel */,
               int          cardinalityIdx /**! Target cardinality */) -> T_ta&
    {
        return mBuildingBlockPartition(voxelHandle.getBuildingBlockCell(), cardinalityIdx);
    }

    /**
     * To access metadata associated to a voxel by this field partition.
     */
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Voxel& voxelHandle /**! Handle to a voxel */,
               int          cardinalityIdx /**! Target cardinality */) const -> const T_ta&
    {
        return mBuildingBlockPartition(voxelHandle.getBuildingBlockCell(), cardinalityIdx);
    }

    /**
     * Returns the cardinality of this field.
     */
    NEON_CUDA_HOST_DEVICE inline auto cardinality() const
        -> int
    {
        return mBuildingBlockPartition.cardinality();
    }

    /**
     * Function to read voxel values from a node
     */
    template <int8_t sx,
              int8_t sy,
              int8_t sz>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghVoxelValue(const Node& node,
                     int         cardinalityIdx,
                     const T_ta& alternativeVal) const -> NghInfo<T_ta>
    {
        // STEPS
        // 1. check locally if the neighbour node exists. if it does not, return 'alternativeVal'
        // 2. read the neighbour value
        {  // STEP 1
            const NodeToVoxelMask& actvieVoxMask = mNodeToVoxelMaskPartition(node.getBuildingBlockCell(), 1);
            const bool             isActive = actvieVoxMask.isNeighbourVoxelValid<sx, sy, sz>();
            if (!isActive) {
                return NghInfo(alternativeVal, false);
            }
        }
        return mBuildingBlockPartition.template nghVal < sx == -1 ? 0 : sx,
               sy == -1 ? 0 : sy,
               sz == -1 ? 0 : sz > (node.getBuildingBlockCell(), cardinalityIdx, alternativeVal);
    }

   private:
    explicit VoxelPartition(const typename BuildingBlocks::Partition&                partition,
                            const typename BuildingBlocks::PartitionNodeToVoxelMask& partitionNodeToVoxelMask)
    {
        mBuildingBlockPartition = partition;
        mNodeToVoxelMaskPartition = partitionNodeToVoxelMask;
    }

    typename BuildingBlocks::Partition                mBuildingBlockPartition;
    typename BuildingBlocks::PartitionNodeToVoxelMask mNodeToVoxelMaskPartition;
};
}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
