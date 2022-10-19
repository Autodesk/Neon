#pragma once

#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/domain/interface/NghInfo.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/voxel/VoxelGeneric.h"
#include "Neon/set/DevSet.h"
#include "Neon/sys/memory/CudaIntrinsics.h"
#include "Neon/sys/memory/mem3d.h"
#include "NodeGeneric.h"
#include "NodeToVoxelMask.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT,
          typename T_ta,
          int cardinality_ta = 0>
struct NodePartition
{
   public:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Partition = typename BuildingBlockGridT::template Partition<T_ta, cardinality_ta>;
    };

    using Self = NodePartition<BuildingBlockGridT, T_ta, cardinality_ta>;
    using PartitionIndexSpace = typename BuildingBlockGridT::PartitionIndexSpace;
    using Node = NodeGeneric<BuildingBlockGridT>;
    using Voxel = VoxelGeneric<BuildingBlockGridT>;
    using Type = T_ta;

   public:
    NodePartition() = default;

    ~NodePartition() = default;

    explicit NodePartition(const typename BuildingBlocks::Partition& partition)
    {
        mBuildingBlockPartition = partition;
    }

    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Self::Node& node,
               int               cardinalityIdx)
        -> T_ta&
    {
        return mBuildingBlockPartition(node.getBuildingBlockCell(), cardinalityIdx);
    }

    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Self::Node& node,
               int               cardinalityIdx) const
        -> const T_ta&
    {
        return mBuildingBlockPartition(node.getBuildingBlockCell(), cardinalityIdx);
    }


    NEON_CUDA_HOST_DEVICE inline auto cardinality() const
        -> int
    {
        return mBuildingBlockPartition.cardinality();
    }

    // From a voxel handle to node data
    template <int8_t sx,
              int8_t sy,
              int8_t sz>
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Self::Voxel& element,
               int                cardinalityIdx) const -> T_ta
    {
        return BuildingBlocks::Partition::nghVal() < sx == -1 ? 0 : sx,
               sy == -1 ? 0 : sy,
               sz == -1 ? 0 : sz > (element, cardinalityIdx);
    }

    template <int8_t sx,
              int8_t sy,
              int8_t sz>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghNodeValue(const Voxel& voxel,
                    int          cardinalityIdx) const -> T_ta
    {
        constexpr int8_t x = sx == -1 ? 0 : 1;
        constexpr int8_t y = sy == -1 ? 0 : 1;
        constexpr int8_t z = sz == -1 ? 0 : 1;
        T_ta             alternative;
        return mBuildingBlockPartition.template nghVal<x, y, z>(voxel.getBuildingBlockCell(), cardinalityIdx, alternative).value;
    }

   private:
    typename BuildingBlocks::Partition mBuildingBlockPartition;
};
}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
