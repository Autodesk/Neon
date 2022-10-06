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
    };

    using Self = NodePartition<BuildingBlockGridT, T_ta, cardinality_ta>;
    using PartitionIndexSpace = typename BuildingBlockGridT::PartitionIndexSpace;
    using Voxel = VoxelGeneric<BuildingBlockGridT>;
    using Type = T_ta;

   public:
    VoxelPartition() = default;

    ~VoxelPartition() = default;

    explicit VoxelPartition(const typename BuildingBlocks::Partition& partition)
    {
        mBuildingBlockPartition = partition;
    }

    NEON_CUDA_HOST_DEVICE inline auto operator()(const Voxel& node,
                                                 int          cardinalityIdx) -> T_ta&
    {
        return mBuildingBlockPartition(node.getBuildingBlockCell(), cardinalityIdx);
    }

    NEON_CUDA_HOST_DEVICE inline auto operator()(const Voxel& node,
                                                 int          cardinalityIdx) const -> const T_ta&
    {
        return mBuildingBlockPartition(node.getBuildingBlockCell(), cardinalityIdx);
    }


    NEON_CUDA_HOST_DEVICE inline auto cardinality() const -> int
    {
        return mBuildingBlockPartition.cardinality();
    }

    template <int8_t sx,
              int8_t sy,
              int8_t sz>
    NEON_CUDA_HOST_DEVICE inline auto operator()(const typename Self::Element& element,
                                                 int                           cardinalityIdx) -> T_ta
    {
        return BuildingBlocks::Partition::nghVal() < sx == -1 ? 0 : sx,
               sy == -1 ? 0 : sy,
               sz == -1 ? 0 : sz > (element, cardinalityIdx);
    }

   private:
    typename BuildingBlocks::Partition mBuildingBlockPartition;
};
}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
