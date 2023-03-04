#pragma once
#include <assert.h>
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"

#include "Neon/domain/interface/NghData.h"
#include "VoxelGeneric.h"

#include "VoxelGeneric.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT, typename T, int C>
struct VoxelStorage;


template <typename BuildingBlockGridT,
          typename TypeT,
          int CardinalityT = 0>
struct VoxelPartition
{
   public:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Partition = typename BuildingBlockGridT::template Partition<TypeT, CardinalityT>;
        using PartitionNodeToVoxelMask = typename BuildingBlockGridT::template Partition<NodeToVoxelMask, 1>;
    };

    using Self = NodePartition<BuildingBlockGridT, TypeT, CardinalityT>;
    using PartitionIndexSpace = typename BuildingBlockGridT::PartitionIndexSpace;
    using Voxel = VoxelGeneric<BuildingBlockGridT>;
    using Node = NodeGeneric<BuildingBlockGridT>;
    using Type = TypeT;

    friend VoxelStorage<BuildingBlockGridT, TypeT, CardinalityT>;

   public:

    VoxelPartition() = default;
    ~VoxelPartition() = default;

    /**
     * To access metadata associated to a voxel by this field partition.
     */
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Voxel& voxelHandle /**! Handle to a voxel */,
               int          cardinalityIdx /**! Target cardinality */)
        -> TypeT&;
    /**
     * To access voxel metadata.
     */
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Voxel& voxelHandle /**! Handle to a voxel */,
               int          cardinalityIdx /**! Target cardinality */)
        const -> const TypeT&;

    /**
     * Returns the number of components of this field.
     */
    NEON_CUDA_HOST_DEVICE inline auto cardinality()
        const -> int;

    /**
     * Function to read voxel values from a node
     */
    template <int8_t sx,
              int8_t sy,
              int8_t sz>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghVoxelValue(const Node& node,
                     int         cardinalityIdx,
                     const TypeT& alternativeVal)
        const -> NghData<TypeT>;

    /**
     * Function to read voxel values from a node
     */
    NEON_CUDA_HOST_DEVICE inline auto
    getNghVoxelValue(const Node&                  node,
                     const std::array<int8_t, 3>& offset3D,
                     int                          cardinalityIdx,
                     const TypeT&                  alternativeVal)
        const -> NghData<TypeT>;

   private:
    /**
     * Private constructor
     */
    explicit VoxelPartition(const typename BuildingBlocks::Partition&                partition,
                            const typename BuildingBlocks::PartitionNodeToVoxelMask& partitionNodeToVoxelMask);

    typename BuildingBlocks::Partition                mBuildingBlockPartition /**< building-block grid partition*/;
    typename BuildingBlocks::PartitionNodeToVoxelMask mNodeToVoxelMaskPartition /**< Active voxel mast to support node to voxel transitions */;
};

}  // namespace Neon::domain::internal::experimental::staggeredGrid::details

#include "VoxelPartition_imp.h"
