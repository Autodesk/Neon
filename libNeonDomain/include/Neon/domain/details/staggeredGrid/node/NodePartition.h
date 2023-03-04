#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/NghData.h"
#include "Neon/domain/details/staggeredGrid/voxel/VoxelGeneric.h"

#include "NodeGeneric.h"
#include "NodeToVoxelMask.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT, typename TypeT, int CardinalityT>
struct NodeStorage;

template <typename BuildingBlockGridT,
          typename TypeT,
          int CardinalityT = 0>
struct NodePartition
{
   public:
    friend NodeStorage<BuildingBlockGridT, TypeT, CardinalityT>;

    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Partition = typename BuildingBlockGridT::template Partition<TypeT, CardinalityT>;
    };

    using Self = NodePartition<BuildingBlockGridT, TypeT, CardinalityT>;
    using PartitionIndexSpace = typename BuildingBlockGridT::PartitionIndexSpace;
    using Node = NodeGeneric<BuildingBlockGridT>;
    using Voxel = VoxelGeneric<BuildingBlockGridT>;
    using Type = TypeT;

   public:
    NodePartition() = default;
    ~NodePartition() = default;

    /**
     * Accessing node metadata.
     */
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Self::Node& node /**<          Node handle */,
               int               cardinalityIdx /** Queried cardinality */)
        -> TypeT&;

    /**
     * Accessing node metadata.
     */
    NEON_CUDA_HOST_DEVICE inline auto
    operator()(const Self::Node& node /**<          Node handle */,
               int               cardinalityIdx /** Queried cardinality */)
        const -> const TypeT&;

    /**
     * Retrieving the number of components of the field
     */
    NEON_CUDA_HOST_DEVICE inline auto
    cardinality()
        const -> int;

    /**
     * Accessing neighbour node metadata
     */
    template <int8_t sx /** Neighbour offset on X */,
              int8_t sy /** Neighbour offset on Y */,
              int8_t sz /** Neighbour offset on Z */>
    NEON_CUDA_HOST_DEVICE inline auto
    getNghNodeValue(const Voxel& voxel,
                    int          cardinalityIdx)
        const -> TypeT;

    /**
     * Accessing neighbour node metadata
     */
    NEON_CUDA_HOST_DEVICE inline auto
    getNghNodeValue(const Voxel&                 voxel,
                    const std::array<int8_t, 3>& offset3D,
                    int                          cardinalityIdx)
        const -> TypeT;

   private:

    /**
     * Private constructor used only by Neon
     */
    explicit NodePartition(const typename BuildingBlocks::Partition& partition);

    typename BuildingBlocks::Partition mBuildingBlockPartition;
};
}  // namespace Neon::domain::internal::experimental::staggeredGrid::details

#include "NodePartition_imp.h"
