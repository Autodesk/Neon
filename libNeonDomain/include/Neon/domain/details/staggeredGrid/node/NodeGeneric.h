#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::details::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT>
struct NodePartitionIndexSpace;

template <typename BuildingBlockGridT,
          typename T_ta,
          int cardinality_ta>
struct NodePartition;

template <typename BuildingBlockGridT,
          typename T_ta,
          int cardinality_ta >
struct VoxelPartition;

template <typename BuildingBlockGridT>
struct NodeGeneric
{
   private:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Cell = typename BuildingBlockGridT::Cell;
    };

   public:
    using OuterIdx = typename BuildingBlockGridT::Cell::OuterIdx;
    using Location = typename BuildingBlocks::Cell::Location;

    friend struct NodePartitionIndexSpace<BuildingBlockGridT>;

    template <typename BuildingBlockGridTT,
              typename T_ta,
              int cardinality_ta>
    friend struct NodePartition;

    template <typename BuildingBlockGridTT,
              typename T_ta,
              int cardinality_ta>
    friend struct VoxelPartition;

    NodeGeneric() = default;

    NEON_CUDA_HOST_DEVICE
    inline explicit NodeGeneric(const Location& location);

   private:
    NEON_CUDA_HOST_DEVICE
    inline auto getBuildingBlockCell()
        -> typename BuildingBlocks::Cell&;

    NEON_CUDA_HOST_DEVICE
    inline auto getBuildingBlockCell() const
        -> const typename BuildingBlocks::Cell&;

    typename BuildingBlockGridT::Cell mBuildingBlockCell;
};


}  // namespace Neon::domain::details::experimental::staggeredGrid::details
