#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::internal::experimental::FeaVoxelGrid {

template <typename BuildingBlockGridT>
struct FeaNode : private BuildingBlockGridT::Cell
{
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Cell = typename BuildingBlockGridT::Cell;
    };


    FeaNode() = default;
    explicit FeaNode(typename BuildingBlocks::Cell&) ;

    auto getBuildingBlock() -> typename BuildingBlocks::Cell&;

    NEON_CUDA_HOST_DEVICE inline explicit FeaNode(const typename BuildingBlocks::Cell::Location::Integer& x,
                                               const typename BuildingBlocks::Cell::Location::Integer& y,
                                               const typename BuildingBlocks::Cell::Location::Integer& z);

    NEON_CUDA_HOST_DEVICE inline explicit FeaNode(const typename BuildingBlocks::Cell::Location& location);

    NEON_CUDA_HOST_DEVICE inline auto set() -> typename BuildingBlocks::Cell::Location&;

    NEON_CUDA_HOST_DEVICE inline auto get() const -> const typename BuildingBlocks::Cell::Location&;
};


}  // namespace Neon::domain::internal::experimental::FeaVoxelGrid

