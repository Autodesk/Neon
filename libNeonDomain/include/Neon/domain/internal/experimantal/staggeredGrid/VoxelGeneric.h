#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT>
struct VoxelGeneric : private BuildingBlockGridT::Cell
{

    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Cell = typename BuildingBlockGridT::Cell;
    };


    VoxelGeneric() = default;
    explicit VoxelGeneric(typename BuildingBlocks::Cell&);

    auto getBuildingBlock() -> typename BuildingBlocks::Cell&;

    NEON_CUDA_HOST_DEVICE inline explicit VoxelGeneric(const typename BuildingBlocks::Cell::Location::Integer& x,
                                                       const typename BuildingBlocks::Cell::Location::Integer& y,
                                                       const typename BuildingBlocks::Cell::Location::Integer& z);

    NEON_CUDA_HOST_DEVICE inline explicit VoxelGeneric(const typename BuildingBlocks::Cell::Location& location);

    NEON_CUDA_HOST_DEVICE inline auto set() -> typename BuildingBlocks::Cell::Location&;

    NEON_CUDA_HOST_DEVICE inline auto get() const -> const typename BuildingBlocks::Cell::Location&;
};

// using Element = Element<void>;

}  // namespace Neon::domain::internal::experimental::staggeredGrid
