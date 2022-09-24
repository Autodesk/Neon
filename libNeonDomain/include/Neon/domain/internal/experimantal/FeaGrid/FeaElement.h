#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::internal::experimental::FeaVoxelGrid {

template <typename BuildingBlockGridT>
struct FeaElement : private BuildingBlockGridT::Cell
{

    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Cell = typename BuildingBlockGridT::Cell;
    };


    FeaElement() = default;
    explicit FeaElement(typename BuildingBlocks::Cell&) ;

    auto getBuildingBlock() -> typename BuildingBlocks::Cell&;

    NEON_CUDA_HOST_DEVICE inline explicit FeaElement(const typename BuildingBlocks::Cell::Location::Integer& x,
                                                  const typename BuildingBlocks::Cell::Location::Integer& y,
                                                  const typename BuildingBlocks::Cell::Location::Integer& z);

    NEON_CUDA_HOST_DEVICE inline explicit FeaElement(const typename BuildingBlocks::Cell::Location& location);

    NEON_CUDA_HOST_DEVICE inline auto set() -> typename BuildingBlocks::Cell::Location&;

    NEON_CUDA_HOST_DEVICE inline auto get() const -> const typename BuildingBlocks::Cell::Location&;
};

// using Element = Element<void>;

}  // namespace Neon::domain::internal::experimental::FeaVoxelGrid

