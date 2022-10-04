#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {

template <typename BuildingBlockGridT>
struct NodeGeneric : public BuildingBlockGridT::Cell
{
    using OuterCell = typename BuildingBlockGridT::Cell::OuterCell;

    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Cell = typename BuildingBlockGridT::Cell;
    };


    NodeGeneric() = default;

    explicit NodeGeneric(typename BuildingBlocks::Cell&);

    auto getBuildingBlock()
        -> typename BuildingBlocks::Cell&;

    NEON_CUDA_HOST_DEVICE
    inline explicit NodeGeneric(const typename BuildingBlocks::Cell::Location& location);

    NEON_CUDA_HOST_DEVICE
    inline auto set()
        -> typename BuildingBlocks::Cell::Location&;

    NEON_CUDA_HOST_DEVICE inline auto get()
        const -> const typename BuildingBlocks::Cell::Location&;

   private:
};


}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
