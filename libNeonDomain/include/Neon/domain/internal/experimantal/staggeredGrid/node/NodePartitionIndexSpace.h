#pragma once

#include "Neon/core/core.h"
#include "NodeGeneric.h"

namespace Neon::domain::internal::experimental::staggeredGrid::details {


template <typename BuildingBlockGridT>
struct NodePartitionIndexSpace
{
   public:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Cell = typename BuildingBlockGridT::Cell;
        using PartitionIndexSpace = typename BuildingBlockGridT::PartitionIndexSpace;
    };

    using Cell = NodeGeneric<BuildingBlockGridT>;
    static constexpr int SpaceDim = BuildingBlockGridT::PartitionIndexSpace::SpaceDim;

    NodePartitionIndexSpace() = default;
    explicit NodePartitionIndexSpace(const typename BuildingBlocks::PartitionIndexSpace&);

    NEON_CUDA_HOST_DEVICE
    inline auto setAndValidate(Cell&                          cell,
                               const size_t&                  x,
                               [[maybe_unused]] const size_t& y,
                               [[maybe_unused]] const size_t& z)
        const
        -> bool;


   private:
    typename BuildingBlocks::PartitionIndexSpace mBuildingBlocksPIdxSpace;
};

}  // namespace Neon::domain::internal::experimental::staggeredGrid::details
