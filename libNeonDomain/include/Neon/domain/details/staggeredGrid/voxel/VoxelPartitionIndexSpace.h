#pragma once

#include "Neon/core/core.h"
#include "VoxelGeneric.h"

namespace Neon::domain::details::experimental::staggeredGrid::details {


template <typename BuildingBlockGridT>
struct VoxelPartitionIndexSpace
{
   public:
    struct BuildingBlocks
    {
        using Grid = BuildingBlockGridT;
        using Cell = typename BuildingBlockGridT::Cell;
        using PartitionIndexSpace = typename BuildingBlockGridT::PartitionIndexSpace;
    };

    using Cell = VoxelGeneric<BuildingBlockGridT>;
    static constexpr int SpaceDim = BuildingBlockGridT::PartitionIndexSpace::SpaceDim;

    VoxelPartitionIndexSpace() = default;
    explicit VoxelPartitionIndexSpace(const typename BuildingBlocks::PartitionIndexSpace&,
                                      const typename BuildingBlockGridT::template Field<uint8_t, 1>::Partition& mask);

    NEON_CUDA_HOST_DEVICE
    inline auto setAndValidate(Cell&                          cell,
                               const size_t&                  x,
                               [[maybe_unused]] const size_t& y,
                               [[maybe_unused]] const size_t& z)
        const -> bool;

   private:
    typename BuildingBlocks::PartitionIndexSpace                       mBuildingBlocksPIdxSpace;
    typename BuildingBlockGridT::template Field<uint8_t, 1>::Partition mActiveFlag;
};

}  // namespace Neon::domain::details::experimental::staggeredGrid::details
