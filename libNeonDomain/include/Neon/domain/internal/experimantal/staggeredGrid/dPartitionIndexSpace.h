#pragma once
#include "Neon/set/DevSet.h"
#include "dCell.h"
namespace Neon::domain::internal::dGrid {

/**
 * Abstraction that represents the Cell space of a partition
 * This abstraction is used by the neon lambda executor to
 * run a containers on aGrid
 */
class dPartitionIndexSpace
{
   public:
    using Cell = dCell;

    friend class dGrid;

    static constexpr int SpaceDim = 3;

    NEON_CUDA_HOST_DEVICE inline auto setAndValidate(Cell&                          cell,
                                                     const size_t&                  x,
                                                     [[maybe_unused]] const size_t& y,
                                                     [[maybe_unused]] const size_t& z) const
        -> bool;

   private:

    Neon::DataView m_dataView;
    int            m_zHaloRadius;
    int            m_zBoundaryRadius;
    Neon::index_3d m_dim;
};

}  // namespace Neon::domain::dense

#include "dPartitionIndexSpace_imp.h"