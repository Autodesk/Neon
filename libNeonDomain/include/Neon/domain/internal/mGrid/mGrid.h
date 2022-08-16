#pragma once

#include "Neon/core/core.h"

#include "Neon/domain/interface/GridBaseTemplate.h"

#include "Neon/domain/internal/bGrid/bGrid.h"

#include "Neon/domain/internal/mGrid/mCell.h"
#include "Neon/domain/internal/mGrid/mField.h"
#include "Neon/domain/internal/mGrid/mPartition.h"
#include "Neon/domain/internal/mGrid/mPartitionIndexSpace.h"
#include "Neon/domain/patterns/PatternScalar.h"
#include "Neon/set/Containter.h"


namespace Neon::domain::internal::mGrid {

template <typename T, int C>
class mField;


template <int... Log2RefFactor>
struct mGridDescriptor
{
    /**
     * @brief get the depth of the tree/grid i.e., how many levels      
    */
    constexpr int getDepth() const
    {
        return sizeof()...(Log2RefFactor);
    }

    /**
     * @brief get the log2 of the refinement factor of certain level     
    */
    int getLevelLog2RefFactor(int level) const
    {
        int counter = 0;
        for (const auto l : {Log2RefFactor...}) {
            if (counter == level) {
                return l;
            }
            counter++;
        }
    }
};

class mGrid : public Neon::domain::interface::GridBaseTemplate<mGrid, mCell>
{
   public:
    using Grid = mGrid;
    using Cell = mCell;

    template <typename T, int C = 0>
    using Partition = mPartition<T, C>;

    template <typename T, int C = 0>
    using Field = Neon::domain::internal::mGrid::mField<T, C>;

    using nghIdx_t = typename Partition<int>::nghIdx_t;

    using PartitionIndexSpace = Neon::domain::internal::mGrid::mPartitionIndexSpace;

    mGrid() = default;
    virtual ~mGrid() = default;

    template <typename Descriptor, typename ActiveCellLambda>
    mGrid(const Neon::Backend&         backend,
          const Neon::int32_3d&        domainSize,
          const Descriptor             descriptor,
          const ActiveCellLambda       activeCellLambda,
          const Neon::domain::Stencil& stencil,
          const double_3d&             spacingData = double_3d(1, 1, 1),
          const double_3d&             origin = double_3d(0, 0, 0));


   private:
    struct Data
    {
        std::vector<Neon::domain::internal::bGrid::bGrid> grids;
        std::vector<Neon::set::MemSet_t<uint32_t>>        parentBlocks;
        std::vector<Neon::set::MemSet_t<uint32_t>>        childeMask;
    };
    std::shared_ptr<Data> mData;
};
}  // namespace Neon::domain::internal::mGrid