#pragma once

#include "Neon/domain/tools/gridTransformer/tField.h"
#include "Neon/domain/tools/gridTransformer/tGrid.h"
#include "Neon/domain/tools/gridTransformer/tGrid_ti.h"
#include "Neon/domain/tools/PartitionTable.h"
#include "Neon/domain/tools/SpanTable.h"

namespace Neon::domain::tool {

/**
 * template <typename FoundationGrid>
 * GridTransformation {
 *      using PartitionIndexSpace
 *      using Partition
 *      using FoundationGrid
 *
 *      getLaunchParameters(Neon::DataView        dataView,
const Neon::index_3d& blockSize,
const size_t&         shareMem)
 * }
 */
template <typename GridTransformation>
class GridTransformer
{
   public:
    template <typename T, int C>
    using Partition = typename GridTransformation::template Partition<T, C>;
    using Span = typename GridTransformation::Span;
    using FoundationGrid = typename GridTransformation::FoundationGrid;

    using Grid = details::tGrid<GridTransformation>;
    template <typename T, int C>
    using Field = details::tField<T, C, GridTransformation>;

    GridTransformer() = default;
    explicit GridTransformer(FoundationGrid& foundationGrid)
    {
        grid = Grid(foundationGrid);
    }
    Grid grid;
};
}  // namespace Neon::domain::tool