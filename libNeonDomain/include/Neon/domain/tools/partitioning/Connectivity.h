#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/tools/partitioning/SpanClassifier.h"
#include "Neon/domain/tools/partitioning/SpanLayout.h"

#include "Neon/domain/aGrid.h"

namespace Neon::domain::tools::partitioning {

template <typename ConnOffset,
          typename IntegerForGlobalIndexing>
class Connectivity
{

    Connectivity() = default;

    Connectivity(
        Neon::Backend const&     backend,
        SpanDecomposition const& spanPartitioner,
        SpanClassifier const&    spanClassifier,
        SpanLayout const&        spanLayout,
        Neon::domain::Stencil&   stencil,
        Neon::MemoryOptions      connMemOption);


   private:
    using Global3dIdx = Neon::Integer_3d<IntegerForGlobalIndexing>;
    Neon::domain::aGrid                        mGrid;
    Neon::domain::aGrid::Field<ConnOffset, 0>  mConnectivity;
    Neon::domain::aGrid::Field<ConnOffset, 0>  mMask;
    Neon::domain::aGrid::Field<Global3dIdx, 0> mGlobal3dIdx;
};

template <typename ConnOffset, typename IntegerForGlobalIndexing>
Connectivity<ConnOffset, IntegerForGlobalIndexing>::Connectivity(
    const Backend&           backend,
    const SpanDecomposition& spanPartitioner,
    const SpanClassifier&    spanClassifier,
    SpanLayout const&        spanLayout,
    Stencil&                 stencil,
    Neon::MemoryOptions      connMemOption)
{
    auto partitionAndGhostCount = backend.devSet().newDataSet<uint64_t>(
        [&](Neon::SetIdx setIdx,
            uint64_t&    val) {
            auto const& bounds = spanLayout.getGhostBoundary(setIdx, ByDirection::down);
            val = bounds.first + bounds.count;
        });

    mGrid = Neon::domain::aGrid(backend, partitionAndGhostCount);
    mConnectivity = mGrid.newField<ConnOffset>("ConnectivityTable",
                               stencil.nPoints(),
                               0,
                               Neon::DataUse::IO_COMPUTE,
                               connMemOption);


};
}  // namespace Neon::domain::tools::partitioning
