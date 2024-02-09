#pragma once
#include <cassert>

#include "Neon/core/core.h"
#include "Neon/core/types/DataUse.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/BlockConfig.h"
#include "Neon/set/Containter.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/MemoryOptions.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/GridConcept.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"

#include "Neon/domain/patterns/PatternScalar.h"
#include "Neon/domain/tools/GridTransformer.h"
#include "Neon/domain/tools/SpanTable.h"

#include "../bGrid.h"
#include "./cSpan.h"

namespace Neon::domain::details::disaggregated::bGridMask {

namespace details::cGrid {

template <typename SBlock, int classSelector>
struct GridTransformation_cGrid
{
    using FoundationGrid = Neon::domain::details::disaggregated::bGridMask::bGrid<SBlock>;

    template <typename T, int C>
    using Partition = Neon::domain::details::disaggregated::bGridMask::bPartition<T, C, SBlock>;
    using Span = cSpan<SBlock, classSelector>;
    static constexpr Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport = Neon::set::internal::ContainerAPI::DataViewSupport::on;

    static constexpr Neon::set::details::ExecutionThreadSpan executionThreadSpan = FoundationGrid::executionThreadSpan;
    using ExecutionThreadSpanIndexType = typename FoundationGrid::ExecutionThreadSpanIndexType;
    using Idx = typename FoundationGrid::Idx;

    static auto getDefaultBlock(FoundationGrid& foundationGrid) -> Neon::index_3d const&
    {
        return foundationGrid.getDefaultBlock();
    }

    static auto initSpan(FoundationGrid&                      foundationGrid,
                         Neon::domain::tool::SpanTable<Span>& spanTable) -> void
    {
        spanTable.forEachConfiguration([&](Neon::Execution execution,
                                           Neon::SetIdx    setIdx,
                                           Neon::DataView  dataViewOfTheTableEntry,
                                           Span& NEON_OUT  span) {
            typename FoundationGrid::Span const& foundationSpan = foundationGrid.getSpan(execution, setIdx, dataViewOfTheTableEntry);
            span = cSpan<SBlock, classSelector>(foundationSpan.mFirstDataBlockOffset,
                                                foundationSpan.mActiveMask,
                                                foundationSpan.mDataView,
                                                foundationGrid.helpGetClassField().getPartition(execution, setIdx, dataViewOfTheTableEntry).mem());
        });
    }

    static auto initLaunchParameters(FoundationGrid&       foundationGrid,
                                     Neon::DataView        dataView,
                                     const Neon::index_3d& blockSize,
                                     const size_t&         shareMem) -> Neon::set::LaunchParameters
    {
        Neon::set::LaunchParameters launchParameters = foundationGrid.getLaunchParameters(dataView, blockSize, shareMem);
        return launchParameters;
    }

    static auto helpGetGridIdx(FoundationGrid&,
                               Neon::SetIdx const&,
                               typename FoundationGrid::Idx const& fgIdx)
        -> GridTransformation::Idx
    {
        GridTransformation::Idx tgIdx = fgIdx;
        return tgIdx;
    }

    template <typename T, int C>
    static auto initFieldPartition(typename FoundationGrid::template Field<T, C>&       foundationField,
                                   Neon::domain::tool::PartitionTable<Partition<T, C>>& partitionTable) -> void
    {
        partitionTable.forEachConfiguration(
            [&](Neon::Execution  execution,
                Neon::SetIdx     setIdx,
                Neon::DataView   dw,
                Partition<T, C>& partition) {
                auto& foundationPartition = foundationField.getPartition(execution, setIdx, dw);
                partition = foundationPartition;
            });
    }
};

template <typename SBlock, int classSelector>
using cGrid = typename Neon::domain::tool::GridTransformer<GridTransformation_cGrid<SBlock, classSelector>>::Grid;

}  // namespace details::cGrid
}  // namespace Neon::domain::details::disaggregated::bGridMask
