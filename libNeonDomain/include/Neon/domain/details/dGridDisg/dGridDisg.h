#pragma once
#include <assert.h>

#include "Neon/core/core.h"
#include "Neon/core/types/DataUse.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/BlockConfig.h"
#include "Neon/set/Containter.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/MemoryOptions.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/domain/aGrid.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/GridConcept.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"

#include "Neon/domain/tools/GridTransformer.h"
#include "Neon/domain/tools/SpanTable.h"

#include "Neon/domain/details/eGrid/eGrid.h"
#include "Neon/domain/patterns/PatternScalar.h"

#include "dPartitionDisg.h"
#include "dSpanDisg.h"

namespace Neon::domain::details::dissagragated::dGrid {

namespace details {
struct Transformation
{
    using FoundationGrid = Neon::domain::details::dGrid::dGrid;
    using Idx = dIndex;
    using Span = dSpan;
    template <typename T, int C>
    using Partition = dPartition<T, C>;

    static constexpr Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport = Neon::set::internal::ContainerAPI::DataViewSupport::on;
    static constexpr Neon::set::details::ExecutionThreadSpan            executionThreadSpan = FoundationGrid::executionThreadSpan;
    using ExecutionThreadSpanIndexType = int32_t;

    static auto getDefaultBlock(FoundationGrid& foundationGrid) -> Neon::index_3d const&
    {
        return foundationGrid.getDefaultBlock();
    }

    static auto initSpan(FoundationGrid& foundationGrid, Neon::domain::tool::SpanTable<Span>& spanTable) -> void
    {
        spanTable.forEachConfiguration([&](Neon::Execution execution,
                                           Neon::SetIdx    setIdx,
                                           Neon::DataView  dw,
                                           Span&           span) {
            span.helpInit(foundationGrid.getSpan(execution, setIdx, dw));
        });
    }

    static auto initLaunchParameters(FoundationGrid&       foundationGrid,
                                     Neon::DataView        dataView,
                                     const Neon::index_3d& blockSize,
                                     const size_t&         shareMem) -> Neon::set::LaunchParameters
    {
        return foundationGrid.getLaunchParameters(dataView, blockSize, shareMem);
    }

    //    static auto helpGetGridIdx(FoundationGrid&,
    //                               Neon::SetIdx const&,
    //                               FoundationGrid::Idx const& fgIdx)
    //        -> dGridSoATransformation::Idx
    //    {
    //        dGridSoATransformation::Idx tgIdx = fgIdx;
    //        return tgIdx;
    //    }

    template <typename T, int C>
    static auto initFieldPartition(FoundationGrid::Field<T, C>&                         foundationField,
                                   Neon::domain::tool::PartitionTable<Partition<T, C>>& partitionTable) -> void
    {
        partitionTable.forEachConfiguration(
            [&](Neon::Execution  execution,
                Neon::SetIdx     setIdx,
                Neon::DataView   dw,
                Partition<T, C>& partition) {
                auto& foundationPartition = foundationField.getPartition(execution, setIdx, dw);
                partition = Partition<T, C>(foundationPartition);
            });
    }
};

}  // namespace details
using dGridSoA = Neon::domain::tool::GridTransformer<details::Transformation>::Grid;

}  // namespace Neon::domain::details::dGridSoA
