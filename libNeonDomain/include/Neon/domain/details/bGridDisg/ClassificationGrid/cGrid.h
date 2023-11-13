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
#include "./ClassSelector.h"
#include "./cSpan.h"

namespace Neon::domain::details::disaggregated::bGrid {

namespace details {

template <typename SBlock, int classSelector>
struct GridTransformation_cGrid
{
    using FoundationGrid = Neon::domain::details::disaggregated::bGrid::bGrid<SBlock>;

    template <typename T, int C>
    using Partition = Neon::domain::details::disaggregated::bGrid::bPartition<T, C, SBlock>;
    using Span = Neon::domain::details::disaggregated::bGrid::cGrid::cSpan<SBlock, FoundationGrid, classSelector>;
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
            typename FoundationGrid::Span const&     foundationSpan = foundationGrid.getSpan(execution, setIdx, dataViewOfTheTableEntry);
            Neon::domain::tool::Partitioner1D const& foundationPartitioner1D = foundationGrid.helpGetPartitioner1D(execution, setIdx, dw);
            auto const&                              spanLayout = foundationPartitioner1D.getSpanLayout();
            typename Idx::DataBlockCount             iCountVirtualSingleClass;
            typename Idx::DataBlockCount             iAndIupCountVirtualSingleClass;
            typename Idx::DataBlockCount             internalClassFirstMemoryOffset;
            typename Idx::DataBlockCount             bUpClassFirstMemoryOffset;
            typename Idx::DataBlockCount             bDwClassFirstMemoryOffset;

            bool skipInternal = dataViewOfTheTableEntry == Neon::DataView::BOUNDARY;

            if constexpr (classSelector == cGrid::ClassSelector::alpha) {
                auto const iCountVirtualAlpha = skipInternal ? 0 : spanLayout.getBoundsInternal(setIdx, Neon::domain::tool::partitioning::ByDomain::bulk).count;
                auto const iAndIupCountVirtualAlpha = iCountVirtualSingleClass +
                                                      spanLayout.getBoundsBoundary(setIdx,
                                                                                   Neon::domain::tool::partitioning::ByDirection::up,
                                                                                   Neon::domain::tool::partitioning::ByDomain::bulk)
                                                          .count;

                iCountVirtualSingleClass = iCountVirtualAlpha;
                iAndIupCountVirtualSingleClass = iAndIupCountVirtualAlpha;

                internalClassFirstMemoryOffset = 0;
                bUpClassFirstMemoryOffset = spanLayout.getBoundsInternal(setIdx,
                                                                         Neon::domain::tool::partitioning::ByDomain::bc)
                                                .count;
                bDwClassFirstMemoryOffset = bUpClassFirstMemoryOffset +
                                            spanLayout.getBoundsBoundary(setIdx,
                                                                         Neon::domain::tool::partitioning::ByDirection::up,
                                                                         Neon::domain::tool::partitioning::ByDomain::bc)
                                                .count;
            } else {
                auto const iCountVirtualBeta = skipInternal
                                                   ? 0
                                                   : spanLayout.getBoundsInternal(setIdx, Neon::domain::tool::partitioning::ByDomain::bc).count;

                auto const iAndIupCountVirtualBeta = iCountVirtualSingleClass +
                                                     spanLayout.getBoundsBoundary(setIdx,
                                                                                  Neon::domain::tool::partitioning::ByDirection::up,
                                                                                  Neon::domain::tool::partitioning::ByDomain::bc)
                                                         .count;

                iCountVirtualSingleClass = iCountVirtualBeta;
                iAndIupCountVirtualSingleClass = iAndIupCountVirtualBeta;

                internalClassFirstMemoryOffset = spanLayout.getBoundsInternal(setIdx,
                                                                              Neon::domain::tool::partitioning::ByDomain::bulk)
                                                     .count;

                bUpClassFirstMemoryOffset = internalClassFirstMemoryOffset +
                                            spanLayout.getBoundsBoundary(setIdx,
                                                                         Neon::domain::tool::partitioning::ByDirection::up,
                                                                         Neon::domain::tool::partitioning::ByDomain::bulk)
                                                .count;
                bDwClassFirstMemoryOffset = bUpClassFirstMemoryOffset +
                                            spanLayout.getBoundsBoundary(setIdx,
                                                                         Neon::domain::tool::partitioning::ByDirection::down,
                                                                         Neon::domain::tool::partitioning::ByDomain::bulk)
                                                .count;
            }

            span.init(foundationSpan,
                      iCountVirtualSingleClass,
                      iAndIupCountVirtualSingleClass,
                      internalClassFirstMemoryOffset,
                      bUpClassFirstMemoryOffset,
                      bDwClassFirstMemoryOffset,
                      dataViewOfTheTableEntry);
        });
    }

    static auto initLaunchParameters(FoundationGrid&       foundationGrid,
                                     Neon::DataView        dataView,
                                     const Neon::index_3d& blockSize,
                                     const size_t&         shareMem) -> Neon::set::LaunchParameters
    {
        Neon::set::LaunchParameters launchParameters = foundationGrid.initLaunchParameters(dataView, blockSize, shareMem);

        launchParameters.forEachSeq([&](Neon::SetIdx setIdx, Neon::sys::GpuLaunchInfo& launchParameter) {
            Neon::domain::tool::Partitioner1D const& foundationPartitioner1D = foundationGrid.helpGetPartitioner1D(Neon::Execution::host,
                                                                                                                   setIdx,
                                                                                                                   DataView::STANDARD);
            auto const&                              spanLayout = foundationPartitioner1D.getSpanLayout();
            int                                      nBlocks;

            Neon::domain::tool::partitioning::ByDomain byDomain = classSelector == cGrid::ClassSelector::alpha
                                                                      ? Neon::domain::tool::partitioning::ByDomain::bulk
                                                                      : Neon::domain::tool::partitioning::ByDomain::bc;

            int countInternal = spanLayout.getBoundsInternal(setIdx, byDomain).count;
            int countBcUp = spanLayout.getBoundsBoundary(setIdx,
                                                         Neon::domain::tool::partitioning::ByDirection::up,
                                                         byDomain)
                                .count;
            int countBcDw = spanLayout.getBoundsBoundary(setIdx,
                                                         Neon::domain::tool::partitioning::ByDirection::down,
                                                         byDomain)
                                .count;

            switch (dataView) {
                case Neon::DataView::INTERNAL:
                    nBlocks = countInternal;
                    break;
                case Neon::DataView::BOUNDARY:
                    nBlocks = countBcUp + countBcDw;
                    break;
                case Neon::DataView::STANDARD:
                    nBlocks = countInternal +
                              countBcUp +
                              countBcDw;
                    break;
                default:
                    throw Neon::NeonException("Unknown data view");
            }

            launchParameter.set(Neon::sys::GpuLaunchInfo::mode_e::cudaGridMode,
                                nBlocks,
                                SBlock::memBlockSize3D.template newType<int32_t>(), shareMem);
        });
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
using ClassViewGrid = Neon::domain::tool::GridTransformer<details::GridTransformation>::Grid;

}  // namespace details

}  // namespace Neon::domain::details::disaggregated::bGrid
