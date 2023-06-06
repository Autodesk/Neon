#pragma once

#include "Neon/set/Containter.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/patterns/PatternScalar.h"
#include "Neon/domain/tools/SpanTable.h"

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
namespace Neon::domain::tool::details {

template <typename T,
          int card,
          typename GridTransformation>
class tField;

template <typename GridTransformation>
class tGrid : public Neon::domain::interface::GridBaseTemplate<tGrid<GridTransformation>,
                                                               typename GridTransformation::FoundationGrid::Cell>
{
   public:
    template <class T, int card>
    using Field = tField<T, card, GridTransformation>;
    template <class T, int card>
    using Partition = typename GridTransformation::template Partition<T, card>;
    using Span = typename GridTransformation::Span;

    static constexpr Neon::set::details::ExecutionThreadSpan executionThreadSpan = GridTransformation::executionThreadSpan;
    using ExecutionThreadSpanIndexType = typename GridTransformation::ExecutionThreadSpanIndexType;
    using Idx = typename GridTransformation::Idx;

   private:
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<tGrid<GridTransformation>,
                                                                       typename GridTransformation::FoundationGrid::Idx>;

    using FoundationGrid = typename GridTransformation::FoundationGrid;

   public:
    tGrid();
    virtual ~tGrid();
    explicit tGrid(FoundationGrid& foundationGrid);
    tGrid(const tGrid& other);                 // copy constructor
    tGrid(tGrid&& other) noexcept;             // move constructor
    tGrid& operator=(const tGrid& other);      // copy assignment
    tGrid& operator=(tGrid&& other) noexcept;  // move assignment

    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         shareMem) const
        -> Neon::set::LaunchParameters;

    auto getSpan(Neon::Execution execution,
                 SetIdx          setIdx,
                 Neon::DataView  dataView)
        -> const Span&;

    template <typename T, int C = 0>
    auto newField(const std::string&  fieldUserName,
                  int                 cardinality,
                  T                   inactiveValue,
                  Neon::DataUse       dataUse = Neon::DataUse::HOST_DEVICE,
                  Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    template <Neon::Execution execution = Neon::Execution::device,
              typename LoadingLambda = void*>
    auto newContainer(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda) const
        -> Neon::set::Container;

    template <Neon::Execution execution = Neon::Execution::device,
              typename LoadingLambda = void*>
    auto newContainer(const std::string& name,
                      LoadingLambda      lambda)
        const
        -> Neon::set::Container;

    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;

    auto getProperties(const Neon::index_3d& idx) const
        -> typename GridBaseTemplate::CellProperties final;

    auto getSetIdx(const Neon::index_3d& idx) const
        -> int32_t final;

    auto getFoundationGrid() -> GridTransformation&;

    auto helpGetSetIdxAndGridIdx(Neon::index_3d idx) const -> std::tuple<Neon::SetIdx, Idx>;

   private:
    struct Data
    {
        Data() = default;
        explicit Data(Neon::Backend& bk)
        {
            spanTable = Neon::domain::tool::SpanTable<Span>(bk);
        }
        // using IndexSpaceInformation = std::array<Neon::set::DataSet<PartitionIndexSpace>, Neon::DataViewUtil::nConfig>;

        FoundationGrid                      foundationGrid;
        Neon::domain::tool::SpanTable<Span> spanTable;
        // std::array<Neon::set::DataSet<PartitionIndexSpace>> partitionIndexSpaceVec;
    };

    std::shared_ptr<Data> mData;
};

template <typename GridTransformation>
auto tGrid<GridTransformation>::helpGetSetIdxAndGridIdx(Neon::index_3d idx) const
    -> std::tuple<Neon::SetIdx, Idx>
{
    auto [setIdx, gIdx] = mData->foundationGrid.helpGetSetIdxAndGridIdx(idx);
    Idx tIdx;
    if (setIdx.isValid()) {
        tIdx = GridTransformation::helpGetGridIdx(mData->foundationGrid, setIdx, gIdx);
    }
    return {setIdx, tIdx};
}

template <typename GridTransformation>
auto tGrid<GridTransformation>::getFoundationGrid() -> GridTransformation&
{
    return mData->foundationGrid;
}


}  // namespace Neon::domain::tool::details