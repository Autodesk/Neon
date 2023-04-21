#pragma once
#include "Neon/core/core.h"

#include "Neon/set/memory/memSet.h"

#include "BlockViewGrid/BlockViewGrid.h"
#include "Neon/domain/aGrid.h"
#include "Neon/domain/details/bGrid/bField.h"
#include "Neon/domain/details/bGrid/bIndex.h"
#include "Neon/domain/details/bGrid/bPartition.h"
#include "Neon/domain/details/bGrid/bSpan.h"
#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/patterns/PatternScalar.h"
#include "Neon/domain/tools/Partitioner1D.h"
#include "Neon/domain/tools/PointHashTable.h"
#include "Neon/domain/tools/SpanTable.h"
#include "Neon/set/Containter.h"
#include "Neon/set/LaunchParametersTable.h"


#include "bField.h"
#include "bPartition.h"
#include "bSpan.h"

namespace Neon::domain::details::bGrid {

template <typename T, int C>
class bField;

class bGrid : public Neon::domain::interface::GridBaseTemplate<bGrid, bIndex>
{
   public:
    using Grid = bGrid;

    template <typename T, int C = 0>
    using Partition = bPartition<T, C>;

    template <typename T, int C = 0>
    using Field = Neon::domain::details::bGrid::bField<T, C>;
    using Span = bSpan;
    using NghIdx = typename Partition<int>::NghIdx;
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<bGrid, bIndex>;

    using Idx = bIndex;
    static constexpr Neon::set::details::ExecutionThreadSpan executionThreadSpan = Neon::set::details::ExecutionThreadSpan::d1b3;
    using ExecutionThreadSpanIndexType = uint32_t;

    using BlockIdx = uint32_t;

    bGrid() = default;
    virtual ~bGrid();

    /**
     * Constructor for the vanilla block data structure with depth of 1
     */
    template <typename ActiveCellLambda>
    bGrid(const Neon::Backend&         backend,
          const Neon::int32_3d&        domainSize,
          const ActiveCellLambda       activeCellLambda,
          const Neon::domain::Stencil& stencil,
          const double_3d&             spacingData = double_3d(1, 1, 1),
          const double_3d&             origin = double_3d(0, 0, 0));


    template <typename ActiveCellLambda>
    bGrid(const Neon::Backend&         backend,
          const Neon::int32_3d&        domainSize,
          const ActiveCellLambda       activeCellLambda,
          const Neon::domain::Stencil& stencil,
          const int                    dataBlockSize,
          const int                    voxelSpacing,
          const double_3d&             spacingData = double_3d(1, 1, 1),
          const double_3d&             origin = double_3d(0, 0, 0));


    auto getProperties(const Neon::index_3d& idx)
        const -> typename GridBaseTemplate::CellProperties final;

    auto isInsideDomain(const Neon::index_3d& idx)
        const -> bool final;

    auto getSetIdx(const Neon::index_3d& idx)
        const -> int32_t final;

    template <typename T, int C = 0>
    auto newField(const std::string          name,
                  int                        cardinality,
                  T                          inactiveValue,
                  Neon::DataUse              dataUse = Neon::DataUse::HOST_DEVICE,
                  const Neon::MemoryOptions& memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    template <typename LoadingLambda>
    auto newContainer(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda,
                      Neon::Execution    execution) const -> Neon::set::Container;

    template <typename LoadingLambda>
    auto newContainer(const std::string& name,
                      LoadingLambda      lambda,
                      Neon::Execution    execution) const -> Neon::set::Container;


    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         sharedMem) const -> Neon::set::LaunchParameters;

    auto getSpan(Neon::Execution execution,
                 SetIdx          setIdx,
                 Neon::DataView  dataView) -> const Span&;

    auto helpGetDataBlockSize() const -> int;
    auto helpGetBlockViewGrid() const -> BlockViewGrid&;
    auto helpGetActiveBitMask() const -> BlockViewGrid::Field<uint64_t, 0>&;
    auto helpGetBlockConnectivity() const -> BlockViewGrid::Field<BlockIdx, 27>&;
    auto helpGetDataBlockOriginField() const -> Neon::aGrid::Field<index_3d, 0>&;
    auto helpGetStencilIdTo3dOffset() const -> Neon::set::MemSet<Neon::int8_3d>&;

    auto helpGetSetIdxAndGridIdx(Neon::index_3d idx) const -> std::tuple<Neon::SetIdx, Idx>;

   private:
    struct Data
    {
        auto init(const Neon::Backend& bk)
        {
            spanTable.init(bk);
            launchParametersTable.init(bk);
        }

        Neon::domain::tool::SpanTable<Span> spanTable /** Span for each data view configurations */;
        Neon::set::LaunchParametersTable launchParametersTable;

        Neon::domain::tool::Partitioner1D partitioner1D;
        Stencil                           stencil;
        Neon::sys::patterns::Engine       reduceEngine;

        Neon::aGrid                     memoryGrid /** memory allocator for fields */;
        Neon::aGrid::Field<index_3d, 0> mDataBlockOriginField;
        Neon::set::MemSet<int8_t>       mStencil3dTo1dOffset;

        BlockViewGrid                      blockViewGrid;
        BlockViewGrid::Field<uint64_t, 0>  activeBitMask;
        BlockViewGrid::Field<BlockIdx, 27> blockConnectivity;

        Neon::set::MemSet<Neon::int8_3d> stencilIdTo3dOffset;

        tool::Partitioner1D::DenseMeta denseMeta;

        int dataBlockSize;
        int voxelSpacing;

        // number of active voxels in each block
        Neon::set::DataSet<uint64_t> mNumActiveVoxel;


        // Stencil neighbor indices
        Neon::set::MemSet<NghIdx> mStencilNghIndex;
    };
    std::shared_ptr<Data> mData;
};


}  // namespace Neon::domain::details::bGrid

#include "bField_imp.h"
#include "bGrid_imp.h"
