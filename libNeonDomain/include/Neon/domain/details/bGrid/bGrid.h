#pragma once
#include "Neon/core/core.h"

#include "Neon/domain/aGrid.h"
#include "Neon/domain/details/bGrid/BlockView.h"
#include "Neon/domain/details/bGrid/StaticBlock.h"
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
#include "Neon/set/memory/memSet.h"

#include "bField.h"
#include "bPartition.h"
#include "bSpan.h"

namespace Neon::domain::details::bGrid {


template <typename T, int C, typename SBlock>
class bField;

template <typename SBlock>
class bGrid : public Neon::domain::interface::GridBaseTemplate<bGrid<SBlock>,
                                                               bIndex<SBlock>>
{
   public:
    using Grid = bGrid<SBlock>;
    template <typename T, int C = 0>
    using Partition = bPartition<T, C, SBlock>;
    template <typename T, int C = 0>
    using Field = Neon::domain::details::bGrid::bField<T, C, SBlock>;

    using Span = bSpan<SBlock>;
    using NghIdx = typename Partition<int>::NghIdx;
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<Grid, bIndex<SBlock>>;

    using Idx = bIndex<SBlock>;
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


    /**
     * Constructor for bGrid. This constructor should be directly used only by mGrid
     */
    template <typename ActiveCellLambda>
    bGrid(const Neon::Backend&         backend /**< Neon backend for the computation */,
          const Neon::int32_3d&        domainSize /**< Size of the bounded Cartesian */,
          const ActiveCellLambda       activeCellLambda /**< Function that identify the user domain inside the boxed Cartesian discretization  */,
          const Neon::domain::Stencil& stencil /**< union of tall the stencil that will be used in the computation */,
          const int                    multiResDiscreteIdxSpacing /**< Parameter for the multi-resolution. Index i and index (i+1) may be remapped as i*voxelSpacing  and (i+1)* voxelSpacing.
                                                                   * For a uniform bGrid, i.e outside the context of multi-resolution this parameter is always 1 */
          ,
          const double_3d& spacingData /** Physical spacing between two consecutive data points in the Cartesian domain */,
          const double_3d& origin /** Physical location in space of the origin of the Cartesian discretization */);

    /**
     * Returns some properties for a given cartesian in the Cartesian domain.
     * The provide index my be inside or outside the user defined bounded Cartesian domain
     */
    auto getProperties(const Neon::index_3d& idx)
        const -> typename GridBaseTemplate::CellProperties final;

    /**
     * Returns true if the query 3D point is inside the user domain
     * @param idx
     * @return
     */
    auto isInsideDomain(const Neon::index_3d& idx)
        const -> bool final;

    /**
     * Retrieves the device index that contains the query point
     * @param idx
     * @return
     */
    auto getSetIdx(const Neon::index_3d& idx)
        const -> int32_t final;

    /**
     * Allocates a new field on the grid
     */
    template <typename T, int C = 0>
    auto newField(const std::string   name,
                  int                 cardinality,
                  T                   inactiveValue,
                  Neon::DataUse       dataUse = Neon::DataUse::HOST_DEVICE,
                  Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    /**
     * Allocates a new field on the block view grid
     */
    template <typename T, int C = 0>
    auto newBlockViewField(const std::string   name,
                           int                 cardinality,
                           T                   inactiveValue,
                           Neon::DataUse       dataUse = Neon::DataUse::HOST_DEVICE,
                           Neon::MemoryOptions memoryOptions = Neon::MemoryOptions()) const
        -> BlockView::Field<T, C>;

    /**
     * Allocates a new container to execute some computation in the grid
     */
    template <Neon::Execution execution = Neon::Execution::device,
              typename LoadingLambda = void*>
    auto newContainer(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda) const -> Neon::set::Container;

    /**
     * Allocates a new container to execute some computation in the grid
     */
    template <Neon::Execution execution = Neon::Execution::device,
              typename LoadingLambda = void*>
    auto newContainer(const std::string& name,
                      LoadingLambda      lambda) const -> Neon::set::Container;

    /**
     * Defines a new set of parameter to launch a Container
     */
    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         sharedMem) const -> Neon::set::LaunchParameters;

    /**
     * Retrieve the span associated to the grid w.r.t. some user defined parameters.
     */
    auto getSpan(Neon::Execution execution,
                 SetIdx          setIdx,
                 Neon::DataView  dataView) -> const Span&;

    /**
     * Retrieve the block vew grid internally used.
     * This grid can be leverage to allocate data at the block level.
     */
    auto getBlockViewGrid() const -> BlockView::Grid&;


    /**
     * Retrieve the block vew grid internally used.
     * This grid can be leverage to allocate data at the block level.
     */
    auto getActiveBitMask() const -> BlockView::Field<typename SBlock::BitMask, 1>&;

    /**
     * Helper function to retrieve the discrete index spacing used for the multi-resolution
     */
    template <int dummy = SBlock::isMultiResMode>
    auto helGetMultiResDiscreteIdxSpacing() const -> std::enable_if_t<dummy == 1, int>;


    /**
     * Help function to retrieve the block connectivity as a BlockViewGrid field
     */
    auto helpGetBlockConnectivity() const -> BlockView::Field<BlockIdx, 27>&;

    /**
     * Help function to retrieve the block origin as a BlockViewGrid field
     */
    auto helpGetDataBlockOriginField() const -> Neon::aGrid::Field<index_3d, 0>&;

    /**
     * Help function to retrieve the map that converts a stencil point id to 3d offset
     */
    auto helpGetStencilIdTo3dOffset() const -> Neon::set::MemSet<Neon::int8_3d>&;

    auto helpGetPartitioner1D() -> Neon::domain::tool::Partitioner1D&;

    /**
     * Help function retriev the device and the block index associated to a point in the BlockViewGrid grid
     */
    auto helpGetSetIdxAndGridIdx(Neon::index_3d idx) const -> std::tuple<Neon::SetIdx, Idx>;

    struct Data
    {
        auto init(const Neon::Backend& bk)
        {
            spanTable.init(bk);
            launchParametersTable.init(bk);
        }

        Neon::domain::tool::SpanTable<Span> spanTable /** Span for each data view configurations */;
        Neon::set::LaunchParametersTable    launchParametersTable;

        Neon::domain::tool::Partitioner1D partitioner1D;
        Stencil                           stencil;
        Neon::sys::patterns::Engine       reduceEngine;

        Neon::aGrid                     memoryGrid /** memory allocator for fields */;
        Neon::aGrid::Field<index_3d, 0> mDataBlockOriginField;
        Neon::set::MemSet<int8_t>       mStencil3dTo1dOffset;

        BlockView::Grid                               blockViewGrid;
        BlockView::Field<typename SBlock::BitMask, 1> activeBitField;
        BlockView::Field<BlockIdx, 27>                blockConnectivity;
        Neon::set::MemSet<Neon::int8_3d>              stencilIdTo3dOffset;

        int mMultiResDiscreteIdxSpacing;

        // number of active voxels in each block
        Neon::set::DataSet<uint64_t> mNumActiveVoxel;


        // Stencil neighbor indices
        Neon::set::MemSet<NghIdx> mStencilNghIndex;
    };
    std::shared_ptr<Data> mData;
};
extern template class bGrid<StaticBlock<8, 8, 8>>;
}  // namespace Neon::domain::details::bGrid

#include "bField_imp.h"
#include "bGrid_imp.h"
