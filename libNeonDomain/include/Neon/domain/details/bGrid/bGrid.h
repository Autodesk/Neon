#pragma once
#include "Neon/core/core.h"

#include "Neon/set/memory/memSet.h"

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
#include "BlockViewGrid/BlockViewGrid.h"

namespace Neon::domain::details::bGrid {

template <typename T, int C>
class bField;

class bGrid : public Neon::domain::interface::GridBaseTemplate<bGrid, bIndex>
{
   public:
    using Grid = bGrid;
    using Cell = bIndex;

    template <typename T, int C = 0>
    using Partition = bPartition<T, C>;

    template <typename T, int C = 0>
    using Field = Neon::domain::details::bGrid::bField<T, C>;
    using Span = bSpan;
    using nghIdx_t = typename Partition<int>::nghIdx_t;

    using PartitionIndexSpace = Neon::domain::details::bGrid::bSpan;
    using GridBaseTemplate = Neon::domain::interface::GridBaseTemplate<bGrid, bIndex>;

    bGrid() = default;
    virtual ~bGrid(){};

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


    auto getProperties(const Neon::index_3d& idx) const -> typename GridBaseTemplate::CellProperties final;

    auto isInsideDomain(const Neon::index_3d& idx) const -> bool final;

    auto getSetIdx(const Neon::index_3d& idx) const -> int32_t final;


    template <typename T, int C = 0>
    auto newField(const std::string          name,
                  int                        cardinality,
                  T                          inactiveValue,
                  Neon::DataUse              dataUse = Neon::DataUse::HOST_DEVICE,
                  const Neon::MemoryOptions& memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda) const -> Neon::set::Container;

    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      LoadingLambda      lambda) const -> Neon::set::Container;


    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         sharedMem) const -> Neon::set::LaunchParameters;

    auto getPartitionIndexSpace(Neon::DeviceType dev,
                                SetIdx           setIdx,
                                Neon::DataView   dataView) -> const PartitionIndexSpace&;


    auto getOrigins() const -> const Neon::set::MemSet<Neon::int32_3d>&;
    auto getNeighbourBlocks() const -> const Neon::set::MemSet<uint32_t>&;
    auto getActiveMask() const -> Neon::set::MemSet<uint32_t>&;
    auto getBlockOriginTo1D() const -> Neon::domain::tool::PointHashTable<int32_t, uint32_t>&;

    // for compatibility with other grids that can work on cub and cublas engine
    auto setReduceEngine(Neon::sys::patterns::Engine eng) -> void;

    template <typename T>
    auto newPatternScalar() const -> Neon::template PatternScalar<T>;

    template <typename T>
    auto dot(const std::string&               name,
             Field<T>&                        input1,
             Field<T>&                        input2,
             Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container;

    template <typename T>
    auto norm2(const std::string&               name,
               Field<T>&                        input,
               Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container;


    auto getDimension() const -> const Neon::index_3d;

    auto getNumBlocks() const -> const Neon::set::DataSet<uint64_t>&;
    auto getBlockSize() const -> int;
    auto getVoxelSpacing() const -> int;
    auto getOriginBlock3DIndex(const Neon::int32_3d idx) const -> Neon::int32_3d;
    auto getStencilNghIndex() const -> const Neon::set::MemSet<nghIdx_t>&;

    auto getDataBlockSize() const -> int;

   private:
    struct Data
    {
        Neon::domain::tool::SpanTable<Span> spanTable /** Span for each data view configurations */;
        Neon::domain::tool::SpanTable<int>  elementsPerPartition /** Number of indexes for each partition */;

        Neon::domain::tool::Partitioner1D partitioner1D;
        Stencil                           stencil;
        Neon::sys::patterns::Engine       reduceEngine;
        Neon::aGrid                       memoryGrid /** memory allocator for fields */;

        Neon::set::MemSet<int8_t>       mStencil3dTo1dOffset;
        Neon::aGrid::Field<int32_t, 0>  mConnectivityAField;
        Neon::aGrid::Field<index_3d, 0> mDataBlockToGlobalMappingAField;

        tool::Partitioner1D::DenseMeta denseMeta;

        int dataBlockSize;
        int voxelSpacing;

        // number of active voxels in each block
        Neon::set::DataSet<uint64_t> mNumActiveVoxel;

        // block origin coordinates
        Neon::set::MemSet<Neon::int32_3d> mOrigin;

        // Stencil neighbor indices
        Neon::set::MemSet<nghIdx_t> mStencilNghIndex;


        Neon::set::DataSet<uint64_t> mActiveMaskSize;
        Neon::set::MemSet<uint32_t>  mActiveMask;


        // 1d index of 26 neighbor blocks
        // every block is typically neighbor to 26 other blocks. Here we store the 1d index of these 26 neighbor blocks
        // we could use this 1d index to (for example) index the origin of the neighbor block or its active mask
        // as maybe needed by stencil operations
        // If one of this neighbor blocks does not exist (e.g., not allocated or at the domain border), we store
        // std::numeric_limits<uint32_t>::max() to indicate that there is no neighbor block at this location
        Neon::set::MemSet<uint32_t> mNeighbourBlocks;

        // Partition index space
        // It is an std vector for the three type of data views we have
        std::array<Neon::set::DataSet<PartitionIndexSpace>, 3> mPartitionIndexSpace;

        // number of blocks in each device
        Neon::set::DataSet<uint64_t> mNumBlocks;
    };
    std::shared_ptr<Data> mData;
};

auto bGrid::getDataBlockSize() const -> int
{
    return mData->dataBlockSize;
}

}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bGrid_imp.h"
