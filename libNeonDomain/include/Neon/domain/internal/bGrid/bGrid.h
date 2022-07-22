#pragma once
#include "Neon/core/core.h"

#include "Neon/set/memory/memSet.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/internal/bGrid/bCell.h"
#include "Neon/domain/internal/bGrid/bField.h"
#include "Neon/domain/internal/bGrid/bPartition.h"
#include "Neon/domain/internal/bGrid/bPartitionIndexSpace.h"
#include "Neon/domain/patterns/PatternScalar.h"
#include "Neon/domain/tools/PointHashTable.h"
#include "Neon/set/Containter.h"

namespace Neon::domain::internal::bGrid {

template <typename T, int C>
class bField;

class bGrid : public Neon::domain::interface::GridBaseTemplate<bGrid, bCell>
{
   public:
    using Grid = bGrid;
    using Cell = bCell;

    template <typename T, int C = 0>
    using Partition = bPartition<T, C>;

    template <typename T, int C = 0>
    using Field = Neon::domain::internal::bGrid::bField<T, C>;

    using nghIdx_t = typename Partition<int>::nghIdx_t;

    using PartitionIndexSpace = Neon::domain::internal::bGrid::bPartitionIndexSpace;

    bGrid() = default;
    virtual ~bGrid() = default;

    template <typename ActiveCellLambda>
    bGrid(const Neon::Backend&         backend,
          const Neon::int32_3d&        domainSize,
          const ActiveCellLambda       activeCellLambda,
          const Neon::domain::Stencil& stencil,
          const double_3d&             spacingData = double_3d(1, 1, 1),
          const double_3d&             origin = double_3d(0, 0, 0));

    auto getProperties(const Neon::index_3d& idx) const
        -> GridBaseTemplate::CellProperties final;

    auto isInsideDomain(const Neon::index_3d& idx) const
        -> bool final;


    template <typename T, int C = 0>
    auto newField(const std::string          name,
                  int                        cardinality,
                  T                          inactiveValue,
                  Neon::DataUse              dataUse = Neon::DataUse::IO_COMPUTE,
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

    auto getNumBlocksPerPartition() const -> const Neon::set::DataSet<uint64_t>&;
    auto getOrigins() const -> const Neon::set::MemSet_t<Neon::int32_3d>&;
    auto getNeighbourBlocks() const -> const Neon::set::MemSet_t<uint32_t>&;
    auto getActiveMask() const -> const Neon::set::MemSet_t<uint32_t>&;
    auto getBlockOriginTo1D() const -> const Neon::domain::tool::PointHashTable<int32_t, uint32_t>&;

    //for compatibility with other grids that can work on cub and cublas engine
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


    auto getKernelConfig(int            streamIdx,
                         Neon::DataView dataView) -> Neon::set::KernelConfig;

    auto getOriginBlock3DIndex(const Neon::int32_3d idx) const -> Neon::int32_3d;

    auto getStencilNghIndex() const -> const Neon::set::MemSet_t<nghIdx_t>&;

   private:
    struct Data
    {
        //number of blocks in each device
        Neon::set::DataSet<uint64_t> mNumBlocks;


        //block origin coordinates
        Neon::set::MemSet_t<Neon::int32_3d> mOrigin;

        //Stencil neighbor indices
        Neon::set::MemSet_t<nghIdx_t> mStencilNghIndex;

        //active voxels bitmask
        Neon::set::DataSet<uint64_t>  mActiveMaskSize;
        Neon::set::MemSet_t<uint32_t> mActiveMask;


        //1d index of 26 neighbor blocks
        //every block is typically neighbor to 26 other blocks. Here we store the 1d index of these 26 neighbor blocks
        //we could use this 1d index to (for example) index the origin of the neighbor block or its active mask
        //as maybe needed by stencil operations
        //If one of this neighbor blocks does not exist (e.g., not allocated or at the domain border), we store
        //std::numeric_limits<uint32_t>::max() to indicate that there is no neighbor block at this location
        Neon::set::MemSet_t<uint32_t> mNeighbourBlocks;

        //Partition index space
        std::vector<Neon::set::DataSet<PartitionIndexSpace>> mPartitionIndexSpace;

        //Store the block origin as a key and its 1d index as value
        Neon::domain::tool::PointHashTable<int32_t, uint32_t> mBlockOriginTo1D;
    };
    std::shared_ptr<Data> mData;
};

}  // namespace Neon::domain::internal::bGrid

#include "Neon/domain/internal/bGrid/bGrid_imp.h"