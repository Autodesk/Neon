#pragma once
#include "Neon/core/core.h"

#include "Neon/set/memory/memSet.h"

#include "Neon/domain/interface/GridBaseTemplate.h"
#include "Neon/domain/internal/bGrid/bCell.h"
#include "Neon/domain/internal/bGrid/bField.h"
#include "Neon/domain/internal/bGrid/bGridDescriptor.h"
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

    /**
     * General-purpose constructor for multi-resolution grid with variable depth and variable refinement factor at each level
     * Check bGridDescriptor to see how to define the grid 
    */
    bGrid(const Neon::Backend&                                    backend,
          const Neon::int32_3d&                                   domainSize,
          std::vector<std::function<bool(const Neon::index_3d&)>> activeCellLambda,
          const Neon::domain::Stencil&                            stencil,
          const bGridDescriptor                                   descriptor,
          const double_3d&                                        spacingData = double_3d(1, 1, 1),
          const double_3d&                                        origin = double_3d(0, 0, 0));

    auto getProperties(const Neon::index_3d& idx) const
        -> GridBaseTemplate::CellProperties final;

    //TODO
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

    auto getNumBlocksPerPartition(int level) const -> const Neon::set::DataSet<uint64_t>&;
    auto getOrigins(int level) const -> const Neon::set::MemSet_t<Neon::int32_3d>&;
    auto getNeighbourBlocks(int level) const -> const Neon::set::MemSet_t<uint32_t>&;
    auto getActiveMask(int level) const -> const Neon::set::MemSet_t<uint32_t>&;
    auto getBlockOriginTo1D(int level) const -> const Neon::domain::tool::PointHashTable<int32_t, uint32_t>&;
    auto getDescriptor() const -> const std::vector<int>&;

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

    //TODO
    auto getOriginBlock3DIndex(const Neon::int32_3d idx) const -> Neon::int32_3d;

    auto getStencilNghIndex() const -> const Neon::set::MemSet_t<nghIdx_t>&;

    void topologyToVTK(std::string fileName) const;

   private:
    struct Data
    {
        //number of blocks in each device
        //std::vector so we store the number of blocks at each level
        std::vector<Neon::set::DataSet<uint64_t>> mNumBlocks;

        //number of active voxels in each block at each level
        std::vector<Neon::set::DataSet<uint64_t>> mNumActiveVoxel;


        //block origin coordinates
        //std::vector to store the origin of each block at each level
        std::vector<Neon::set::MemSet_t<Neon::int32_3d>> mOrigin;

        //Stencil neighbor indices
        Neon::set::MemSet_t<nghIdx_t> mStencilNghIndex;

        //active voxels bitmask
        //std::vector to store the active mask (and its size) per block per level
        std::vector<Neon::set::DataSet<uint64_t>>  mActiveMaskSize;
        std::vector<Neon::set::MemSet_t<uint32_t>> mActiveMask;


        //1d index of 26 neighbor blocks
        //every block is typically neighbor to 26 other blocks. Here we store the 1d index of these 26 neighbor blocks
        //we could use this 1d index to (for example) index the origin of the neighbor block or its active mask
        //as maybe needed by stencil operations
        //If one of this neighbor blocks does not exist (e.g., not allocated or at the domain border), we store
        //std::numeric_limits<uint32_t>::max() to indicate that there is no neighbor block at this location
        //std::vector to store the neighbor blocks per block per level
        std::vector<Neon::set::MemSet_t<uint32_t>> mNeighbourBlocks;

        //Partition index space
        //It is an std vector for the three type of data views we have
        std::vector<Neon::set::DataSet<PartitionIndexSpace>> mPartitionIndexSpace;

        //Store the block origin as a key and its 1d index as value
        //std::vector to store the map from the block origin to its 1d index per level
        std::vector<Neon::domain::tool::PointHashTable<int32_t, uint32_t>> mBlockOriginTo1D;

        std::vector<int> descriptor;
    };
    std::shared_ptr<Data> mData;
};

}  // namespace Neon::domain::internal::bGrid

#include "Neon/domain/internal/bGrid/bGrid_imp.h"