#pragma once
#include "Neon/core/core.h"

#include "Neon/set/memory/memSet.h"

#include "Neon/domain/bGrid.h"
#include "Neon/domain/internal/bGrid/bCell.h"
#include "Neon/domain/internal/bGrid/bPartition.h"

#include "Neon/domain/internal/mGrid/mField.h"
#include "Neon/domain/internal/mGrid/mGridDescriptor.h"
#include "Neon/domain/patterns/PatternScalar.h"
#include "Neon/set/Containter.h"


namespace Neon::domain::internal::mGrid {

template <typename T, int C>
class mField;

class mGrid
{
   public:
    using Grid = mGrid;
    using Cell = Neon::domain::internal::bGrid::bCell;

    template <typename T, int C = 0>
    using Partition = Neon::domain::internal::bGrid::bPartition<T, C>;

    template <typename T, int C = 0>
    using Field = Neon::domain::internal::mGrid::mField<T, C>;

    using nghIdx_t = typename Partition<int>::nghIdx_t;

    using PartitionIndexSpace = Neon::domain::internal::bGrid::bPartitionIndexSpace;

    mGrid() = default;
    virtual ~mGrid(){};

    /**
     * General-purpose constructor for multi-resolution grid with variable depth and variable refinement factor at each level
     * Check mGridDescriptor to see how to define the grid 
    */
    mGrid(const Neon::Backend&                                    backend,
          const Neon::int32_3d&                                   domainSize,
          std::vector<std::function<bool(const Neon::index_3d&)>> activeCellLambda,
          const Neon::domain::Stencil&                            stencil,
          const mGridDescriptor                                   descriptor,
          bool                                                    isStrongBalanced = true,
          const double_3d&                                        origin = double_3d(0, 0, 0));


    auto isInsideDomain(const Neon::index_3d& idx, int level) const -> bool;


    template <typename T, int C = 0>
    auto newField(const std::string          name,
                  int                        cardinality,
                  T                          inactiveValue,
                  Neon::DataUse              dataUse = Neon::DataUse::IO_COMPUTE,
                  const Neon::MemoryOptions& memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      int                level,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda) const -> Neon::set::Container;

    template <typename LoadingLambda>
    auto getContainer(const std::string& name,
                      int                level,
                      LoadingLambda      lambda) const -> Neon::set::Container;


    auto getLaunchParameters(Neon::DataView        dataView,
                             const Neon::index_3d& blockSize,
                             const size_t&         sharedMem,
                             int                   level) const -> Neon::set::LaunchParameters;


    auto getNumBlocksPerPartition(int level) const -> const Neon::set::DataSet<uint64_t>&;
    auto getParentsBlockID(int level) const -> const Neon::set::MemSet_t<uint32_t>&;
    auto getParentLocalID(int level) const -> const Neon::set::MemSet_t<Cell::Location>&;
    auto getChildBlockID(int level) const -> const Neon::set::MemSet_t<uint32_t>&;

    //for compatibility with other grids that can work on cub and cublas engine
    auto setReduceEngine(Neon::sys::patterns::Engine eng) -> void;

    /*template <typename T>
    auto newPatternScalar() const -> Neon::template PatternScalar<T>;

    template <typename T>
    auto dot(const std::string&               name,
             Field<T>&                        input1,
             Field<T>&                        input2,
             Neon::template PatternScalar<T>& scalar,
             const int                        level) const -> Neon::set::Container;

    template <typename T>
    auto norm2(const std::string&               name,
               Field<T>&                        input,
               Neon::template PatternScalar<T>& scalar,
               const int                        level) const -> Neon::set::Container;*/

    auto getDimension(int level) const -> const Neon::index_3d;

    auto getDimension() const -> const Neon::index_3d;

    /**
     * Number of blocks is the number blocks at a certain level where each block subsumes 
     * a number of voxels defined by the refinement factor at this level. Note that level-0 is composed of
     * number of blocks each containing number of voxels. In case of using bGrid as a uniform grid, the 
     * total number of voxels can be obtained from getDimension
    */
    auto getNumBlocks(int level) const -> const Neon::index_3d&;
    auto getGrid(int level) const -> const Neon::domain::bGrid&;
    auto getOriginBlock3DIndex(const Neon::int32_3d idx, int level) const -> Neon::int32_3d;
    auto getDescriptor() const -> const mGridDescriptor&;
    auto getRefFactors() const -> const Neon::set::MemSet_t<int>&;
    auto getLevelSpacing() const -> const Neon::set::MemSet_t<int>&;
    void topologyToVTK(std::string fileName, bool filterOverlaps) const;


   private:
    struct Data
    {
        Neon::index_3d domainSize;

        //stores the parent of the block
        std::vector<Neon::set::MemSet_t<uint32_t>> mParentBlockID;

        //Given a block at level L, we store R children block IDs for each block in L where R is the refinement factor
        std::vector<Neon::set::MemSet_t<uint32_t>> mChildBlockID;

        //store the parent local index within its block
        std::vector<Neon::set::MemSet_t<Cell::Location>> mParentLocalID;


        //gird levels refinement factors
        Neon::set::MemSet_t<int> mRefFactors;

        //gird levels spacing
        Neon::set::MemSet_t<int> mSpacing;


        //Total number of blocks in the domain
        //std::vector so we store the number of blocks at each level
        std::vector<Neon::index_3d> mTotalNumBlocks;


        mGridDescriptor mDescriptor{};

        bool mStrongBalanced;

        //bitmask of the active cells at each level and works as if the grid is dense at each level
        std::vector<std::vector<uint32_t>> denseLevelsBitmask;

        //collection of bGrids that make up the multi-resolution grid
        std::vector<Neon::domain::bGrid> grids;

        Neon::Backend backend;
    };
    std::shared_ptr<Data> mData;
};

}  // namespace Neon::domain::internal::mGrid

#include "Neon/domain/internal/mGrid/mGrid_imp.h"