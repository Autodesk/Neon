
#pragma once
#include "Neon/core/core.h"

#include "Neon/set/memory/memSet.h"

#include "Neon/domain/details/bGrid/bGrid.h"

#include "Neon/domain/details/mGrid/mGridDescriptor.h"

#include "Neon/domain/details/bGrid/bIndex.h"
#include "Neon/domain/details/mGrid/mField.h"
#include "Neon/domain/details/mGrid/mPartition.h"

#include "Neon/set/Containter.h"


namespace Neon::domain::details::mGrid {

template <typename T, int C>
class mField;

/**
 * Multi-resolution gird represented as a stack of bGrids i.e., block-sparse data structures. Level 0 represents the root of the grid
 * i.e., the finest level of the gird. mGird can store data (through mField) and operate on each level of the grid
*/
class mGrid
{
   public:
    using Grid = mGrid;
    using InternalGrid = Neon::domain::details::bGrid::bGrid<kStaticBlock>;
    using Idx = typename InternalGrid::Idx;
    using Descriptor = mGridDescriptor<1>;

    template <typename T, int C = 0>
    using Partition = Neon::domain::details::mGrid::mPartition<T, C>;

    template <typename T, int C = 0>
    using Field = Neon::domain::details::mGrid::mField<T, C>;

    using NghIdx = typename Partition<int>::NghIdx;

    using Span = typename InternalGrid::Span;

    template <typename T, int C>
    friend class Neon::domain::details::mGrid::mField;

    mGrid() = default;
    virtual ~mGrid(){};

    /**
     * Main constructor of the grid where the user can define the sparsity pattern for each level of the grid
     * @param backend backend of the grid (CPU, GPU)
     * @param domainSize the size of domain as defined by the finest level of the grid (Level 0)
     * @param activeCellLambda the activation functions the defines the sparsity pattern of each level of the grid 
     * it is an std::vector since the user should define the activation function for each level where 
     * activeCellLambda[L] is the activation function of level L 
     * @param stencil the union of all stencils that will be needed 
     * @param descriptor defines the number of levels in the grid and the branching factor of each level 
     * @param isStrongBalanced if the strong balanced condition should be enforced by the data structure 
     * @param spacingData the size of the voxel 
     * @param origin the origin of the grid 
    */
    mGrid(const Neon::Backend&                                    backend,
          const Neon::int32_3d&                                   domainSize,
          std::vector<std::function<bool(const Neon::index_3d&)>> activeCellLambda,
          const Neon::domain::Stencil&                            stencil,
          const Descriptor                                        descriptor,
          bool                                                    isStrongBalanced = true,
          const double_3d&                                        spacingData = double_3d(1, 1, 1),
          const double_3d&                                        origin = double_3d(0, 0, 0));

    /**
     * Given a voxel and its level, returns if the voxel is inside the domain. The voxel should be 
     * define based on the index space of the finest level (Level 0) 
     * @param idx the voxel 3D index 
     * @param level the level at which we query the voxel      
    */
    auto isInsideDomain(const Neon::index_3d& idx, int level) const -> bool;

    /**
     * Since mGird is internally represented by a stack of grids, this return the grid at certain level 
     * @param level at which the grid is queried      
    */
    auto operator()(int level) -> InternalGrid&;

    /**
     * Since mGird is internally represented by a stack of grids, this return the grid at certain level 
     * @param level at which the grid is queried      
    */
    auto operator()(int level) const -> const InternalGrid&;

    /**
     * Create new field on the multi-resolution grid. Data is allocated at all levels following the sparsity pattern define on the mGird 
     * @tparam T type of the data 
     * @param name meaningful name for the field 
     * @param cardinality the number of components for vector-valued data 
     * @param inactiveValue what to return if the field is queried at a voxel that is not present as defined by the sparsity pattern 
     * @param dataUse 
     * @param memoryOptions 
     * @return 
    */
    template <typename T, int C = 0>
    auto newField(const std::string          name,
                  int                        cardinality,
                  T                          inactiveValue,
                  Neon::DataUse              dataUse = Neon::DataUse::HOST_DEVICE,
                  const Neon::MemoryOptions& memoryOptions = Neon::MemoryOptions()) const
        -> Field<T, C>;

    /**
     * Create a new container to do some work on the grid at a certain level 
     * @tparam LoadingLambda inferred 
     * @param name meaningful name for the containers 
     * @param level at which the work/kernel will be launched 
     * @param blockSize the block size for CUDA kernels 
     * @param sharedMem amount of shared memory in bytes for CUDA kernels 
     * @param lambda the lambda function that will do the computation      
    */
    template <typename LoadingLambda>
    auto newContainer(const std::string& name,
                      int                level,
                      index_3d           blockSize,
                      size_t             sharedMem,
                      LoadingLambda      lambda) const -> Neon::set::Container;

    /**
     * Create a new container to do some work on the grid at a certain level 
     * @tparam LoadingLambda inferred 
     * @param name meaningful name for the containers 
     * @param level at which the work/kernel will be launched      
     * @param lambda the lambda function that will do the computation      
    */
    template <typename LoadingLambda>
    auto newContainer(const std::string& name,
                      int                level,
                      LoadingLambda      lambda) const -> Neon::set::Container;

    auto getParentsBlockID(int level) const -> Neon::set::MemSet<uint32_t>&;
    auto getChildBlockID(int level) const -> const Neon::set::MemSet<uint32_t>&;


    /**
     * define the reduction engine. This is done for compatibility with other grids that can work on cub and cublas engine
     * @param eng the reduction engine which could be CUB or cublas. Only CUB is supported.      
    */
    auto setReduceEngine(Neon::sys::patterns::Engine eng) -> void;

    /**
     * Get the size of the domain at a certain level
     * @param level at which the domain size is queried      
    */
    auto getDimension(int level) const -> const Neon::index_3d;

    /**
     * @brief get the dimension of the domain as defined at the finest level (Level 0) of the grid      
    */
    auto getDimension() const -> const Neon::index_3d;

    /**
     * Number of blocks is the number blocks at a certain level where each block subsumes 
     * a number of voxels defined by the refinement factor at this level. Note that level-0 is composed of
     * number of blocks each containing number of voxels. In case of using bGrid as a uniform grid, the 
     * total number of voxels can be obtained from getDimension
     * @param level at which the number of blocks are queried 
    */
    auto getNumBlocks(int level) const -> const Neon::index_3d&;

    auto getOriginBlock3DIndex(const Neon::int32_3d idx, int level) const -> Neon::int32_3d;
    auto getDescriptor() const -> const Descriptor&;
    auto getRefFactors() const -> const Neon::set::MemSet<int>&;
    auto getLevelSpacing() const -> const Neon::set::MemSet<int>&;
    auto getBackend() const -> const Backend&;
    auto getBackend() -> Backend&;


   private:
    //check if the bitmask is set assuming a dense domain
    auto levelBitMaskIndex(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) const -> std::pair<int, int>;

    auto levelBitMaskIsSet(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) const -> bool;

    //set the bitmask assuming a dense domain
    auto setLevelBitMask(int l, const Neon::index_3d& blockID, const Neon::index_3d& localChild) -> void;

    struct Data
    {
        Neon::index_3d domainSize;

        //stores the parent of the block
        std::vector<Neon::set::MemSet<Idx::DataBlockIdx>> mParentBlockID;

        //Given a block at level L, we store R children block IDs for each block in L where R is the refinement factor
        std::vector<Neon::set::MemSet<Idx::DataBlockIdx>> mChildBlockID;

        //gird levels refinement factors
        Neon::set::MemSet<int> mRefFactors;

        //gird levels spacing
        Neon::set::MemSet<int> mSpacing;


        //Total number of blocks in the domain
        //std::vector so we store the number of blocks at each level
        std::vector<Neon::index_3d> mTotalNumBlocks;


        Descriptor mDescriptor;

        bool mStrongBalanced;

        //bitmask of the active cells at each level and works as if the grid is dense at each level
        std::vector<std::vector<uint32_t>> denseLevelsBitmask;

        //collection of bGrids that make up the multi-resolution grid
        std::vector<InternalGrid> grids;

        Neon::Backend backend;
    };
    std::shared_ptr<Data> mData;
};

}  // namespace Neon::domain::details::mGrid

#include "Neon/domain/details/mGrid/mGrid_imp.h"
