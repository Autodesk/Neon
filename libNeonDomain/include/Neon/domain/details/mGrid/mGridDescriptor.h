#pragma once

#include <assert.h>
#include <tuple>

namespace Neon {

/** 
 * @brief mGrid descriptor that defines the depth of the grid levels and how each level is partitioned. 
 * Each level is defined by its refinement factor e.g., block data structure (i.e., single depth grid) with 8^3 blocks 
 * can be described as mGridDescriptor<3>, a multi-resolution grid with depth of 3 that is partitioned into
 * 1) 0-level (finest) as 8^3 voxels  
 * 2) 1-level as 16^3 blocks 
 * 3) 2-level (coarsest) as 32^3 blocks
 * can be described as bGridDescriptor<3,4,5>
 * The refinement factor have to be of power of 2 and thus it is defined by its log2
 * There are two ways to define the depth of the tree. 
 * 1) At compile time using the variadic template parameter Log2RefFactor from which we can infer the depth as the sizeof this parameter
 * 2) At run time using the constructor that takes depth as an input in which case all levels of the tree will have the same branching factor 
 * thus, the sizeof Log2RefFactor should be 1 
*/
template <int... Log2RefFactor>
struct mGridDescriptor
{
    /**
     * @brief Constructor that defines the depth using the variadic template parameter Log2RefFactor 
    */
    mGridDescriptor()
        : m_depth(0) {}

    /**
     * @brief Constructor that define the depth of the tree at run time. It is only allowed when sizeof Log2RefFactor is 1
     * @param depth 
    */
    mGridDescriptor(int depth)
        : m_depth(depth)
    {
        assert(sizeof...(Log2RefFactor) == 1);
    }

    /**
     * get the depth of the tree/grid i.e., how many levels 
    */
    inline int getDepth() const
    {
        constexpr std::size_t depth = sizeof...(Log2RefFactor);
        static_assert(depth != 0,
                      "mGridDescriptor::getDepth() should be at least be of depth 1 i.e., it should have at least single template parameter!");
        if (depth == 1) {
            return m_depth;
        } else {
            return depth;
        }
    }


    /**
     * get the log2 of the refinement factor of certain level     
     * @param level at which the refinement level is queried 
    */
    int getLog2RefFactor(int level) const
    {
        // Runtime input level should be less than the grid depth
        assert(level < int(getDepth()));

        //case where depth is defined at runtime where all levels have the same
        //branching factor
        constexpr std::size_t depth = sizeof...(Log2RefFactor);
        if (depth == 1) {
            return std::get<0>(std::forward_as_tuple(Log2RefFactor...));
        }

        //case where depth is defined at a compile time
        int counter = 0;
        for (const auto l : {Log2RefFactor...}) {
            if (counter == level) {
                return l;
            }
            counter++;
        }
        return -1;
    }

    /**
     * get the refinement factor (i.e., block size) of certain level     
     * @param level at which the refinement level is queried 
    */
    int getRefFactor(int level) const
    {
        return 1 << getLog2RefFactor(level);
    }


    /**
     * return the spacing at a certain level
    */
    int getSpacing(int level) const
    {
        // Runtime input level should be less than the grid depth
        assert(level < int(getDepth()));

        int ret = 1;
        for (int l = 0; l <= level; ++l) {
            ret *= getRefFactor(l);
        }
        return ret;
    }


    /**
     * Convert a voxel id from its index within its level (local index) to its corresponding virtual/base index          
    */
    Neon::index_3d toBaseIndexSpace(const Neon::index_3d& id, const int level) const
    {
        Neon::index_3d ret = id * getSpacing(level - 1);
        return ret;
    }

    /**
     * Convert a voxel id from its virtual/base index to the index of the block containing it at a given level     
    */
    Neon::index_3d toLevelIndexSpace(const Neon::index_3d& id, const int level) const
    {
        Neon::index_3d ret = id / getSpacing(level - 1);
        return ret;
    }

    /**
     * Given a parent on the base/virtual index space, find the index of the child on the base/virtual index space 
     * @param baseParent parent index on the base/virtual index space 
     * @param parentLevel the level at which the parent resides 
     * @param localChild child local index within the parent block 
    */
    Neon::index_3d parentToChild(const Neon::index_3d& baseParent, const int parentLevel, const Neon::index_3d& localChild) const
    {
        //Child local index should be >=0. The input localChild
        assert(localChild.x >= 0 && localChild.y >= 0 && localChild.z >= 0);

        //Child local index should be less than the refinement factor of the parent. The input localChild
        assert(localChild.x < getRefFactor(parentLevel) &&
               localChild.y < getRefFactor(parentLevel) &&
               localChild.z < getRefFactor(parentLevel));

        Neon::index_3d ret = baseParent + localChild * getSpacing(parentLevel - 1);
        return ret;
    }

    /**
     * Given a child on the base/virtual index space, find its parent index in the parent level  
     * @param baseChild the child index on the base/virtual index space 
     * @param parentLevel the level at which the parent resides
    */
    Neon::index_3d childToParent(const Neon::index_3d& baseChild, const int parentLevel) const
    {
        Neon::index_3d ret = baseChild / getSpacing(parentLevel);
        return ret;
    }

    /**
     * Convert the child local 3d index to 1d index 
     * @param localChild child local index within the parent block 
     * @param parentLevel the level at which the parent resides
    */
    int child1DIndex(const Neon::index_3d& localChild, int parentLevel) const
    {
        const int parentRefFactor = getRefFactor(parentLevel);

        return localChild.x +
               localChild.y * parentRefFactor +
               localChild.z * parentRefFactor * parentRefFactor;
    }

    /**
     * Compute the 1d index of a child assuming that the grids (at the parent and child level) are dense. 
     * @param parentID the parent index (at the parent level)
     * @param parentLevel the level at which the parent resides 
     * @param parenLevelNumBlocks number of blocks of in the grid at the parent level 
     * @param localChild child local index within the parent block
    */
    int flattened1DIndex(const Neon::index_3d& parentID,
                         const int             parentLevel,
                         const Neon::index_3d& parenLevelNumBlocks,
                         const Neon::index_3d& localChild) const
    {
        const int parentRefFactor = getRefFactor(parentLevel);

        int parentBlockID = parentID.x +
                            parentID.y * parenLevelNumBlocks.x +
                            parentID.z * parenLevelNumBlocks.x * parenLevelNumBlocks.y;

        const int childID = child1DIndex(localChild, parentLevel) +
                            parentBlockID * parentRefFactor * parentRefFactor * parentRefFactor;

        return childID;
    }


    /**
     * Given a child index in the base/virtual index space, return its local index within the parent of a given level 
     * @return 
    */
    Neon::int32_3d toLocalIndex(const Neon::index_3d& baseChild, const int parentLevel) const
    {
        //fist lift up the child to the parent level -1
        Neon::index_3d child = toLevelIndexSpace(baseChild, parentLevel);

        const int parentRefFactor = getRefFactor(parentLevel);

        return {child.x % parentRefFactor,
                child.y % parentRefFactor,
                child.z % parentRefFactor};
    }


    /**
     * Given a block in the base/virtual index space, a level at which this block resides, and a direction, 
     * return the neighbor block in this direction on the base/virtual index space 
     * 
     * @return 
    */
    Neon::index_3d neighbourBlock(const Neon::index_3d& baseBlock, const int blockLevel, const Neon::index_3d& dir)
    {
        Neon::index_3d ret = dir * getSpacing(blockLevel - 1) + baseBlock;
        return ret;
    }

   private:
    int m_depth;
};

/**
 * mGrid descriptor similar to NanoVDB default refinement 
*/
static mGridDescriptor<3, 4, 5> sBGridVDBDescr;
}  // namespace Neon