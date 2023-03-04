#include <tuple>
#include <vector>

namespace Neon::domain {

/**
 * @brief mGrid descriptor that defines the depth of the grid levels and how each level is partitioned. 
*/
struct mGridDescriptor
{
    /**     
     * Each level is defined by its refinement factor e.g., block data structure (i.e., a single-depth grid) with 8^3 blocks 
     * can be described using log 2 of the refinement factor of its blocks which is 3.
     * A multi-resolution grid with depth of 3 that is partitioned into
     * 1) 0-level (finest) as 8^3 voxels  
     * 2) 1-level as 16^3 blocks 
     * 3) 2-level (coarsest) as 32^3 blocks
     * can be described as mGridDescriptor({3,4,5})
     * The refinement factor have to be of power of 2 and thus it is defined by its log2
     * @param levels as described above defaulted to 3 levels octree 
    */
    mGridDescriptor(std::initializer_list<int> log2RefFactors)
        : mLog2RefFactors(log2RefFactors), mSpacing(log2RefFactors)
    {
        computeSpacing();
    }

    /**
     * This constructor can be use for the default mGrid descriptor that defines a grid of single depth partitioned 
     * by 8^3 blocks i.e., block data structure 
    */
    mGridDescriptor()
        : mLog2RefFactors({3}), mSpacing({2 * 2 * 2})
    {
    }


    /**
     * get the depth of the tree/grid i.e., how many levels      
    */
    int getDepth() const
    {
        return int(mLog2RefFactors.size());
    }


    /**
     * get the log2 of the refinement factor of certain level     
     * @param level at which the refinement level is queried 
    */
    int getLog2RefFactor(int level) const
    {
        if (level >= int(getDepth())) {
            NeonException ex("mGridDescriptor::getLog2RefFactor()");
            ex << "Runtime input level is greater than the grid depth!";
            NEON_THROW(ex);
        }
        return mLog2RefFactors[level];
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
        if (level < 0) {
            return 1;
        }

        if (level >= int(getDepth())) {
            NeonException ex("mGridDescriptor::getSpacing()");
            ex << "Runtime input level is greater than the grid depth!";
            NEON_THROW(ex);
        }

        return mSpacing[level];
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
        if (localChild.x < 0 || localChild.y < 0 || localChild.z < 0) {
            NeonException ex("mGridDescriptor::parentToChild()");
            ex << "Child local index should be >=0. The input localChild: " << localChild;
            NEON_THROW(ex);
        }

        if (localChild.x >= getRefFactor(parentLevel) ||
            localChild.y >= getRefFactor(parentLevel) ||
            localChild.z >= getRefFactor(parentLevel)) {
            NeonException ex("mGridDescriptor::parentToChild()");
            ex << "Child local index should be less than the refinement factor of the parent. The input localChild: " << localChild;
            NEON_THROW(ex);
        }

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
    void computeSpacing()
    {
        mSpacing.resize(mLog2RefFactors.size());
        int acc = 1;
        for (int l = 0; l < int(getDepth()); ++l) {
            mSpacing[l] = acc * getRefFactor(l);
            acc = mSpacing[l];
        }
    }

    std::vector<int> mLog2RefFactors;
    std::vector<int> mSpacing;
};
}  // namespace Neon::domain