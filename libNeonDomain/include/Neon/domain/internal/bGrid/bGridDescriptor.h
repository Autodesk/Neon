#include <tuple>
#include <vector>

namespace Neon::domain::internal::bGrid {

/**
 * @brief bGrid descriptor that defines the depth of the grid levels and how each level is partitioned. 
*/
struct bGridDescriptor
{
    /**     
     * Each level is defined by its refinement factor e.g., block data structure (i.e., a single-depth grid) with 8^3 blocks 
     * can be described using log 2 of the refinement factor of its blocks which is 3.
     * A multi-resolution grid with depth of 3 that is partitioned into
     * 1) 0-level (finest) as 8^3 voxels  
     * 2) 1-level as 16^3 blocks 
     * 3) 2-level (coarsest) as 32^3 blocks
     * can be described as bGridDescriptor({3,4,5})
     * The refinement factor have to be of power of 2 and thus it is defined by its log2
     * @param levels as described above defaulted to 3 levels octree 
    */
    bGridDescriptor(std::initializer_list<int> log2RefFactors)
        : mLog2RefFactors(log2RefFactors), mSpacing(log2RefFactors)
    {
        computeSpacing();
    }

    /**
     * This constructor can be use for the default bGrid descriptor that defines a grid of single depth partitioned 
     * by 8^3 blocks i.e., block data structure 
    */
    bGridDescriptor()
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
     * get the log2 of the refinement level of the 0 level of the grid. 
     * This level defines how the fine voxels are grouped 
    */
    int get0LevelLog2RefFactor() const
    {
        return mLog2RefFactors[0];
    }

    /**
     * get the refinement level of the 0 level of the grid i.e., the block size of the leaf
    */
    int get0LevelRefFactor() const
    {
        return 1 << get0LevelLog2RefFactor();
    }

    /**
     * get the log2 of the refinement factor of certain level     
     * @param level at which the refinement level is queried 
    */
    int getLevelLog2RefFactor(int level) const
    {
        if (level >= int(getDepth())) {
            NeonException ex("bGridDescriptor::getLevelLog2RefFactor()");
            ex << "Runtime input level is greater than the grid depth!";
            NEON_THROW(ex);
        }
        return mLog2RefFactors[level];
    }

    /**
     * get the refinement factor (i.e., block size) of certain level     
     * @param level at which the refinement level is queried 
    */
    int getLevelRefFactor(int level) const
    {
        return 1 << getLevelLog2RefFactor(level);
    }

    /**
     * get the sum of log2 refinement factors of all levels (starting with level 0) up to certain level
     * @param level the end of recurse      
    */
    int getLog2RefFactorRecurse(int level) const
    {
        if (level >= int(getDepth())) {
            NeonException ex("bGridDescriptor::getLog2RefFactorRecurse()");
            ex << "Runtime input level is greater than the grid depth!";
            NEON_THROW(ex);
        }

        int sum = 0;
        for (int i = 0; i <= level; ++i) {
            sum += mLog2RefFactors[i];
        }
        return sum;
    }

    /**
     * get the product of  refinement factors of all levels (starting with level 0) up to certain level
     * @param level the level at which refinement factors product is desired       
    */
    int getRefFactorRecurse(int level) const
    {
        if (level < 0) {
            return 1;
        }

        return 1 << getLog2RefFactorRecurse(level);
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
            NeonException ex("bGridDescriptor::getSpacing()");
            ex << "Runtime input level is greater than the grid depth!";
            NEON_THROW(ex);
        }

        return mSpacing[level];
    }


    /**
     * convert a voxel id from its index within its level (local index) to its corresponding virtual/base index     
    */
    Neon::index_3d toBaseIndexSpace(const Neon::index_3d id, int level) const
    {
        Neon::index_3d ret = id * mSpacing[level];
        return ret;
    }

   private:
    void computeSpacing()
    {
        mSpacing.resize(mLog2RefFactors.size());
        int acc = 1;
        for (int l = 0; l < int(getDepth()); ++l) {
            mSpacing[l] = acc * getLevelRefFactor(l);
            acc = mSpacing[l];
        }
    }

    std::vector<int> mLog2RefFactors;
    std::vector<int> mSpacing;
};
}  // namespace Neon::domain::internal::bGrid