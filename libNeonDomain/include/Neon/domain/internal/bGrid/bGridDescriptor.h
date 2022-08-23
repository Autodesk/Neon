#include <tuple>

namespace Neon::domain::internal::bGrid {

/**
 * bGrid descriptor that defines the depth of the grid levels and how each level is partitioned. 
 * Each level is defined by its refinement factor e.g., block data structure (i.e., single depth grid) with 8^3 blocks 
 * can be described as bGridDescriptor<3>, a multi-resolution grid with depth of 3 that is partitioned into
 * 1) 0-level (finest) as 8^3 voxels  
 * 2) 1-level as 16^3 blocks 
 * 3) 2-level (coarsest) as 32^3 blocks
 * can be described as bGridDescriptor<3,4,5>
 * The refinement factor have to be of power of 2 and thus it is defined by its log2
*/
template <int... Log2RefFactor>
struct bGridDescriptor
{
    /**
     * get the depth of the tree/grid i.e., how many levels      
    */
    constexpr static std::size_t getDepth()
    {
        constexpr std::size_t depth = sizeof...(Log2RefFactor);
        static_assert(depth != 0,
                      "bGridDescriptor::getDepth() should be at least be of depth 1 i.e., it should have at least single template parameter!");
        return depth;
    }

    /**
     * get the log2 of the refinement level of the 0 level of the grid. 
     * This level defines how the fine voxels are grouped 
    */
    constexpr static int get0LevelLog2RefFactor()
    {
        auto&& t = std::forward_as_tuple(Log2RefFactor...);
        return std::get<0>(t);
    }

    /**
     * get the refinement level of the 0 level of the grid i.e., the block size of the leaf
    */
    constexpr static int get0LevelRefFactor()
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
        int counter = 0;
        int ret = 0;
        for (const auto l : {Log2RefFactor...}) {
            ret += l;
            if (counter == level) {
                return ret;
            }
            counter++;
        }
        return -1;
    }

    /**
     * get the product of  refinement factors of all levels (starting with level 0) up to certain level
     * @param level the level at which refinement factors prodcut is desired       
    */
    int getRefFactorRecurse(int level) const
    {
        return 1 << getLog2RefFactorRecurse(level);
    }
};

/**
 * Default bGrid descriptor that defines a grid of single depth partitioned by 8^3 blocks i.e., block data structure 
*/
static bGridDescriptor<3> sBGridDefaultDescriptor;

/**
 * Default bGrid descriptor that defines an octree of three levels 
*/
static bGridDescriptor<1, 1, 1> sBGridOctreeDescriptor;

/**
 * bGrid descriptor similar to NanoVDB default refinement 
*/
static bGridDescriptor<3, 4, 5> sBGridVDBDescriptor;

}  // namespace Neon::domain::internal::bGrid