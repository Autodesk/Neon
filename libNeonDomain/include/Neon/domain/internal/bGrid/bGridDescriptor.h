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
     * get the log2 of the refinement factor of certain level     
     * @param level at which the refinement level is queried 
    */
    int getLevelLog2RefFactor(int level) const
    {
        if (level >= getDepth()) {
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
};

/**
 * Default bGrid descriptor that defines a grid of single depth partitioned by 8^3 blocks i.e., block data structure 
*/
static bGridDescriptor<3> sBGridDefaultDescriptor;

/**
 * bGrid descriptor similar to NanoVDB default refinement 
*/
static bGridDescriptor<3, 4, 5> sBGridVDBDescriptor;

}  // namespace Neon::domain::internal::bGrid