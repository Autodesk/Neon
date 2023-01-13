#pragma once

#include <unordered_map>

#include "Neon/core/core.h"

namespace Neon::domain::tool {

/**
 * This is an has table for 3D discrete points in finite back ground grid.
 * The class uses a std unordered map of integers.
 * The pitch of the 3D discrete point on the background grid is used as key.
 */
template <typename IntegerT,
          typename MetaT>
class PointHashTable
{
   public:
    using Meta = MetaT /**< Information stored for each point by the hash table */;
    using Integer = IntegerT /**< Type for each discrete point coordinate */;
    using Point = Neon::Integer_3d<Integer> /**< Point type */;
    PointHashTable();

    /**
     * The construct requires the dimension of the grid.
     * @param bbox
     */
    explicit PointHashTable(const Point& bbox);

    /**
     * Retrieve meta data associated with the point.
     * If the point is not resent an exception is thorwn
     * @return
     */
    auto getMetadata(Point const&) const
        -> Meta const*;

    /**
     * Retrieve meta data associated with the point.
     * If the point is not resent an exception is thorwn
     * @return
     */
     auto getMetadata(Point const&)
        -> Meta*;

    /**
     * Adding a point in the hash table
     */
    auto addPoint(Point const&,
                  Meta const&)
        -> void;

    /**
     * Execute a function for each element in the hash table
     */
    template <typename UserLambda>
    auto forEach(const UserLambda&);

    /**
     * get the current size of the map
    */
    auto size() const -> size_t;

   private:
    using Key = size_t;

    /**
     * Get the key for a 3D point
     * @return
     */
    auto helpGetKey(Point const&)
        -> Key;

    /**
     * Get a 3D point from its key
     * @return
     */
    auto helpGetPoint(Key const&)
        -> Point;

    std::unordered_map<Key, Meta> mMap;
    Point                         mBBox;
};

}  // namespace Neon::domain::tool

#include "Neon/domain/tools/PointHashTable_imp.h"