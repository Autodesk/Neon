#pragma once

#include "Neon/core/core.h"
#include "Neon/domain/tools/PointHashTable.h"

namespace Neon::domain::tool {

template <typename IntegerT, typename MetaT>
class PointHashTableSet
{
   public:
    using Meta = MetaT;
    using Integer = IntegerT;
    using Point = Neon::Integer_3d<Integer>;

    PointHashTableSet();
    explicit PointHashTableSet(const Neon::domain::interface::GridBase& baseGrid);

    /**
     * Retrieve meta data associated with the point.
     * If the point is not resent an exception is thorwn
     * @return
     */
    auto getMetadata(Point const&,
                     SetIdx& setIdx,
                     DataView&) const
        -> Meta const*;

    auto getMetadata(Point const&,
                     SetIdx& setIdx,
                     DataView&)
        -> Meta*;

    auto addPoint(Point const&,
                  Meta const&,
                  const SetIdx& setIdx,
                  const DataView&)
        -> void;

    template <typename UserLambda>
    auto forEach(const UserLambda&);

    template <typename UserLambda>
    auto forEach(Neon::SetIdx setIdx, DataView dataView, const UserLambda&);

   private:
    auto HelpFromDataViewToLocalNaming(DataView dw) -> int;

    static constexpr int HelpInternal = 0;
    static constexpr int HelpBoundary = 1;
    static constexpr int HelpNumOptions = 2;

    using HashTable = PointHashTable<Integer, Meta>;
    using HashTableSetDw = Neon::set::DataSet<std::array<HashTable, HelpNumOptions>>;

    int            mNumDevices;
    index_3d       mBbox;
    HashTableSetDw mTablesSetDw;
};


}  // namespace Neon::domain::tool

#include "Neon/domain/tools/PointHashTableSet_imp.h"
