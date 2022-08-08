#pragma once

#include <cstdint>
#include "vector"

namespace Neon::set::container {

class GraphData
{
   public:
    using Uid = uint32_t;
    using Index = uint32_t;

    constexpr static Uid notSet = 0;
    constexpr static Uid beginUid = 1;
    constexpr static Uid endUid = 2;
    constexpr static Uid firstInternal = 3;

    GraphData();
    GraphData(int uid);

    auto setUid(Uid uid) -> void;
    auto setIndex(Index index) -> void;

    auto getUid() const -> Uid;
    auto getIndex() const -> Index;

   private:
    Uid   mUid /** unique identifier for the node */;
    Index mIndex /** relative index w.r.t the completed graph. This value may change during the life of the graph */;
};

}  // namespace Neon::set::container
