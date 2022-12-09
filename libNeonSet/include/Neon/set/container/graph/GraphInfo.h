#pragma once

#include <cstdint>
#include "vector"

namespace Neon::set::container {

class GraphInfo
{
   public:
    using NodeUid = uint64_t;
    using NodeIdx = uint64_t;

    constexpr static NodeUid notSet = 0;
    constexpr static NodeUid beginUid = 1;
    constexpr static NodeUid endUid = 2;
    constexpr static NodeUid firstInternal = 3;

    GraphInfo();
    GraphInfo(int uid);

    auto setUid(NodeUid uid) -> void;
    auto setIndex(NodeIdx index) -> void;

    auto getUid() const -> NodeUid;
    auto getIndex() const -> NodeIdx;

   private:
    NodeUid mUid /**< unique identifier for the node.
              * This is different from a Container uid as in a graph the same container can appear more than once.
              * */
        ;
    NodeIdx mIndex /** relative index w.r.t the completed graph. This value may change during the life of the graph */;
};

}  // namespace Neon::set::container
