#include "Neon/set/container/graph/GraphInfo.h"

namespace Neon::set::container {

GraphInfo::GraphInfo()
{
    mUid = notSet;
    mIndex = notSet;
}
GraphInfo::GraphInfo(int uid)
{
    mUid = uid;
    mIndex = notSet;
}

auto GraphInfo::setUid(NodeUid uid) -> void
{
    mUid = uid;
}
auto GraphInfo::setIndex(NodeIdx index) -> void
{
    mIndex = index;
}

auto GraphInfo::getUid() const -> NodeUid
{
    return mUid;
}

auto GraphInfo::getIndex() const -> NodeIdx
{
    return mIndex;
}

}  // namespace Neon::set::container
