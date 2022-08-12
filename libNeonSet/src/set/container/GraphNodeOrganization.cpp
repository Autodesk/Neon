#include "Neon/set/container/graph/GraphData.h"

namespace Neon::set::container {

GraphData::GraphData()
{
    mUid = notSet;
    mIndex = notSet;
}
GraphData::GraphData(int uid)
{
    mUid = uid;
    mIndex = notSet;
}

auto GraphData::setUid(Uid uid) -> void
{
    mUid = uid;
}
auto GraphData::setIndex(Index index) -> void
{
    mIndex = index;
}

auto GraphData::getUid() const -> Uid
{
    return mUid;
}

auto GraphData::getIndex() const -> Index
{
    return mIndex;
}

}  // namespace Neon::set::container
