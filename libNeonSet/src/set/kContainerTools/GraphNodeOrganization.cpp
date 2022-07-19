#include "Neon/set/ContainerTools/GraphNodeOrganization.h"

namespace Neon::set::container {

GraphNodeOrganization::GraphNodeOrganization()
{
    mUid = notSet;
    mIndex = notSet;
}
GraphNodeOrganization::GraphNodeOrganization(int uid)
{
    mUid = uid;
    mIndex = notSet;
}

auto GraphNodeOrganization::setUid(Uid uid) -> void
{
    mUid = uid;
}
auto GraphNodeOrganization::setIndex(Index index) -> void
{
    mIndex = index;
}

auto GraphNodeOrganization::getUid() const -> Uid
{
    return mUid;
}

auto GraphNodeOrganization::getIndex() const -> Index
{
    return mIndex;
}

}  // namespace Neon::set::container
