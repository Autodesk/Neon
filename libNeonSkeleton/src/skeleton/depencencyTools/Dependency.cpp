#include "Neon/skeleton/internal/dependencyTools/DataDependency.h"

namespace Neon::skeleton::internal {

DataDependency::DataDependency(Neon::set::container::GraphInfo::NodeUid               t1,
                               Neon::internal::dataDependency::DataDependencyType type,
                               Neon::internal::dataDependency::MdObjUid uid,
                               Neon::set::container::GraphInfo::NodeUid               t0)
{
    mT1 = t1;
    mType = type;
    mDataUid = uid;
    mT0 = t0;
}

bool DataDependency::isValid()
{
    return mType != Neon::internal::dataDependency::DataDependencyType::NONE;
}

auto DataDependency::toString() -> std::string
{
    return std::to_string(mT1) +
           " -> (" + Neon::internal::dataDependency::DataDependencyTypeUtils::toString(mType) +
           " [" + std::to_string(mDataUid) +
           "]) -> " + std::to_string(mT0);
}

auto DataDependency::type() -> Neon::internal::dataDependency::DataDependencyType
{
    return mType;
}

DataDependency DataDependency::getEmpty()
{
    return {};
}

auto DataDependency::t0() -> Neon::set::container::GraphInfo::NodeUid
{
    return mT0;
}

auto DataDependency::t1() -> Neon::set::container::GraphInfo::NodeUid
{
    return mT1;
}

}  // namespace Neon::skeleton::internal