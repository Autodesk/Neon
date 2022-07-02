#include "Neon/set/ContainerTools/ContainerAPI.h"


/**
 * Abstract interface to hide
 */

namespace Neon::set::internal {


auto ContainerAPI::addToken(Neon::set::internal::dependencyTools::DataToken& dataParsing) -> void
{
    mParsed.push_back(dataParsing);
}

/**
 *
 */
auto ContainerAPI::getName() const
    -> const std::string&
{
    return mName;
}

auto ContainerAPI::getTokens() const
    -> const std::vector<Neon::set::internal::dependencyTools::DataToken>&
{
    return mParsed;
}

auto ContainerAPI::getTokenRef()
    -> std::vector<Neon::set::internal::dependencyTools::DataToken>&
{
    return mParsed;
}

auto ContainerAPI::setName(const std::string& name)
    -> void
{
    mName = name;
}

auto ContainerAPI::setLaunchParameters(Neon::DataView dw) -> Neon::set::LaunchParameters&
{
    return mLaunchParameters[DataViewUtil::toInt(dw)];
}

auto ContainerAPI::getLaunchParameters(Neon::DataView dw) const -> const Neon::set::LaunchParameters&
{
    return mLaunchParameters[DataViewUtil::toInt(dw)];
}

auto ContainerAPI::setContainerType(ContainerType containerType) -> void
{
    mContainerType = containerType;
}

auto ContainerAPI::getContainerType() const -> ContainerType
{
    return mContainerType;
}

auto ContainerAPI::getDataViewSupport() -> ContainerAPI::DataViewSupport
{
    return mDataViewSupport;
}

auto ContainerAPI::setDataViewSupport(ContainerAPI::DataViewSupport dataViewSupport) -> void
{
    mDataViewSupport = dataViewSupport;
}

auto ContainerAPI::toLog(uint64_t uid) -> void
{
    std::stringstream listOfTokes;
    for (auto& token : mParsed) {
        if (&token != &mParsed[0]) {
            listOfTokes << " ";
        }
        listOfTokes << token.toString();
    }
    NEON_INFO("Container {}: tokens = [{}]", uid, listOfTokes.str());
}

}  // namespace Neon