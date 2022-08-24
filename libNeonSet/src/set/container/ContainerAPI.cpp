#include "Neon/set/container/ContainerAPI.h"


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

// auto ContainerAPI::clearTokens()
//     -> void
//{
//     mParsed.clear();
// }

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

auto ContainerAPI::getDataViewSupport() const -> ContainerAPI::DataViewSupport
{
    return mDataViewSupport;
}

auto ContainerAPI::setDataViewSupport(ContainerAPI::DataViewSupport dataViewSupport) -> void
{
    mDataViewSupport = dataViewSupport;
}

auto ContainerAPI::setContainerPattern(Neon::set::ContainerPatternType patternType) -> void
{
    this->mContainerPatternType = patternType;
}

auto ContainerAPI::setContainerPattern(const std::vector<Neon::set::internal::dependencyTools::DataToken>& tokens) -> void
{
    Neon::set::ContainerPatternType patternType = Neon::set::ContainerPatternType::map;

    for (const auto& token : tokens) {
        if (token.compute() == Neon::Compute::STENCIL) {
            if (patternType == Neon::set::ContainerPatternType::reduction) {
                NEON_THROW_UNSUPPORTED_OPTION("Mixing reduction and stencil patterns is currently not supported");
            }
            patternType = Neon::set::ContainerPatternType::stencil;
        }
        if (token.compute() == Neon::Compute::REDUCE) {
            if (patternType == Neon::set::ContainerPatternType::stencil) {
                NEON_THROW_UNSUPPORTED_OPTION("Mixing reduction and stencil patterns is currently not supported");
            }
            patternType = Neon::set::ContainerPatternType::reduction;
        }
    }
    this->mContainerPatternType = patternType;
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

auto ContainerAPI::getContainerExecutionType() const -> ContainerExecutionType
{
    return mContainerExecutionType;
}

auto ContainerAPI::getContainerOperationType() const -> ContainerOperationType
{
    return mContainerOperationType;
}

auto ContainerAPI::getContainerPatternType() const -> ContainerPatternType
{
    return mContainerPatternType;
}

auto ContainerAPI::setContainerExecutionType(ContainerExecutionType containerType) -> void
{
    mContainerExecutionType = containerType;
}

auto ContainerAPI::setContainerOperationType(ContainerOperationType containerType) -> void
{
    mContainerOperationType = containerType;
}

auto ContainerAPI::getGraph() -> const Neon::set::container::Graph&
{
    NEON_THROW_UNSUPPORTED_OPERATION();
}


}  // namespace Neon::set::internal
