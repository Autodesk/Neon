#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/dependency/Token.h"

namespace Neon::set::internal {


auto ContainerAPI::
    addToken(Neon::set::dataDependency::Token& dataParsing)
        -> void
{
    mParsed.push_back(dataParsing);
}

auto ContainerAPI::
    getName() const
    -> const std::string&
{
    return mName;
}

auto ContainerAPI::
    getTokens()
        const -> const std::vector<Neon::set::dataDependency::Token>&
{
    return mParsed;
}

auto ContainerAPI::
    getTokensRef()
        -> std::vector<Neon::set::dataDependency::Token>&
{
    return mParsed;
}

auto ContainerAPI::
    setName(const std::string& name)
        -> void
{
    mName = name;
}

auto ContainerAPI::
    setLaunchParameters(Neon::DataView dw)
        -> Neon::set::LaunchParameters&
{
    return mLaunchParameters[DataViewUtil::toInt(dw)];
}

auto ContainerAPI::
    getLaunchParameters(Neon::DataView dw)
        const -> const Neon::set::LaunchParameters&
{
    return mLaunchParameters[DataViewUtil::toInt(dw)];
}

auto ContainerAPI::
    getDataViewSupport() const -> ContainerAPI::DataViewSupport
{
    return mDataViewSupport;
}

auto ContainerAPI::
    setDataViewSupport(ContainerAPI::DataViewSupport dataViewSupport)
        -> void
{
    mDataViewSupport = dataViewSupport;
}

auto ContainerAPI::
    setContainerPattern(Neon::set::ContainerPatternType patternType)
        -> void
{
    this->mContainerPatternType = patternType;
}

auto ContainerAPI::
    setContainerPattern(const std::vector<Neon::set::dataDependency::Token>& tokens)
        -> void
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

auto ContainerAPI::
    toLog(uint64_t uid)
        -> void
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

auto ContainerAPI::
    getContainerExecutionType()
        const
    -> ContainerExecutionType
{
    return mContainerExecutionType;
}

auto ContainerAPI::
    getContainerOperationType()
        const
    -> ContainerOperationType
{
    return mContainerOperationType;
}

auto ContainerAPI::
    getContainerPatternType()
        const
    -> ContainerPatternType
{
    return mContainerPatternType;
}

auto ContainerAPI::
    setContainerExecutionType(ContainerExecutionType containerType)
        -> void
{
    mContainerExecutionType = containerType;
}

auto ContainerAPI::
    setContainerOperationType(ContainerOperationType containerType)
        -> void
{
    mContainerOperationType = containerType;
}

auto ContainerAPI::
    getGraph()
        const -> const Neon::set::container::Graph&
{
    std::string         description = helpGetNameForError();
    Neon::NeonException exp("ContainerAPI");
    exp << description << " "
        << "getGraph"
        << " is not supported.";
    NEON_THROW(exp);
}

auto ContainerAPI::
    getHostContainer()
        -> std::shared_ptr<ContainerAPI>
{
    std::string         description = helpGetNameForError();
    Neon::NeonException exp("ContainerAPI");
    exp << description << " "
        << "getHostContainer"
        << " is not supported.";
    NEON_THROW(exp);
}

auto ContainerAPI::
    helpGetNameForError()
        const -> std::string
{
    std::stringstream s;
    s << getName()
      << "[" << getContainerExecutionType()
      << " - " << getContainerOperationType()
      << " - " << getContainerPatternType();

    return s.str();
}

auto ContainerAPI::
    getDeviceContainer()
        -> std::shared_ptr<ContainerAPI>
{
    std::string         description = helpGetNameForError();
    Neon::NeonException exp("ContainerAPI");
    exp << description << " "
        << "getDeviceContainer"
        << " is not supported.";
    NEON_THROW(exp);
}

auto ContainerAPI::
    isParsingDataUpdated() -> bool
{
    return mParsingDataUpdated;
}

auto ContainerAPI::
    setParsingDataUpdated(bool status) -> void
{
    mParsingDataUpdated = status;
}

auto ContainerAPI::
    parse()
        -> const std::vector<Neon::set::dataDependency::Token>&
{
    std::string         description = helpGetNameForError();
    Neon::NeonException exp("ContainerAPI");
    exp << description << " "
        << "parse"
        << " is not supported.";
    NEON_THROW(exp);
}

auto ContainerAPI::
    getTransferMode()
        const -> Neon::set::TransferMode
{
    std::string         description = helpGetNameForError();
    Neon::NeonException exp("ContainerAPI");
    exp << description << " "
        << "getTransferMode"
        << " is not supported.";
    NEON_THROW(exp);
}


auto ContainerAPI::
 configureWithScheduling(Neon::set::container::GraphNode& graphNode)
        -> void{

}

}  // namespace Neon::set::internal
