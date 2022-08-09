#pragma once
#include "Neon/core/core.h"

#include "Neon/set/ContainerTools/AnchorContainer.h"

namespace Neon::set::internal {

AnchorContainer::AnchorContainer(const std::string& name)
{
    setName(name);
    setContainerExecutionType(ContainerExecutionType::none);
    setContainerOperationType(ContainerOperationType::anchor);
    setDataViewSupport(ContainerAPI::DataViewSupport::off);
}


auto AnchorContainer::parse() -> const std::vector<Neon::set::internal::dependencyTools::DataToken>&
{
    return mEmtpy;
}

auto AnchorContainer::getHostContainer() -> std::shared_ptr<ContainerAPI>
{
    NEON_THROW_UNSUPPORTED_OPTION("This Container type can not be decoupled.");
}

auto AnchorContainer::getDeviceContainer() -> std::shared_ptr<ContainerAPI>
{
    NEON_THROW_UNSUPPORTED_OPTION("This Container type can not be decoupled.");
}

auto AnchorContainer::run(int streamIdx , Neon::DataView dataView ) -> void
{
}

auto AnchorContainer::run(Neon::SetIdx setIdx, int streamIdx, Neon::DataView dataView) -> void
{
}

}  // namespace Neon::set::internal
