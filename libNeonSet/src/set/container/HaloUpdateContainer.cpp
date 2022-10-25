#include "Neon/core/core.h"

#include "Neon/set/container/HaloUpdateContainer.h"

namespace Neon::set::internal {

HaloUpdateContainer::
    HaloUpdateContainer(const std::string& name)
{
    setName(name);
    setContainerExecutionType(ContainerExecutionType::communication);
    setContainerOperationType(ContainerOperationType::halo);
    setDataViewSupport(ContainerAPI::DataViewSupport::off);
}


auto HaloUpdateContainer::
    parse()
        -> const std::vector<Neon::set::dataDependency::Token>&
{
    return mEmtpy;
}

auto HaloUpdateContainer::getHostContainer() -> std::shared_ptr<ContainerAPI>
{
    NEON_THROW_UNSUPPORTED_OPTION("This Container type can not be decoupled.");
}

auto HaloUpdateContainer::getDeviceContainer() -> std::shared_ptr<ContainerAPI>
{
    NEON_THROW_UNSUPPORTED_OPTION("This Container type can not be decoupled.");
}

auto HaloUpdateContainer::run(int /*streamIdx*/, Neon::DataView /*dataView*/) -> void
{
}

auto HaloUpdateContainer::run(Neon::SetIdx /*setIdx*/, int /*streamIdx*/, Neon::DataView /*dataView*/) -> void
{
}

}  // namespace Neon::set::internal
