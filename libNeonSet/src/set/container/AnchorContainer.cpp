#include "Neon/core/core.h"

#include "Neon/set/container/AnchorContainer.h"

namespace Neon::set::internal {

AnchorContainer::AnchorContainer(const std::string& name)
{
    setName(name);
    setContainerExecutionType(ContainerExecutionType::none);
    setContainerOperationType(ContainerOperationType::anchor);
    setDataViewSupport(ContainerAPI::DataViewSupport::off);
}


auto AnchorContainer::
    parse()
        -> const std::vector<Neon::set::dataDependency::Token>&
{
    return mEmtpy;
}

auto AnchorContainer::
    run(int /*streamIdx*/,
        Neon::DataView /*dataView*/)
        -> void
{
}

auto AnchorContainer::
    run(Neon::SetIdx /*setIdx*/,
        int /*streamIdx*/,
        Neon::DataView /*dataView*/)
        -> void
{
}

}  // namespace Neon::set::internal
