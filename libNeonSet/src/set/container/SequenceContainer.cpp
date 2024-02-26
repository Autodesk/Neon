#include "Neon/set/container/SequenceContainer.h"
#include "Neon/set/container/Graph.h"

namespace Neon::set::internal {

SequenceContainer::
    SequenceContainer(const std::string&                       name,
                      std::vector<Neon::set::Container> const& sequence)
    : mSequence(sequence)
{
    setContainerExecutionType(ContainerExecutionType::sequence);
    setContainerOperationType(ContainerOperationType::sequence);
    setDataViewSupport(DataViewSupport::off);
    setName(name);
}

auto SequenceContainer::
    parse()
        -> const std::vector<Neon::set::dataDependency::Token>&
{
    NEON_THROW_UNSUPPORTED_OPERATION("SequenceContainer");
}

auto SequenceContainer::
    getSequence()
        const -> const std::vector<Neon::set::Container>&
{
    return mSequence;
}

/**
 * Run container over streams
 * @param streamIdx
 * @param dataView
 */
auto SequenceContainer::
    run(int            streamIdx,
        Neon::DataView dataView) -> void
{
    for (auto& container : mSequence) {
        container.run(streamIdx, dataView);
    }
}

auto SequenceContainer::
    run(Neon::SetIdx   setIdx,
        int            streamIdx,
        Neon::DataView dataView) -> void
{
    for (auto& container : mSequence) {
        container.run(setIdx, streamIdx, dataView);
    }
}

}  // namespace Neon::set::internal
