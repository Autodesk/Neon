#include "Neon/set/Containter.h"
#include "Neon/set/container/AnchorContainer.h"
#include "Neon/set/container/Loader.h"
#include "Neon/set/container/SequenceContainer.h"
#include "Neon/set/container/SynchronizationContainer.h"

namespace Neon::set {

auto Container::run(int            streamIdx,
                    Neon::DataView dataView)
    -> void
{
    mContainer->run(streamIdx, dataView);
}


auto Container::run(Neon::SetIdx   setIdx,
                    int            streamIdx,
                    Neon::DataView dataView)
    -> void
{
    mContainer->run(setIdx, streamIdx, dataView);
}

auto Container::getContainerInterface()
    -> Neon::set::internal::ContainerAPI&
{
    return mContainer.operator*();
}

auto Container::getContainerInterface() const
    -> const Neon::set::internal::ContainerAPI&
{
    return mContainer.operator*();
}


auto Container::getContainerInterfaceShrPtr()
    -> std::shared_ptr<Neon::set::internal::ContainerAPI>
{
    return mContainer;
}

auto Container::factoryDeviceThenHostManaged(const std::string& name,
                                             Container&         device,
                                             Container&         host) -> Container
{
    auto k = new Neon::set::internal::DeviceThenHostManagedContainer(name,
                                                                     device.getContainerInterfaceShrPtr(),
                                                                     host.getContainerInterfaceShrPtr());

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return Container(tmp);
}

auto Container::factoryAnchor(const std::string& name) -> Container
{
    auto                                               k = new Neon::set::internal::AnchorContainer(name);
    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return Container(tmp);
}


auto Container::getName() const
    -> const std::string&
{
    return mContainer->getName();
}

auto Container::getUid() const
    -> uint64_t
{
    const auto uid = (uint64_t)mContainer.get();
    return uid;
}

auto Container::logTokens()
    -> void
{
    return mContainer->toLog(getUid());
}

auto Container::getHostContainer() const
    -> Container
{
    std::shared_ptr<Neon::set::internal::ContainerAPI> hostAPI =
        mContainer->getHostContainer();
    return Container(hostAPI);
}

auto Container::getDeviceContainer() const -> Container
{
    std::shared_ptr<Neon::set::internal::ContainerAPI> deviceAPI =
        mContainer->getDeviceContainer();
    return Container(deviceAPI);
}

auto Container::getDataViewSupport() const
    -> Neon::set::internal::ContainerAPI::DataViewSupport
{
    auto&      api = this->getContainerInterface();
    auto const dwSupport = api.getDataViewSupport();
    return dwSupport;
}

auto Container::getContainerExecutionType() const
    -> Neon::set::ContainerExecutionType
{
    auto&      api = this->getContainerInterface();
    auto const type = api.getContainerExecutionType();
    return type;
}


auto Container::getSequence() const
    -> const std::vector<Neon::set::Container>&
{
    auto executionType = this->getContainerInterface().getContainerExecutionType();
    if (executionType == Neon::set::ContainerExecutionType::sequence) {
        auto&       api = this->getContainerInterface();
        auto const& seq = api.getSequence();
        return seq;
    } else {
        throw std::runtime_error("Container is not a sequence");
    }
}

auto Container::isSequence() const
    -> bool
{
    auto executionType = this->getContainerInterface().getContainerExecutionType();
    return executionType == Neon::set::ContainerExecutionType::sequence;
}

Container::Container(std::shared_ptr<Neon::set::internal::ContainerAPI>& container)
    : mContainer(container)
{
    // Empty
}

Container::Container(std::shared_ptr<Neon::set::internal::ContainerAPI>&& container)
    : mContainer(container)
{
    // Empty
}

auto Container::factoryGraph(const std::string&                         name,
                             const container::Graph&                    graph,
                             std::function<void(Neon::SetIdx, Loader&)> loadingLambda) -> Container
{
    auto k = new Neon::set::internal::GraphContainer(name, graph, loadingLambda);

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return Container(tmp);
}

auto Container::factorySequence(const std::string&                 name,
                                std::vector<Neon::set::Container>& sequence) -> Container
{
    auto k = new Neon::set::internal::SequenceContainer(name, sequence);

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return Container(tmp);
}

}  // namespace Neon::set
