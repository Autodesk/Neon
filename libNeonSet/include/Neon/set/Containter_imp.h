#pragma once

#include "Neon/set/DevSet.h"

#include "functional"
#include "type_traits"

#include "Neon/set/container/Loader.h"

#include "Neon/set/container/DataTransferContainer.h"
#include "Neon/set/container/DeviceContainer.h"
#include "Neon/set/container/DeviceManagedContainer.h"
#include "Neon/set/container/DeviceThenHostManagedContainer.h"
#include "Neon/set/container/GraphContainer.h"
#include "Neon/set/container/HostContainer.h"
#include "Neon/set/container/HostManagedContainer.h"
#include "Neon/set/container/OldDeviceManagedContainer.h"
#include "Neon/set/container/SynchronizationContainer.h"


namespace Neon::set {


template <Neon::Execution execution,
          typename DataContainerT,
          typename UserLoadingLambdaT>
auto Container::factory(const std::string&                                 name,
                        Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport,
                        const DataContainerT&                              a,
                        const UserLoadingLambdaT&                          f,
                        const index_3d&                                    blockSize,
                        std::function<int(const index_3d& blockSize)>      shMemSizeFun) -> Container
{
    using LoadingLambda = typename std::invoke_result<decltype(f), Neon::set::Loader&>::type;
    if constexpr (Neon::Execution::device == execution) {
        auto k = new Neon::set::internal::DeviceContainer<DataContainerT, LoadingLambda>(name,
                                                                                         execution, dataViewSupport,
                                                                                         a, f,
                                                                                         blockSize, shMemSizeFun);

        std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
        return {tmp};
    } else {
        auto k = new Neon::set::internal::HostContainer<DataContainerT, LoadingLambda>(name, dataViewSupport,
                                                                                       a, f,
                                                                                       blockSize, shMemSizeFun);

        std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
        return {tmp};
    }
    NEON_THROW_UNSUPPORTED_OPERATION("Execution type not supported");
}

template <typename DataContainerT, typename UserLoadingLambdaT>
auto Container::hostFactory(const std::string&                                 name,
                            Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport,
                            const DataContainerT&                              a,
                            const UserLoadingLambdaT&                          f,
                            const index_3d&                                    blockSize,
                            std::function<int(const index_3d& blockSize)>      shMemSizeFun) -> Container
{
    using LoadingLambda = typename std::invoke_result<decltype(f), Neon::set::Loader&>::type;
    auto k = new Neon::set::internal::HostContainer<DataContainerT, LoadingLambda>(name, dataViewSupport,
                                                                                   a, f,
                                                                                   blockSize, shMemSizeFun);

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return {tmp};
}

template <typename DataContainerT,
          typename UserLoadingLambdaT>
auto Container::factoryOldManaged(const std::string&                                 name,
                                  Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport,
                                  Neon::set::ContainerPatternType                    patternType,
                                  DataContainerT                                     a,
                                  const UserLoadingLambdaT&                          f)
    -> Container
{
    using ManagedLaunch = typename std::invoke_result<decltype(f), Neon::set::Loader&>::type;
    auto k = new Neon::set::internal::OldDeviceManagedContainer<DataContainerT, ManagedLaunch>(name,
                                                                                               dataViewSupport,
                                                                                               patternType,
                                                                                               a,
                                                                                               f);

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return {tmp};
}

template <typename DataContainerT,
          typename UserLoadingLambdaT>
auto Container::factoryHostManaged(const std::string&                                 name,
                                   Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport,
                                   Neon::set::internal::HostManagedSyncType           preSyncType,
                                   Neon::set::internal::HostManagedSyncType           presSyncType,
                                   DataContainerT                                     a,
                                   const UserLoadingLambdaT&                          f)
    -> Container
{
    using ManagedLaunch = typename std::invoke_result<decltype(f), Neon::SetIdx, Neon::set::Loader&>::type;
    auto k = new Neon::set::internal::HostManagedContainer<DataContainerT, ManagedLaunch>(name, dataViewSupport,
                                                                                          a, f,
                                                                                          preSyncType,
                                                                                          presSyncType);

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return {tmp};
}

template <typename DataContainerT,
          typename UserLoadingLambdaT>
auto Container::factoryDeviceManaged(const std::string&                                 name,
                                     Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport,
                                     DataContainerT                                     a,
                                     const UserLoadingLambdaT&                          f)
    -> Container
{
    using ManagedLaunch = typename std::invoke_result<decltype(f), Neon::SetIdx, Neon::set::Loader&>::type;
    auto k = new Neon::set::internal::DeviceManagedContainer<DataContainerT, ManagedLaunch>(name, dataViewSupport,
                                                                                            a, f);

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return {tmp};
}

template <typename MultiXpuDataT>
auto Container::
    factoryDataTransfer(const MultiXpuDataT&                                              multiXpuData,
                        Neon::set::TransferMode                                           transferMode,
                        Neon::set::StencilSemantic                                        transferSemantic,
                        Neon::set::DataSet<std::vector<Neon::set::MemoryTransfer>> const& memoryTransfers,
                        Neon::Execution                                                   execution)
        -> Neon::set::Container
{
    auto k = new Neon::set::internal::DataTransferContainer(multiXpuData,
                                                            transferMode,
                                                            transferSemantic,
                                                            memoryTransfers,
                                                            execution);

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return {tmp};
}

template <typename MxpuDataT>
auto Container::
    factorySynchronization(const MxpuDataT&             multiXpuData,
                           SynchronizationContainerType syncType) -> Container
{
    auto k = new Neon::set::internal::SynchronizationContainer(multiXpuData,
                                                               syncType);

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return {tmp};
}

}  // namespace Neon::set
