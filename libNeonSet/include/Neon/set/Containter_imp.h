#pragma once

#include "Neon/set/DevSet.h"
#include "Neon/set/dependencyTools/DataParsing.h"
#include "functional"
#include "type_traits"

#include "Neon/set/ContainerTools/DeviceContainer.h"
#include "Neon/set/ContainerTools/DeviceManagedContainer.h"
#include "Neon/set/ContainerTools/DeviceThenHostManagedContainer.h"
#include "Neon/set/ContainerTools/HostManagedContainer.h"
#include "Neon/set/ContainerTools/OldDeviceManagedContainer.h"

namespace Neon::set {


template <typename DataContainerT, typename UserLoadingLambdaT>
auto Container::factory(const std::string&                                 name,
                        Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport,
                        const DataContainerT&                              a,
                        const UserLoadingLambdaT&                          f,
                        const index_3d&                                    blockSize,
                        std::function<int(const index_3d& blockSize)>      shMemSizeFun) -> Container
{
    using LoadingLambda = typename std::invoke_result<decltype(f), Neon::set::Loader&>::type;
    auto k = new Neon::set::internal::DeviceContainer<DataContainerT, LoadingLambda>(name, dataViewSupport,
                                                                                     a, f,
                                                                                     blockSize, shMemSizeFun);

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return Container(tmp);
}

template <typename DataContainerT,
          typename UserLoadingLambdaT>
auto Container::factoryOldManaged(const std::string&                                 name,
                                  Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport,
                                  DataContainerT                                     a,
                                  const UserLoadingLambdaT&                          f)
    -> Container
{
    using ManagedLaunch = typename std::invoke_result<decltype(f), Neon::set::Loader&>::type;
    auto k = new Neon::set::internal::OldDeviceManagedContainer<DataContainerT, ManagedLaunch>(name, dataViewSupport,
                                                                                               a, f);

    std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
    return Container(tmp);
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
    return Container(tmp);
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
    return Container(tmp);
}


}  // namespace Neon::set
