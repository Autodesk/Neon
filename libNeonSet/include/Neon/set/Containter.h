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

struct Container
{
   public:
    Container() = default;

    /**
     * Run a Neon Container on a given stream and with a given data view
     */
    auto run(int            streamIdx /**< Target stream */,
             Neon::DataView dataView = Neon::DataView::STANDARD /**< Target data view ( STANDARD by default ) */ )
        -> void
    {
        mContainer->run(streamIdx, dataView);
    }

    /**
     * Runs a Neon Container on a given stream and with a given data view
     */
    auto run(Neon::SetIdx   setIdx /**< Set index of a target device */,
             int            streamIdx /**< Target stream */,
             Neon::DataView dataView = Neon::DataView::STANDARD /**< Target data view ( STANDARD by default ) */)
        -> void
    {
        mContainer->run(setIdx, streamIdx, dataView);
    }

    /**
     * Returns the internal interface of a Neon Container.
     */
    auto getContainerInterface()
        -> Neon::set::internal::ContainerAPI&
    {
        return mContainer.operator*();
    }

    /**
     * Returns the internal interface of a Neon Container.
     */
    auto getContainerInterface() const
        -> const Neon::set::internal::ContainerAPI&
    {
        return mContainer.operator*();
    }

    /**
    * Returns the internal interface of a Neon Container.
    */
    auto getContainerInterfaceShrPtr()
        -> std::shared_ptr<Neon::set::internal::ContainerAPI>
    {
        return mContainer;
    }

    /**
     * Factory function to create a Neon Container
     */
    template <typename DataContainerT, typename UserLoadingLambdaT>
    static auto factory(const std::string&                                 name /**< A user's string to identify the computation done by the Container. */,
                        Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport /**< Defines the data view support for the new Container */,
                        const DataContainerT&                              a /**< Multi device object that will be used for the creating of the Container */,
                        const UserLoadingLambdaT&                          f /**< User's loading lambda for the new Container */,
                        const index_3d&                                    blockSize /**< Block size for the thread grid */,
                        std::function<int(const index_3d& blockSize)>      shMemSizeFun /**< User's function to implicitly compute the required shared memory */) -> Container
    {
        using LoadingLambda = typename std::invoke_result<decltype(f), Neon::set::Loader&>::type;
        auto k = new Neon::set::internal::DeviceContainer<DataContainerT, LoadingLambda>(name, dataViewSupport,
                                                                                         a, f,
                                                                                         blockSize, shMemSizeFun);

        std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
        return Container(tmp);
    }

    /**
     * Factory function to generate a kContainer object.
     * @tparam A: the type of the structure managing the iterator
     * @tparam F: the type of the lambda function that iterated over the data
     * @param a: the the structure managing the iterator
     * @param f: the the lambda function that iterated over the data
     * @return
     */
    template <typename DataContainerT,
              typename UserLoadingLambdaT>
    static auto factoryOldManaged(const std::string&                                 name,
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
    static auto factoryHostManaged(const std::string&                                 name,
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
    static auto factoryDeviceManaged(const std::string&                                 name,
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

    static auto factoryDeviceThenHostManaged(const std::string& name,
                                             Container&         device,
                                             Container&         host) -> Container
    {
        auto k = new Neon::set::internal::DeviceThenHostManagedContainer(name,
                                                                         device.getContainerInterfaceShrPtr(),
                                                                         host.getContainerInterfaceShrPtr());

        std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(k);
        return Container(tmp);
    }

    auto getName() const
        -> const std::string&
    {
        return mContainer->getName();
    }

    auto getUid() const
        -> uint64_t
    {
        const auto uid = (uint64_t)mContainer.get();
        return uid;
    }

    auto logTokens()
        -> void
    {
        return mContainer->toLog(getUid());
    }

    auto getHostContainer() const
        -> Container
    {
        std::shared_ptr<Neon::set::internal::ContainerAPI> hostAPI =
            mContainer->getHostContainer();
        return Container(hostAPI);
    }

    virtual auto getDeviceContainer() const -> Container
    {
        std::shared_ptr<Neon::set::internal::ContainerAPI> deviceAPI =
            mContainer->getDeviceContainer();
        return Container(deviceAPI);
    }

    auto getDataViewSupport() const
        -> Neon::set::internal::ContainerAPI::DataViewSupport
    {
        auto&      api = this->getContainerInterface();
        auto const dwSupport = api.getDataViewSupport();
        return dwSupport;
    }

    auto getContainerType() const
        -> Neon::set::internal::ContainerType
    {
        auto&      api = this->getContainerInterface();
        auto const type = api.getContainerType();
        return type;
    }

   protected:
    std::shared_ptr<Neon::set::internal::ContainerAPI> mContainer;

    Container(std::shared_ptr<Neon::set::internal::ContainerAPI>& container)
        : mContainer(container)
    {
        // Empty
    }
    Container(std::shared_ptr<Neon::set::internal::ContainerAPI>&& container)
        : mContainer(container)
    {
        // Empty
    }
};

}  // namespace Neon::set
