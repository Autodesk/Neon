#pragma once

#include "Neon/set/DevSet.h"
#include "Neon/set/dependencyTools/DataParsing.h"
#include "functional"
#include "type_traits"

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/HostManagedSyncType.h"
#include "Neon/set/container/Loader.h"

namespace Neon::set {


struct Container
{
   public:
    Container() = default;

    /**
     * Run a Neon Container on a given stream and with a given data view
     */
    auto run(int            streamIdx /**< Target stream */,
             Neon::DataView dataView = Neon::DataView::STANDARD /**< Target data view ( STANDARD by default ) */)
        -> void;

    /**
     * Runs a Neon Container on a given stream and with a given data view
     */
    auto run(Neon::SetIdx   setIdx /**< Set index of a target device */,
             int            streamIdx /**< Target stream */,
             Neon::DataView dataView = Neon::DataView::STANDARD /**< Target data view ( STANDARD by default ) */)
        -> void;

    /**
     * Returns the internal interface of a Neon Container.
     */
    auto getContainerInterface()
        -> Neon::set::internal::ContainerAPI&;

    /**
     * Returns the internal interface of a Neon Container.
     */
    auto getContainerInterface() const
        -> const Neon::set::internal::ContainerAPI&;

    /**
     * Returns the internal interface of a Neon Container.
     */
    auto getContainerInterfaceShrPtr()
        -> std::shared_ptr<Neon::set::internal::ContainerAPI>;

    /**
     * Factory function to create a Neon Container
     */
    template <typename DataContainerT, typename UserLoadingLambdaT>
    static auto factory(const std::string&                                 name /**< A user's string to identify the computation done by the Container. */,
                        Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport /**< Defines the data view support for the new Container */,
                        const DataContainerT&                              a /**< Multi device object that will be used for the creating of the Container */,
                        const UserLoadingLambdaT&                          f /**< User's loading lambda for the new Container */,
                        const index_3d&                                    blockSize /**< Block size for the thread grid */,
                        std::function<int(const index_3d& blockSize)>      shMemSizeFun /**< User's function to implicitly compute the required shared memory */)
        -> Container;

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
        -> Container;

    template <typename DataContainerT,
              typename UserLoadingLambdaT>
    static auto factoryHostManaged(const std::string&                                 name,
                                   Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport,
                                   Neon::set::internal::HostManagedSyncType           preSyncType,
                                   Neon::set::internal::HostManagedSyncType           presSyncType,
                                   DataContainerT                                     a,
                                   const UserLoadingLambdaT&                          f)
        -> Container;

    template <typename DataContainerT,
              typename UserLoadingLambdaT>
    static auto factoryDeviceManaged(const std::string&                                 name,
                                     Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport,
                                     DataContainerT                                     a,
                                     const UserLoadingLambdaT&                          f)
        -> Container;

    static auto factoryGraph(const std::string&                         name,
                             const container::Graph&                    graph,
                             std::function<void(Neon::SetIdx, Loader&)> loadingLambda) -> Container;

    static auto factoryDeviceThenHostManaged(const std::string& name,
                                             Container&         device,
                                             Container&         host)
        -> Container;

    static auto factoryAnchor(const std::string& name /**< A user's string to identify the computation done by the Container. */)
        -> Container;

    auto getName() const
        -> const std::string&;

    auto getUid() const
        -> uint64_t;

    auto logTokens()
        -> void;

    auto getHostContainer() const
        -> Container;

    virtual auto getDeviceContainer() const -> Container;

    auto getDataViewSupport() const
        -> Neon::set::internal::ContainerAPI::DataViewSupport;

    auto getContainerExecutionType() const
        -> Neon::set::ContainerExecutionType;

   protected:
    std::shared_ptr<Neon::set::internal::ContainerAPI> mContainer;

    Container(std::shared_ptr<Neon::set::internal::ContainerAPI>& container);

    Container(std::shared_ptr<Neon::set::internal::ContainerAPI>&& container);
};

}  // namespace Neon::set

#include "Neon/set/Containter_imp.h"