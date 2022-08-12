#pragma once

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

namespace Neon::set::internal {

/**
 * Specialized implementation of KContainer_i
 *
 *
 * @tparam DataContainer
 * @tparam ComputeLambdaT
 */
template <typename DataContainer,
          typename ComputeLambdaT>
struct OldDeviceManagedContainer : ContainerAPI
{
   public:
    ~OldDeviceManagedContainer() override = default;

    /**
     * User facing API to define a kernel
     * @param data
     * @param userLambda
     */
    OldDeviceManagedContainer(const std::string&                                 name,
                              Neon::set::internal::ContainerAPI::DataViewSupport dataViewSupport,
                              const DataContainer&                               dataIteratorContainer,
                              std::function<ComputeLambdaT(Loader&)>             loadingLambda)
        : mLoadingLambda(loadingLambda),
          mDataContainer(dataIteratorContainer)
    {
        setContainerExecutionType(ContainerExecutionType::deviceManaged);
        setDataViewSupport(dataViewSupport);
        setName(name);
    }

    auto newLoader(Neon::DeviceType devE,
                   Neon::SetIdx     setIdx,
                   Neon::DataView   dataView,
                   LoadingMode_e::e loadingMode) -> Loader
    {
        auto loader = Loader(*this,
                             devE,
                             setIdx,
                             dataView,
                             loadingMode);
        return loader;
    }

    auto newParser() -> Loader
    {
        auto parser = Loader(*this,
                             Neon::DeviceType::CPU,
                             Neon::SetIdx(0),
                             Neon::DataView::STANDARD,
                             Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA);
        return parser;
    }

    auto parse() -> const std::vector<Neon::set::internal::dependencyTools::DataToken>& override
    {
        auto parser = newParser();
        this->mLoadingLambda(parser);
        return getTokens();
    }

    auto getHostContainer() -> std::shared_ptr<ContainerAPI> final
    {
        NEON_THROW_UNSUPPORTED_OPTION("This Container type can not be decoupled.");
    }

    virtual auto getDeviceContainer() -> std::shared_ptr<ContainerAPI> final
    {
        NEON_THROW_UNSUPPORTED_OPTION("This Container type can not be decoupled.");
    }

    /**
     * Run container over streams
     * @param streamIdx
     * @param dataView
     */
    virtual auto run(int streamIdx = 0, Neon::DataView dataView = Neon::DataView::STANDARD) -> void override
    {
        const Neon::Backend& bk = mDataContainer.getBackend();

        // We use device 0 as a dummy setIdx to create a loader.
        // The actual value is not important as the managed container will take care of launching on all devices.
        SetIdx         dummyTargetSetIdx = 0;
        Loader         loader = this->newLoader(bk.devType(), dummyTargetSetIdx, dataView, LoadingMode_e::EXTRACT_LAMBDA);
        ComputeLambdaT computeLambda = this->mLoadingLambda(loader);
        computeLambda(streamIdx, dataView);
    }

    virtual auto run(Neon::SetIdx   setIdx,
                     int            streamIdx = 0,
                     Neon::DataView dataView = Neon::DataView::STANDARD) -> void override
    {
        (void)setIdx;
        (void)streamIdx;
        (void)dataView;
        NEON_DEV_UNDER_CONSTRUCTION("");
        //        Neon::set::KernelConfig kernelConfig = m_dataContainer.getKernelConfig(streamIdx, dataView);
        //        if (!isManaged()) {
        //            NEON_THROW_UNSUPPORTED_OPTION("");
        //        }
        //        const Neon::Backend& bk = m_dataContainer.getBackend();
        //
        //        Loader        loader = this->newLoader(bk.devType(), 0, dataView, LoadingMode_e::EXTRACT_LAMBDA);
        //        ComputeLambdaT managedLaunchFun = this->m_loadingLambda(loader);
        //        managedLaunchFun(setIdx, streamIdx, dataView);
    }

   private:
    std::function<ComputeLambdaT(Loader&)> mLoadingLambda;
    /**
     * This is the container on which the function will be called
     * Most probably, this is going to be one of the grids: dGrid, eGrid
     */
    DataContainer mDataContainer;
};

}  // namespace Neon
