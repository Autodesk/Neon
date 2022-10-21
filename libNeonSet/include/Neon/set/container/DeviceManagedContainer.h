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
struct DeviceManagedContainer : ContainerAPI
{
   public:
    ~DeviceManagedContainer() override = default;

    /**
     * User facing API to define a kernel
     * @param data
     * @param userLambda
     */
    DeviceManagedContainer(const std::string&                                   name,
                           Neon::set::internal::ContainerAPI::DataViewSupport   dataViewSupport,
                           const DataContainer&                                 dataIteratorContainer,
                           std::function<ComputeLambdaT(Neon::SetIdx, Loader&)> loadingLambda)
        : mLoadingLambda(loadingLambda),
          mDataContainer(dataIteratorContainer)
    {
        setContainerExecutionType(ContainerExecutionType::deviceManaged);
        setDataViewSupport(dataViewSupport);
        setName(name);

        this->parse();

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

    auto parse() -> const std::vector<Neon::set::dataDependency::Token>& override
    {
        Neon::SetIdx setIdx(0);
        if (!this->mParsingDataUpdated) {
            auto parser = newParser();
            this->m_loadingLambda(setIdx, parser);
            this->mParsingDataUpdated = true;

            this->setContainerPattern(this->getTokens());
        }
        return getTokens();
    }

    auto getHostContainer() -> std::shared_ptr<internal::ContainerAPI> override
    {
        NEON_THROW_UNSUPPORTED_OPTION("A managed Container Container is not associated with any host operation.");
    }

    auto getDeviceContainer() -> std::shared_ptr<internal::ContainerAPI> override
    {
        NEON_THROW_UNSUPPORTED_OPTION("A managed Container Container is not associated with any host operation.");
    }

    /**
     * Run container over streams
     * @param streamIdx
     * @param dataView
     */
    auto run(int streamIdx = 0, Neon::DataView dataView = Neon::DataView::STANDARD) -> void override
    {
        const Neon::Backend& bk = mDataContainer.getBackend();
        const int            setCardinality = bk.devSet().setCardinality();

#pragma omp parallel for num_threads(setCardinality)
        for (int i = 0; i < setCardinality; ++i) {
            run(Neon::SetIdx(i), streamIdx, dataView);
        }
    }

    auto run(Neon::SetIdx   setIdx,
             int            streamIdx = 0,
             Neon::DataView dataView = Neon::DataView::STANDARD) -> void override
    {
        if (ContainerExecutionType::deviceManaged == this->getContainerType()) {
            const Neon::Backend& bk = mDataContainer.getBackend();

            // We use device 0 as a dummy setIdx to create a loader.
            // The actual value is not important as the managed container will take care of launching on all devices.
            SetIdx         dummyTargetSetIdx = 0;
            Loader         loader = this->newLoader(bk.devType(), setIdx, dataView, LoadingMode_e::EXTRACT_LAMBDA);
            ComputeLambdaT computeLambda = this->mLoadingLambda(setIdx, loader);
            computeLambda(streamIdx, dataView);
            return;
        }
        NEON_THROW_UNSUPPORTED_OPTION("");
    }

   private:
    std::function<ComputeLambdaT(Neon::SetIdx, Loader&)> mLoadingLambda;
    /**
     * This is the container on which the function will be called
     * Most probably, this is going to be one of the grids: dGrid, eGrid
     */
    DataContainer mDataContainer;
};

}  // namespace Neon::set::internal
