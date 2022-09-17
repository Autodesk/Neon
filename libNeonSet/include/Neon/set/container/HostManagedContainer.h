#pragma once

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/HostManagedSyncType.h"
#include "Neon/set/container/Loader.h"

namespace Neon::set::internal {

/**
 * This Container run operation on the host
 * In terms of interaction with streams it always implement a barrier on the associated stream
 * The host computation is to be considered as synchronous, i.e. the executor will not proceed until
 * the host computation is completed. This guarantee the correctness of a host container embedded in a multi-GPU graph.
 *
 * A managed compute lambda will be launch for each device.
 * If the user objective is to launch some sequential application, then omp mechanism should be used in the managed
 * compute lambda to achieve it.
 */
template <typename DataContainerT,
          typename ComputeLambdaT>
struct HostManagedContainer : ContainerAPI
{
   public:
    virtual ~HostManagedContainer() override = default;


   public:
    /**
     * User facing API to define a kernel
     * @param data
     * @param userLambda
     */
    HostManagedContainer(const std::string&                                   name,
                         ContainerAPI::DataViewSupport                        dataViewSupport,
                         const DataContainerT&                                dataIteratorContainer,
                         std::function<ComputeLambdaT(Neon::SetIdx, Loader&)> loadingLambda,
                         HostManagedSyncType                                  preSyncType,
                         HostManagedSyncType                                  presSyncType)
        : mLoadingLambda(loadingLambda),
          mDataContainer(dataIteratorContainer)
    {
        setContainerExecutionType(ContainerExecutionType::hostManaged);
        setDataViewSupport(dataViewSupport);
        setName(name);
        mPreSyncType = preSyncType;
        mPostSyncType = presSyncType;

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

    auto parse() -> const std::vector<Neon::internal::dataDependency::Token>& override
    {
        auto         parser = newParser();
        Neon::SetIdx setIdx(0);
        this->mLoadingLambda(setIdx, parser);
        return getTokens();
    }

    auto getHostContainer() -> std::shared_ptr<ContainerAPI> override
    {
        NEON_THROW_UNSUPPORTED_OPTION("This is already a host Container.");
    }

    auto getDeviceContainer() -> std::shared_ptr<ContainerAPI> override
    {
        NEON_THROW_UNSUPPORTED_OPTION("This is a host Container.");
    }
    /**
     * Run container over streams
     * @param streamIdx
     * @param dataView
     */
    auto run(int            streamIdx = 0,
             Neon::DataView dataView = Neon::DataView::STANDARD)
        -> void override
    {
        const Neon::Backend& bk = mDataContainer.getBackend();
        int const            nDevs = bk.devSet().setCardinality();
#pragma omp parallel for num_threads(nDevs)
        for (int i = 0; i < nDevs; i++) {
            run(i, streamIdx, dataView);
        }
    }

    auto run(Neon::SetIdx   setIdx,
             int            streamIdx = 0,
             Neon::DataView dataView = Neon::DataView::STANDARD) -> void override
    {
        if (ContainerExecutionType::deviceManaged == this->getContainerType()) {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
        // REMEMBER that this is run in parallel withing omp
        const Neon::Backend& bk = mDataContainer.getBackend();
        switch (mPreSyncType) {
            case HostManagedSyncType::multiGPU: {
                bk.sync(setIdx, streamIdx);
#pragma omp barrier
                break;
            }
            case HostManagedSyncType::intraGPU: {
                bk.sync(setIdx, streamIdx);

                break;
            }
        }

        Loader         loader = this->newLoader(bk.devType(), setIdx, dataView, LoadingMode_e::EXTRACT_LAMBDA);
        ComputeLambdaT computeLambda = this->mLoadingLambda(setIdx, loader);
        computeLambda(streamIdx, dataView);

        switch (mPostSyncType) {
            case HostManagedSyncType::multiGPU: {
#pragma omp barrier
                break;
            }
            case HostManagedSyncType::intraGPU: {
                break;
            }
        }
    }


   private:
    std::function<ComputeLambdaT(Neon::SetIdx, Loader&)> mLoadingLambda;
    /**
     * This is the container on which the function will be called
     * Most probably, this is going to be one of the grids: dGrid, eGrid
     */
    DataContainerT      mDataContainer;
    HostManagedSyncType mPreSyncType;
    HostManagedSyncType mPostSyncType;
};

}  // namespace Neon::set::internal
