#pragma once
#include "Neon/core/core.h"

#include "Neon/set/ContainerTools/ContainerAPI.h"
#include "Neon/set/ContainerTools/Loader.h"

namespace Neon {
namespace set {
namespace internal {

/**
 * Specialized implementation of KContainer_i
 *
 *
 * @tparam DataIteratorContainerT
 * @tparam UserComputeLambdaT
 */
template <typename DataIteratorContainerT,
          typename UserComputeLambdaT>
struct DeviceContainer : ContainerAPI
{
   public:
    virtual ~DeviceContainer() override = default;

   public:
    DeviceContainer(const std::string&                            name,
                    ContainerAPI::DataViewSupport                 dataViewSupport,
                    const DataIteratorContainerT&                 dataIteratorContainer,
                    std::function<UserComputeLambdaT(Loader&)>    loadingLambda,
                    const Neon::index_3d&                         blockSize,
                    std::function<int(const index_3d& blockSize)> shMemSizeFun)
        : m_loadingLambda(loadingLambda),
          m_dataIteratorContainer(dataIteratorContainer)
    {
        setName(name);
        setContainerType(ContainerType::device);
        setDataViewSupport(dataViewSupport);

        initLaunchParameters(dataIteratorContainer, blockSize, shMemSizeFun);
    }

    auto initLaunchParameters(const DataIteratorContainerT&                 dataIteratorContainer,
                              const Neon::index_3d&                         blockSize,
                              std::function<int(const index_3d& blockSize)> shMemSizeFun)
    {
        size_t sharedMem = shMemSizeFun(blockSize);
        for (auto dw : {DataView::STANDARD,
                        DataView::BOUNDARY,
                        DataView::INTERNAL}) {
            this->setLaunchParameters(dw) = dataIteratorContainer.getLaunchParameters(dw, blockSize, sharedMem);
        }
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
        this->m_loadingLambda(parser);
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

        const Neon::Backend&    bk = m_dataIteratorContainer.getBackend();
        Neon::set::KernelConfig kernelConfig(dataView, bk, streamIdx, this->getLaunchParameters(dataView));

        if (ContainerType::device == this->getContainerType()) {
            bk.devSet().template kernelLambdaWithIterator<DataIteratorContainerT, UserComputeLambdaT>(
                kernelConfig,
                m_dataIteratorContainer,
                [&](Neon::DeviceType devE, Neon::SetIdx setIdx, Neon::DataView dataView) -> UserComputeLambdaT {
                    Loader             loader = this->newLoader(devE, setIdx, dataView, LoadingMode_e::EXTRACT_LAMBDA);
                    UserComputeLambdaT userLambda = this->m_loadingLambda(loader);
                    return userLambda;
                });
            return;
        }

        NEON_THROW_UNSUPPORTED_OPTION("");
    }

    /**
     * Run container over streams
     * @param streamIdx
     * @param dataView
     */
    virtual auto run(Neon::SetIdx setIdx, int streamIdx, Neon::DataView dataView) -> void override
    {

        const Neon::Backend&    bk = m_dataIteratorContainer.getBackend();
        Neon::set::KernelConfig kernelConfig(dataView, bk, streamIdx, this->getLaunchParameters(dataView));

        if (ContainerType::device == this->getContainerType()) {
            bk.devSet().template kernelLambdaWithIterator<DataIteratorContainerT, UserComputeLambdaT>(
                setIdx,
                kernelConfig,
                m_dataIteratorContainer,
                [&](Neon::DeviceType devE, Neon::SetIdx setIdx, Neon::DataView dataView) -> UserComputeLambdaT {
                    Loader             loader = this->newLoader(devE, setIdx, dataView, LoadingMode_e::EXTRACT_LAMBDA);
                    UserComputeLambdaT userLambda = this->m_loadingLambda(loader);
                    return userLambda;
                });
            return;
        }

        NEON_THROW_UNSUPPORTED_OPTION("");
    }

   private:
    std::function<UserComputeLambdaT(Loader&)> m_loadingLambda;
    /**
     * This is the container on which the function will be called
     * Most probably, this is going to be one of the grids: dGrid, eGrid
     */
    DataIteratorContainerT m_dataIteratorContainer;
};

}  // namespace internal
}  // namespace set
}  // namespace Neon
