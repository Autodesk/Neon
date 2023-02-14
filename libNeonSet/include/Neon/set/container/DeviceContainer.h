#pragma once
#include "Neon/core/core.h"

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

namespace Neon::set::internal {

template <typename DataIteratorContainerT,
          typename UserComputeLambdaT>
struct DeviceContainer : ContainerAPI
{
   public:
    virtual ~DeviceContainer() override = default;

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
        setContainerExecutionType(ContainerExecutionType::device);
        setContainerOperationType(ContainerOperationType::compute);

        setDataViewSupport(dataViewSupport);

        initLaunchParameters(dataIteratorContainer, blockSize, shMemSizeFun);

        this->parse();
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

    auto parse() -> const std::vector<Neon::set::dataDependency::Token>& override
    {
        if (!this->isParsingDataUpdated()) {
            auto parser = newParser();
            this->m_loadingLambda(parser);
            this->setParsingDataUpdated(true);

            this->setContainerPattern(this->getTokens());
        }
        return getTokens();
    }

    /**
     * Run container over streams
     * @param streamIdx
     * @param dataView
     */
    virtual auto run(int            streamIdx = 0,
                     Neon::DataView dataView = Neon::DataView::STANDARD) -> void override
    {

        const Neon::Backend&    bk = m_dataIteratorContainer.getBackend();
        Neon::set::KernelConfig kernelConfig(dataView, bk, streamIdx, this->getLaunchParameters(dataView));

        if (ContainerExecutionType::device == this->getContainerExecutionType()) {
            bk.devSet().template kernelLambdaWithIterator<DataIteratorContainerT, UserComputeLambdaT>(
                kernelConfig,
                m_dataIteratorContainer,
                [&](Neon::DeviceType devE,
                    Neon::SetIdx     setIdx,
                    Neon::DataView   dataView)
                    -> UserComputeLambdaT {
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
    virtual auto run(Neon::SetIdx   setIdx,
                     int            streamIdx,
                     Neon::DataView dataView) -> void override
    {

        const Neon::Backend&    bk = m_dataIteratorContainer.getBackend();
        Neon::set::KernelConfig kernelConfig(dataView, bk, streamIdx, this->getLaunchParameters(dataView));

#pragma omp critical
        {
            [[maybe_unused]] int const threadRank = omp_get_thread_num();
            NEON_TRACE("TRACE DeviceContainer run rank {} setIdx {} stream {} dw {}",
                       threadRank, setIdx.idx(), kernelConfig.stream(), Neon::DataViewUtil::toString(kernelConfig.dataView()));
        };

        if (ContainerExecutionType::device == this->getContainerExecutionType()) {
            bk.devSet().template kernelLambdaWithIterator<DataIteratorContainerT, UserComputeLambdaT>(
                setIdx,
                kernelConfig,
                m_dataIteratorContainer,
                [&](Neon::DeviceType devE,
                    Neon::SetIdx     setIdx,
                    Neon::DataView   dataView)
                    -> UserComputeLambdaT {
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

}  // namespace Neon::set::internal
