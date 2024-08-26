#include <dlfcn.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <rapidjson/reader.h>

#include "Neon/Neon.h"
#include "Neon/core/core.h"
#include "Neon/domain/Grids.h"
#include "Neon/domain/interface/GridBase.h"
#include "Neon/py/CudaDriver.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

namespace Neon::py {

template <typename Grid>
struct WarpContainer : Neon::set::internal::ContainerAPI
{
    using kernel = Neon::py::CudaDriver::kernel;

   public:
    virtual ~WarpContainer() override = default;

    WarpContainer(
        const Neon::Execution execution,
        Neon::py::CudaDriver* cuda_driver,
        Grid*                 grid,
        void**                kernels_matrix)
        : m_cudaDriver(cuda_driver),
          m_gridPtr(grid),
          m_backendPtr(cuda_driver->get_bk_prt()),
          m_execution(execution)
    {
        this->setName("WarpContainer");

        setContainerExecutionType(Neon::set::ContainerExecutionType::device);
        setContainerOperationType(Neon::set::ContainerOperationType::compute);
        setDataViewSupport(Neon::set::internal::ContainerAPI::DataViewSupport::on);


        initLaunchParameters(*grid);

        int const ndevs = m_backendPtr->getDeviceCount();

        for (const auto& dw : Neon::DataViewUtil::validOptions()) {
            int dw_idx = Neon::DataViewUtil::toInt(dw);
            m_kernels[dw_idx] = m_backendPtr->newDataSet<kernel>();
            for (int dev_idx = 0; dev_idx < ndevs; dev_idx++) {
                m_kernels[dw_idx][dev_idx] = kernels_matrix[dev_idx * Neon::DataViewUtil::nConfig + dw_idx];
            }
        }

        // this->parse();
    }

    auto initLaunchParameters(
        const Grid&                                   grid,
        const Neon::index_3d&                         blockSize,
        std::function<int(const index_3d& blockSize)> shMemSizeFun)
    {
        size_t sharedMem = shMemSizeFun(blockSize);
        for (auto dw :
             {DataView::STANDARD,
              DataView::BOUNDARY,
              DataView::INTERNAL}) {
            this->setLaunchParameters(dw) = grid.getLaunchParameters(dw, blockSize, sharedMem);
        }
    }

    auto initLaunchParameters(
        const Grid& grid)
    {
        //std::cout << "Grid " << grid.toString() << std::endl;

        size_t sharedMem = 0;
        for (auto dw : {DataView::STANDARD,
                        DataView::BOUNDARY,
                        DataView::INTERNAL}) {
            this->setLaunchParameters(dw) = grid.getLaunchParameters(dw, grid.getDefaultBlock(), sharedMem);
        }
    }

    //
    // auto newLoader(Neon::SetIdx     setIdx,
    //                Neon::DataView   dataView,
    //                LoadingMode_e::e loadingMode) -> Loader
    // {
    //     auto loader = Loader(*this,
    //                          mExecution,
    //                          setIdx,
    //                          dataView,
    //                          loadingMode);
    //     return loader;
    // }
    //
    // auto newParser() -> Loader
    // {
    //     auto parser = Loader(*this,
    //                          Neon::Execution::host,
    //                          Neon::SetIdx(0),
    //                          Neon::DataView::STANDARD,
    //                          Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA);
    //     return parser;
    // }
    //
    auto parse() -> const std::vector<Neon::set::dataDependency::Token>& override
    {
        // if (!this->isParsingDataUpdated()) {
        //     auto parser = newParser();
        //     this->m_loadingLambda(parser);
        //     this->setParsingDataUpdated(true);
        //
        //     this->setContainerPattern(this->getTokens());
        // }
        // return getTokens();
        NEON_THROW_UNSUPPORTED_OPTION("");
    }


    virtual auto run(int            streamIdx = 0,
                     Neon::DataView dataView = Neon::DataView::STANDARD) -> void override
    {
        auto                    launchParameters = this->getLaunchParameters(dataView);
        Neon::set::KernelConfig kernelConfig(
            dataView,
            *m_backendPtr,
            streamIdx,
            launchParameters);


        m_cudaDriver->run_kernel(
            m_kernels[Neon::DataViewUtil::toInt(dataView)],
            launchParameters,
            streamIdx);

        return;
    }

    /**
     * Run container over streams
     * @param streamIdx
     * @param dataView
     */
    virtual auto run(
        [[maybe_unused]] Neon::SetIdx   setIdx,
        [[maybe_unused]] int            streamIdx,
        [[maybe_unused]] Neon::DataView dataView) -> void override
    {

        //         const Neon::Backend&    bk = m_grid.getBackend();
        //         Neon::set::KernelConfig kernelConfig(dataView, bk, streamIdx, this->getLaunchParameters(dataView));
        //
        // #pragma omp critical
        //         {
        //             [[maybe_unused]] int const threadRank = omp_get_thread_num();
        //             NEON_TRACE("TRACE DeviceContainer run rank {} setIdx {} stream {} dw {}",
        //                        threadRank, setIdx.idx(), kernelConfig.stream(), Neon::DataViewUtil::toString(kernelConfig.dataView()));
        //         };
        //
        //         if (ContainerExecutionType::device == this->getContainerExecutionType()) {
        //             bk.devSet().template kernelDeviceLambdaWithIterator<DataIteratorContainerT, UserComputeLambdaT>(
        //                 mExecution,
        //                 setIdx,
        //                 kernelConfig,
        //                 m_dataIteratorContainer,
        //                 [&](Neon::SetIdx   setIdx,
        //                     Neon::DataView dataView)
        //                 -> UserComputeLambdaT {
        //                     Loader             loader = this->newLoader(setIdx, dataView, LoadingMode_e::EXTRACT_LAMBDA);
        //                     UserComputeLambdaT userLambda = this->m_loadingLambda(loader);
        //                     return userLambda;
        //                 });
        //             return;
        //         }

        NEON_THROW_UNSUPPORTED_OPTION("");
    }

   private:
    Neon::py::CudaDriver*      m_cudaDriver;
    Grid*                      m_gridPtr = nullptr;
    Neon::Backend*             m_backendPtr;
    Neon::Execution            m_execution;
    Neon::set::DataSet<kernel> m_kernels[Neon::DataViewUtil::nConfig];
};
}  // namespace Neon::py


extern "C" void warp_dgrid_container_new(
    void**          out_handle,
    Neon::Execution execution,
    void*           handle_cudaDriver,
    void*           handle_dgrid,
    void**          kernels_matrix,
    Neon::index_3d* /*blockSize*/)
{
    auto* cudaDriverPtr =
        reinterpret_cast<Neon::py::CudaDriver*>(handle_cudaDriver);

    auto* dGridPtr =
        reinterpret_cast<Neon::dGrid*>(handle_dgrid);

    auto warp_container = new (std::nothrow)
        Neon::py::WarpContainer<Neon::dGrid>(
            execution,
            cudaDriverPtr,
            dGridPtr,
            kernels_matrix);

    if (warp_container == nullptr) {
        Neon::NeonException e("warp_dgrid_container_new");
        NEON_THROW(e);
    }

    *out_handle = reinterpret_cast<void*>(warp_container);
}

extern "C" void warp_container_delete(
    uint64_t** handle)
{
    auto* warp_container =
        reinterpret_cast<Neon::py::WarpContainer<Neon::dGrid>*>(*handle);

    if (warp_container != nullptr) {
        delete warp_container;
    }

    (*handle) = 0;
}

extern "C" void warp_container_run(
    void*          handle,
    int            streamIdx,
    Neon::DataView dataView)
{
    auto* warpContainerPtr =
        reinterpret_cast<Neon::py::WarpContainer<Neon::dGrid>*>(handle);

    warpContainerPtr->run(streamIdx, dataView);
}