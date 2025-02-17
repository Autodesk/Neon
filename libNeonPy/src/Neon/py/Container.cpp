#include <dlfcn.h>

#include <cuda.h>
#include <cudaTypedefs.h>
#include <rapidjson/reader.h>

#include "Neon/Neon.h"
#include "Neon/core/core.h"
#include "Neon/domain/Grids.h"
#include "Neon/domain/interface/GridBase.h"
#include "Neon/py/CudaDriver.h"
#include "Neon/py/macros.h"
#include "Neon/set/Containter.h"
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
             {
                 DataView::STANDARD,
                 DataView::BOUNDARY,
                 DataView::INTERNAL}) {
            this->setLaunchParameters(dw) = grid.getLaunchParameters(dw, blockSize, sharedMem);
        }
    }

    auto initLaunchParameters(
        const Grid& grid)
    {
        // std::cout << "Grid " << grid.toString() << std::endl;

        size_t sharedMem = 0;
        for (auto dw : {
                 DataView::STANDARD,
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


    auto parse() -> const std::vector<Neon::set::dataDependency::Token>& override
    {
        if (!this->isParsingDataUpdated()) {
            auto parser = Neon::set::Loader(*this,
                                            Neon::Execution::host,
                                            Neon::SetIdx(0),
                                            Neon::DataView::STANDARD,
                                            Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA);
            this->loadingLambda(parser);
            this->setParsingDataUpdated(true);

            this->setContainerPattern(this->getTokens());
        }
        return getTokens();
        // NEON_THROW_UNSUPPORTED_OPTION("");
    }

    template <typename Field>
    auto register_manual_loading_step(Field&                     f,
                                      Neon::Pattern              computeE,
                                      Neon::set::StencilSemantic stencilSemantic)
    {
        if constexpr (std::is_const_v<Field>) {
            auto step = [=](Neon::set::Loader& loader) {
                const Field& fConstView = f;
                loader.load(fConstView, computeE, stencilSemantic);
            };
            std::function<void(Neon::set::Loader&)> stepFunction = step;
            m_loadingLambdaSteps.push_back(stepFunction);
        } else {
            auto step = [=](Neon::set::Loader& loader) mutable {
                loader.load(f, computeE, stencilSemantic);
            };
            std::function<void(Neon::set::Loader&)> stepFunction = step;
            m_loadingLambdaSteps.push_back(stepFunction);
        }
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
        NEON_THROW_UNSUPPORTED_OPTION("");
    }

    auto loadingLambda(Neon::set::Loader& loader) -> void
    {
        for (auto loadingStep : m_loadingLambdaSteps) {
            loadingStep(loader);
        }
    }

   private:
    std::vector<std::function<void(Neon::set::Loader&)> > m_loadingLambdaSteps;

    Neon::py::CudaDriver*      m_cudaDriver;
    Grid*                      m_gridPtr = nullptr;
    Neon::Backend*             m_backendPtr;
    Neon::Execution            m_execution;
    Neon::set::DataSet<kernel> m_kernels[Neon::DataViewUtil::nConfig];
};

template <typename Grid>
struct container_warp_data
{
    Neon::py::CudaDriver*          m_cuda_driver_ptr;
    Grid*                          m_grid_ptr;
    Neon::py::WarpContainer<Grid>* m_warp_container_ptr;
    Neon::set::Container*          m_container_prt;
    Neon::set::Loader*             m_parser_ptr;

   public:
    container_warp_data(Neon::Execution execution,
                        void*           cuda_driver_handle,
                        void*           grid_handle,
                        void**          kernels_matrix,
                        Neon::index_3d* /*blockSize*/)
    {
        m_cuda_driver_ptr = reinterpret_cast<Neon::py::CudaDriver*>(cuda_driver_handle);
        m_grid_ptr = reinterpret_cast<Grid*>(grid_handle);

        m_warp_container_ptr = new (std::nothrow)
            Neon::py::WarpContainer<Grid>(
                execution,
                m_cuda_driver_ptr,
                m_grid_ptr,
                kernels_matrix);

        if (m_warp_container_ptr == nullptr) {
            Neon::NeonException e("warp_container_new");
            NEON_THROW(e);
        }

        std::shared_ptr<Neon::set::internal::ContainerAPI> tmp(m_warp_container_ptr);
        m_container_prt = Neon::set::Container::factoryNewWarp(tmp);

        if (m_container_prt == nullptr) {
            Neon::NeonException e("warp_container_new");
            NEON_THROW(e);
        }

        m_parser_ptr = new Neon::set::Loader(*m_warp_container_ptr,
                                             Neon::Execution::host,
                                             Neon::SetIdx(0),
                                             Neon::DataView::STANDARD,
                                             Neon::set::internal::LoadingMode_e::PARSE_AND_EXTRACT_LAMBDA);
        if (m_parser_ptr == nullptr) {
            Neon::NeonException e("warp_container_new");
            NEON_THROW(e);
        }
    }

    ~container_warp_data()
    {
        delete m_container_prt;
    }
};

auto container_warp_data_get_container_prt(void* data_prt) -> Neon::set::Container*
{
    // We use the type dGrid to instantiate the object.
    // We could have used any other type as the return value
    // does not depend on the Grid type.
    auto data = static_cast<container_warp_data<Neon::dGrid>*>(data_prt);
    return data->m_container_prt;
}

}  // namespace Neon::py


extern "C" auto warp_dGrid_container_new(
    void**          handle,
    Neon::Execution execution,
    void*           handle_cudaDriver,
    void*           handle_dgrid,
    void**          kernels_matrix,
    Neon::index_3d* blockSize) -> int
{
    NEON_PY_PRINT_BEGIN(*handle)

    auto data = new (std::nothrow)
        Neon::py::container_warp_data<Neon::dGrid>(execution,
                                                   handle_cudaDriver,
                                                   handle_dgrid,
                                                   kernels_matrix,
                                                   blockSize);

    if (data == nullptr) {
        Neon::NeonException e("warp_dgrid_container_new");
        NEON_THROW(e);
    }

    *handle = reinterpret_cast<void*>(data);
    NEON_PY_PRINT_END(*handle);
    return 0;
}

extern "C" auto warp_bGrid_container_new(
    void**          handle,
    Neon::Execution execution,
    void*           handle_cudaDriver,
    void*           handle_dgrid,
    void**          kernels_matrix,
    Neon::index_3d* blockSize) -> int
{
    NEON_PY_PRINT_BEGIN(*handle)
    using Grid = Neon::bGrid;
    auto data = new (std::nothrow)
        Neon::py::container_warp_data<Grid>(execution,
                                            handle_cudaDriver,
                                            handle_dgrid,
                                            kernels_matrix,
                                            blockSize);

    if (data == nullptr) {
        Neon::NeonException e("warp_bGrid_container_new");
        NEON_THROW(e);
    }

    *handle = reinterpret_cast<void*>(data);
    NEON_PY_PRINT_END(*handle);
    return 0;
}

extern "C" auto warp_mGrid_container_new(
    void**          handle,
    int32_t         grid_level,
    Neon::Execution execution,
    void*           handle_cudaDriver,
    void*           handle_dgrid,
    void**          kernels_matrix,
    Neon::index_3d* blockSize) -> int
{
    NEON_PY_PRINT_BEGIN(*handle)
    using Grid = Neon::domain::mGrid;
    auto* grid = reinterpret_cast<Grid*>(handle_dgrid);
    auto  level_grid = grid->operator()(grid_level);
    auto  data = new (std::nothrow)
        Neon::py::container_warp_data<decltype(level_grid)>(execution,
                                                            handle_cudaDriver,
                                                            &level_grid,
                                                            kernels_matrix,
                                                            blockSize);

    if (data == nullptr) {
        Neon::NeonException e("warp_mGrid_container_new");
        NEON_THROW(e);
    }

    *handle = reinterpret_cast<void*>(data);
    NEON_PY_PRINT_END(*handle);
    return 0;
}

extern "C" auto warp_container_delete(
    void** handle) -> int
{
    auto* data = reinterpret_cast<Neon::py::container_warp_data<Neon::dGrid>*>(*handle);

    if (data != nullptr) {
        delete data;
    }

    (*handle) = nullptr;
    return 0;
}

extern "C" auto warp_container_parse(
    void* handle) -> int
{
    auto* data = reinterpret_cast<Neon::py::container_warp_data<Neon::dGrid>*>(handle);
    data->m_warp_container_ptr->parse();
    return 0;
}

extern "C" auto warp_container_run(
    void*          handle,
    int            streamIdx,
    Neon::DataView dataView) -> int
{
    auto* data = reinterpret_cast<Neon::py::container_warp_data<Neon::dGrid>*>(handle);
    data->m_container_prt->run(streamIdx, dataView);
    return 0;
}


#define DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(TYPE, CARD)                                            \
    extern "C" auto warp_dgrid_container_add_parse_token_##TYPE##_##CARD(                                  \
        void* handle,                                                                                      \
        void* field_handle,                                                                                \
        int   access_int,                                                                                  \
        int   pattern_int,                                                                                 \
        int   stencilSemantic_int)                                                                         \
        ->int                                                                                              \
    {                                                                                                      \
        using Field = Neon::dGrid::Field<TYPE, CARD>;                                                      \
        auto  pattern = Neon::PatternUtils::fromInt(pattern_int);                                          \
        auto  access = Neon::set::dataDependency::AccessTypeUtils::fromInt(access_int);                    \
        auto  stenSemantic = Neon::set::StencilSemanticUtils::fromInt(stencilSemantic_int);                \
        auto* data = reinterpret_cast<Neon::py::container_warp_data<Neon::dGrid>*>(handle);                \
                                                                                                           \
        Field* field = reinterpret_cast<Field*>(field_handle);                                             \
        if (field == nullptr) {                                                                            \
            Neon::NeonException e("parse_token");                                                          \
            NEON_THROW(e);                                                                                 \
        }                                                                                                  \
                                                                                                           \
        if (access == Neon::set::dataDependency::AccessType::READ) {                                       \
            const Field& parsingField = *field;                                                            \
            data->m_warp_container_ptr->register_manual_loading_step(parsingField, pattern, stenSemantic); \
        } else {                                                                                           \
            Field& parsingField = *field;                                                                  \
            data->m_warp_container_ptr->register_manual_loading_step(parsingField, pattern, stenSemantic); \
        }                                                                                                  \
        return 0;                                                                                          \
    }

DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint8_t, 0)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint8_t, 1)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint8_t, 2)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint8_t, 3)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint8_t, 4)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint8_t, 5)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint8_t, 19)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint8_t, 27)

DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int8_t, 0)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int8_t, 1)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int8_t, 2)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int8_t, 3)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int8_t, 4)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int8_t, 5)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int8_t, 19)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int8_t, 27)

DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int32_t, 0)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int32_t, 1)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int32_t, 2)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int32_t, 3)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int32_t, 4)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int32_t, 5)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int32_t, 19)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int32_t, 27)

DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint32_t, 0)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint32_t, 1)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint32_t, 2)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint32_t, 3)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint32_t, 4)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint32_t, 5)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint32_t, 19)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint32_t, 27)

DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int64_t, 0)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int64_t, 1)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int64_t, 2)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int64_t, 3)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int64_t, 4)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int64_t, 5)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int64_t, 19)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(int64_t, 27)

DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint64_t, 0)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint64_t, 1)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint64_t, 2)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint64_t, 3)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint64_t, 4)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint64_t, 5)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint64_t, 19)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(uint64_t, 27)

DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(float, 0)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(float, 1)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(float, 2)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(float, 3)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(float, 4)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(float, 5)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(float, 19)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(float, 27)

DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(double, 0)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(double, 1)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(double, 2)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(double, 3)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(double, 4)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(double, 5)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(double, 19)
DEFINE_WARP_DGRID_CONTAINER_ADD_PARSE_TOKEN(double, 27)