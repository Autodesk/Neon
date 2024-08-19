#include <dlfcn.h>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "Neon/Neon.h"
#include "Neon/core/core.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"
#include "Neon/py/CudaDriver.h"
#include "Neon/py/macros.h"

namespace Neon::py {
CudaDriver::CudaDriver(Neon::Backend* bk_prt)
{
    Neon::init();
    this->backend = *bk_prt;
    cu_devices = backend.newDataSet<CUdevice>();
    cu_contexts = backend.newDataSet<CUcontext>();

    auto     dev_list = backend.devSet();
    CUresult res = cuInit(0);
    check_cuda_res(res, "cuInit");

    int const ndevs = backend.getDeviceCount();
    auto      devSet = backend.devSet();
#pragma omp parallel for num_threads(ndevs)
    for (int setIdx = 0; setIdx < ndevs; setIdx++) {
        Neon::sys::DeviceID devId = devSet.devId(setIdx);
        int                 cuda_dev_idx = devId.idx();
        res = cuDeviceGet(&cu_devices[setIdx], cuda_dev_idx);
        check_cuda_res(res, std::string("cuDeviceGet (dev ") + std::to_string(cuda_dev_idx) + std::string(")"));

        res = cuCtxCreate(&cu_contexts[setIdx], 0, cuda_dev_idx);
        check_cuda_res(res, std::string("cuCtxCreate (dev ") + std::to_string(cuda_dev_idx) + std::string(")"));
    }

    // kernelSet = bk.newDataSet<void*>([](Neon::SetIdx const&, void*& ptr) {
    //     ptr = nullptr;
    // });

}

CudaDriver::~CudaDriver()
{

    auto dev_list = backend.devSet();

    int  ndevs = backend.getDeviceCount();
    auto devSet = backend.devSet();
    for (int setIdx = 0; setIdx < ndevs; setIdx++) {
        Neon::sys::DeviceID devId = devSet.devId(setIdx);
        int                 cuda_dev_idx = devId.idx();

        CUresult res = cuCtxDestroy(cu_contexts[setIdx]);
        check_cuda_res(res, std::string("cuCtxCreate (dev ") + std::to_string(cuda_dev_idx) + std::string(")"));
    }
}


auto CudaDriver::run_kernel(
    Neon::set::DataSet<kernel> const&  kernelSet,
    Neon::set::LaunchParameters const& launch_params,
    Neon::StreamIdx                    streamIdx) -> void
{
    [[maybe_unused]] auto& streamSet = backend.streamSet(streamIdx);

    int const ndevs = backend.getDeviceCount();
    //#pragma omp parallel for num_threads(ndevs)
    for (int setIdx = 0; setIdx < ndevs; setIdx++) {
        // cudaStream_t const& cuda_stream = streamSet.cudaStream(setIdx);
        // CUstream            driverStream = (CUstream)cuda_stream;
        CUfunction function = static_cast<CUfunction>(kernelSet[setIdx]);
        std::cout << "foo " << function << std::endl;
        backend.devSet().setActiveDevContext(setIdx);
        auto& launch_info = launch_params[setIdx];
        // auto const cudaGrid = launch_info.cudaGrid();
        // auto const cudaBlock = launch_info.cudaBlock();
        // Set the created context as the current context
        //CUresult res = cuCtxSetCurrent(cu_contexts[setIdx]);
        //check_cuda_res(res, "cuCtxSetCurrent");
        int64_t pywarp_size = 1;
        std::cout << "pywarp_size" << pywarp_size << std::endl;
        const int LAUNCH_MAX_DIMS = 4; // should match types.py
        struct launch_bounds_t
        {
            int    shape[LAUNCH_MAX_DIMS]; // size of each dimension
            int    ndim; // number of valid dimension
            size_t size; // total number of threads
        };

        launch_bounds_t bounds;
        std::cout << "domaina " << Neon::index_3d(launch_info.cudaGrid().x,
                                                  launch_info.cudaGrid().y,
                                                  launch_info.cudaGrid().z) << std::endl;
        int n = 2;
        bounds.ndim = 1;
        bounds.shape[0] = n;
        bounds.size = n;


        std::vector<void*> args;
        args.push_back(&bounds);

        [[maybe_unused]] auto devset = backend.devSet();
        devset.setActiveDevContext(setIdx);
        [[maybe_unused]] auto const& gpuDev = devset.gpuDev(setIdx);
        [[maybe_unused]] auto        kinfo = launch_params.operator[](setIdx);
        // try {
        //     gpuDev.kernel.cudaLaunchKernel<Neon::run_et::sync>(streamSet[setIdx], kinfo, function, args.data());
        // } catch (...) {
        //
        // }
        int block_dim = 256;
        int grid_dim = (n + block_dim - 1) / block_dim;
        std::cout << "block_dim " << block_dim << std::endl;
        std::cout << "grid_dim " << grid_dim << std::endl;
        std::cout << "n  " << n << std::endl;
        std::cout << "cuLaunchKernel" << std::endl;
        CUresult res = cuLaunchKernel(
            function,
            grid_dim, 1, 1,
            block_dim, 1, 1,
            0,
            nullptr,
            args.data(),
            0);

        check_cuda_res(res, "cuLaunchKernel");
        cuCtxSynchronize();
    }
}

auto CudaDriver::get_bk_prt() -> Neon::Backend* { return &backend; }

}

extern "C" int cuda_driver_new(void** handle, void* bk_handle)
{
    NEON_PY_PRINT_BEGIN((*handle));
    auto* backendPtr = reinterpret_cast<Neon::Backend*>(bk_handle);
    auto cuda_driver = new(std::nothrow) Neon::py::CudaDriver(backendPtr);
    (*handle) = reinterpret_cast<void*>(cuda_driver);
    NEON_PY_PRINT_END((*handle));

    return 0;
}

extern "C" int cuda_driver_delete(void** handle)
{
    NEON_PY_PRINT_BEGIN((*handle));;
    auto* cuda_driver_ptr = reinterpret_cast<Neon::py::CudaDriver*>(*handle);

    if (cuda_driver_ptr != nullptr) {
        delete cuda_driver_ptr;
    }
    *handle = nullptr;
    NEON_PY_PRINT_END((*handle));
    return 0;
}