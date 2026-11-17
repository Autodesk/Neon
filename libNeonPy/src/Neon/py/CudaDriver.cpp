#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <limits>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "Neon/Neon.h"
#include "Neon/core/core.h"
#include "Neon/py/CudaDriver.h"
#include "Neon/py/macros.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"

namespace Neon::py {

namespace {

// RAII guard so the caller's CUDA context is restored even if launch fails.
struct CtxPushGuard
{
    bool pushed = false;

    explicit CtxPushGuard(CUcontext ctx)
    {
        CUresult res = cuCtxPushCurrent(ctx);
        if (res != CUDA_SUCCESS) {
            const char* errorName = nullptr;
            const char* errorString = nullptr;
            cuGetErrorName(res, &errorName);
            cuGetErrorString(res, &errorString);
            Neon::NeonException e("CudaDriverEntryPoint");
            e << "cuCtxPushCurrent failed with \n" << res << " ";
            e << (errorName ? errorName : "?") << "\n";
            e << (errorString ? errorString : "?");
            NEON_THROW(e);
        }
        pushed = true;
    }

    ~CtxPushGuard()
    {
        if (!pushed) {
            return;
        }
        CUcontext popped = nullptr;
        CUresult  res = cuCtxPopCurrent(&popped);
        if (res != CUDA_SUCCESS) {
            const char* errorName = nullptr;
            const char* errorString = nullptr;
            cuGetErrorName(res, &errorName);
            cuGetErrorString(res, &errorString);
            std::cerr << "[Neon][CudaDriver] cuCtxPopCurrent failed: "
                      << (errorName ? errorName : "?") << " "
                      << (errorString ? errorString : "?") << std::endl;
        }
    }

    CtxPushGuard(const CtxPushGuard&) = delete;
    CtxPushGuard& operator=(const CtxPushGuard&) = delete;
};

}  // namespace

CudaDriver::CudaDriver(Neon::Backend* bk_prt)
{
    Neon::init();
    this->backend = *bk_prt;
    cu_devices = backend.newDataSet<CUdevice>();
    cu_contexts = backend.newDataSet<CUcontext>();

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

        // Share Warp's primary context. A private cuCtxCreate context becomes
        // current, gets registered by Warp as cuda:N.M, and then cuCtxDestroy
        // leaves Warp holding a destroyed context (issue 3 / suite SIGSEGV).
        res = cuDevicePrimaryCtxRetain(&cu_contexts[setIdx], cu_devices[setIdx]);
        check_cuda_res(res, std::string("cuDevicePrimaryCtxRetain (dev ") + std::to_string(cuda_dev_idx) + std::string(")"));
    }
}

CudaDriver::~CudaDriver() noexcept
{
    int  ndevs = backend.getDeviceCount();
    auto devSet = backend.devSet();
    for (int setIdx = 0; setIdx < ndevs; setIdx++) {
        Neon::sys::DeviceID devId = devSet.devId(setIdx);
        int                 cuda_dev_idx = devId.idx();

        // Release our retain only. Warp (and any other retainer) keeps the
        // primary context alive; this must not throw from a destructor.
        CUresult res = cuDevicePrimaryCtxRelease(cu_devices[setIdx]);
        if (res != CUDA_SUCCESS) {
            const char* errorName = nullptr;
            const char* errorString = nullptr;
            cuGetErrorName(res, &errorName);
            cuGetErrorString(res, &errorString);
            std::cerr << "[Neon][CudaDriver] cuDevicePrimaryCtxRelease (dev "
                      << cuda_dev_idx << ") failed: "
                      << (errorName ? errorName : "?") << " "
                      << (errorString ? errorString : "?") << std::endl;
        }
        cu_contexts[setIdx] = nullptr;
    }
}


auto CudaDriver::run_kernel(
    Neon::set::DataSet<kernel> const&  kernelSet,
    Neon::set::LaunchParameters const& launch_params,
    Neon::StreamIdx                    streamIdx) -> void
{
    [[maybe_unused]] auto& streamSet = backend.streamSet(streamIdx);
    int const              ndevs = backend.getDeviceCount();
#pragma omp parallel for num_threads(ndevs)
    for (int setIdx = 0; setIdx < ndevs; setIdx++) {
        backend.devSet().setActiveDevContext(setIdx);

        cudaStream_t const& cuda_stream = streamSet.cudaStream(setIdx);
        CUstream            driverStream = (CUstream)cuda_stream;
        CUfunction          function = static_cast<CUfunction>(kernelSet[setIdx]);
        auto&               launch_info = launch_params[setIdx];

        // An empty partition (e.g. an mGrid level with no active blocks) has a
        // zero-sized grid, which cuLaunchKernel rejects with CUDA_ERROR_INVALID_VALUE.
        auto const launch_grid = launch_info.cudaGrid();
        if (launch_grid.x * launch_grid.y * launch_grid.z == 0) {
            continue;
        }

        // Push/pop so Neon does not leave the primary context current on the
        // calling thread after the launch (OpenMP may run this on the main thread).
        CtxPushGuard ctx_guard(cu_contexts[setIdx]);

        // should match warp's launch_bounds_t<1> (types.py / builtin.h)
        struct launch_bounds_t
        {
            int    shape[1];    // size of each dimension
            size_t size;        // total number of threads
            size_t coord_mult;  // threads sharing each coord tuple
        };
        auto      cuda_block = launch_info.cudaBlock();
        auto      cuda_grid = launch_info.cudaGrid();
        int const n = cuda_block.x * cuda_grid.x;

        launch_bounds_t bounds{};
        bounds.shape[0] = n;
        bounds.size = n;
        bounds.coord_mult = 1;

        std::vector<void*> args;
        args.push_back(&bounds);

        CUresult res = cuLaunchKernel(
            function,
            launch_info.cudaGrid().x,
            launch_info.cudaGrid().y,
            launch_info.cudaGrid().z,
            launch_info.cudaBlock().x,
            launch_info.cudaBlock().y,
            launch_info.cudaBlock().z,
            0,
            driverStream,
            args.data(),
            0);

        check_cuda_res(res, "cuLaunchKernel");
    }
}

auto CudaDriver::get_bk_prt() -> Neon::Backend*
{
    return &backend;
}

}  // namespace Neon::py

extern "C" int cuda_driver_new(void** handle, void* bk_handle)
{
    NEON_PY_PRINT_BEGIN((*handle));
    auto* backendPtr = reinterpret_cast<Neon::Backend*>(bk_handle);
    auto  cuda_driver = new (std::nothrow) Neon::py::CudaDriver(backendPtr);
    (*handle) = reinterpret_cast<void*>(cuda_driver);
    NEON_PY_PRINT_END((*handle));

    return 0;
}

extern "C" int cuda_driver_delete(void** handle)
{
    NEON_PY_PRINT_BEGIN((*handle));
    ;
    auto* cuda_driver_ptr = reinterpret_cast<Neon::py::CudaDriver*>(*handle);

    if (cuda_driver_ptr != nullptr) {
        delete cuda_driver_ptr;
    }
    *handle = nullptr;
    NEON_PY_PRINT_END((*handle));
    return 0;
}
