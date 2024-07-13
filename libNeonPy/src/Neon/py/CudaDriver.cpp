#include <dlfcn.h>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "Neon/Neon.h"
#include "Neon/core/core.h"
#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Loader.h"
#include "Neon/py/CudaDriver.h"

namespace Neon::py {
CudaDriver::CudaDriver(Neon::Backend* bk_prt)
    : bk_prt(bk_prt)
{
    Neon::init();
    Neon::Backend& bk = *bk_prt;
    cu_devices = bk.newDataSet<CUdevice>();
    cu_contexts = bk.newDataSet<CUcontext>();

    auto     dev_list = bk.devSet();
    CUresult res = cuInit(0);
    check_cuda_res(res, "cuInit");

    int const ndevs = bk.getDeviceCount();
    auto      devSet = bk.devSet();
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
    Neon::Backend& bk = *bk_prt;

    auto dev_list = bk.devSet();

    int  ndevs = bk.getDeviceCount();
    auto devSet = bk.devSet();
    for (int setIdx = 0; setIdx < ndevs; setIdx++) {
        Neon::sys::DeviceID devId = devSet.devId(setIdx);
        int                 cuda_dev_idx = devId.idx();

        CUresult res = cuCtxDestroy(cu_contexts[setIdx]);
        check_cuda_res(res, std::string("cuCtxCreate (dev ") + std::to_string(cuda_dev_idx) + std::string(")"));
    }
}


auto CudaDriver::run_kernel(
    Neon::set::DataSet<kernel> const&         kernelSet,
    Neon::set::LaunchParameters const& launch_params,
    Neon::StreamIdx                    streamIdx) -> void
{
    auto& bk = *this->bk_prt;
    auto& streamSet = bk.streamSet(streamIdx);

    int const ndevs = bk.getDeviceCount();
#pragma omp parallel for num_threads(ndevs)
    for (int setIdx = 0; setIdx < bk.getDeviceCount(); setIdx++) {
        cudaStream_t const& cuda_stream = streamSet.cudaStream(setIdx);
        CUstream            driverStream = (CUstream)cuda_stream;
        CUfunction          function = static_cast<CUfunction>(kernelSet[setIdx]);

        auto&      launch_info = launch_params[setIdx];
        auto const cudaGrid = launch_info.cudaGrid();
        auto const cudaBlock = launch_info.cudaBlock();
        // Set the created context as the current context
        CUresult res = cuCtxSetCurrent(cu_contexts[setIdx]);
        check_cuda_res(res, "cuCtxSetCurrent");

        res = cuLaunchKernel(
            function,
            cudaGrid.x, cudaGrid.y, cudaGrid.z,
            cudaBlock.x, cudaBlock.y, cudaBlock.z,
            0,
            driverStream,
            nullptr,
            0);

        check_cuda_res(res, "cuLaunchKernel");
    }
}

auto CudaDriver::get_bk_prt() -> Neon::Backend* { return bk_prt; }

}

extern "C" void cuda_driver_entry_point_new(uint64_t& handle, uint64_t* bk_handle)
{
    auto* backendPtr = reinterpret_cast<Neon::Backend*>(bk_handle);
    auto  cuda_driver_entry_point = new(std::nothrow) Neon::py::CudaDriver(backendPtr);
    handle = uint64_t(cuda_driver_entry_point);
}

extern "C" void cuda_driver_entry_point_delete(uint64_t& handle)
{
    auto* cuda_driver_entry_point_ptr = reinterpret_cast<Neon::py::CudaDriver*>(handle);

    if (cuda_driver_entry_point_ptr != nullptr) {
        delete cuda_driver_entry_point_ptr;
    }
    handle = 0;
}