#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/GpuDevice.h"

#if defined(NEON_OS_WINDOWS)
#include "windows.h"
#elif defined(NEON_OS_LINUX)
#include "sys/sysinfo.h"
#include "sys/types.h"
#else  // defined(NEON_OS_MAC)
#include <mach/mach.h>

#include <sys/param.h>
#include <sys/types.h>
#include <unistd.h>

#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_types.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace Neon {
namespace sys {

void GpuDevice::memory_t::enablePeerAccsessWith(ComputeID gpuId) const
{
    gpuDev.tools.setActiveDevContext();
    cudaError_t res = cudaDeviceEnablePeerAccess(gpuId.idx(), 0);
    if (res != cudaSuccess) {
        if (cudaErrorInvalidDevice == res) {
            NeonException exc;
            exc << "CUDA error completing cudaDeviceEnablePeerAccess operation: "
                << "CUDA Dev " << gpuDev.getIdx() << " can not read/write from/to Cuda device " << gpuId;
            NEON_THROW(exc);
            return;
        }

        if (cudaErrorPeerAccessAlreadyEnabled == res) {
            cudaGetLastError();  //so that we don't see this error when we do cuda error checking
            //NEON_WARNING("GpuDev: CUDA device {} is already enabled for peer access w.r.t. CUDA device, {}", gpuDev.getIdx().idx(), gpuId.idx());
            return;
        }

        if (cudaErrorPeerAccessUnsupported == res) {
            cudaGetLastError();  //so that we don't see this error when we do cuda error checking
            NEON_WARNING("GpuDev: CUDA error completing cudaDeviceEnablePeerAccess operation. CUDA device {} does not support peer access {}", gpuDev.getIdx().idx(), gpuId.idx());
            return;
        }

        NeonException exc;
        exc << "CUDA error completing cudaDeviceEnablePeerAccess operation: "
            << "CUDA Dev " << gpuDev.getIdx()
            << " with error" << cudaGetErrorString(res);
        NEON_THROW(exc);
        return;
    }
}

void GpuDevice::memory_t::peerTransfer(const GpuStream& gpuStream, ComputeID dstDevId, char* dest, ComputeID srcDevId, const char* src, size_t numBytes) const
{
    if (dstDevId != gpuDev.idx && srcDevId != gpuDev.idx) {
        Neon::NeonException exc("GpuDev::Memory");
        exc << "In order to successfully call intraGpuTransfer, one between source or destination gpu_id has to match the id of the used GpuDeviceobject.";
        NEON_THROW(exc);
    }
    gpuDev.tools.setActiveDevContext();
    cudaError_t res = cudaMemcpyPeerAsync(dest, dstDevId.idx(), src, srcDevId.idx(), numBytes, gpuStream.stream());

    if (res != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaMemcpyPeerAsync operation: "
            << "\n   dst GPU         " << dstDevId.idx() << " addr:" << (void*)(dest)
            << "\n   src GPU         " << srcDevId.idx() << " addr:" << (void*)(src)
            << "\n   transfer Size:  " << numBytes;
        exc << "\n Error: " << cudaGetErrorString(res);

        NEON_THROW(exc);
    }
}

void GpuDevice::memory_t::memSet(void* mem, uint8_t val, size_t size) const
{
    gpuDev.tools.setActiveDevContext();
    cudaError_t err = cudaMemset((void*)(mem), val, size);
    if (err != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaMemset operation. ";
        exc << "\n Error: " << cudaGetErrorString(err);
        NEON_THROW(exc);
    }
    gpuDev.tools.sync();
}


void* GpuDevice::memory_t::mallocByte(size_t size) const
{
    void*        mem = nullptr;
    const size_t allocationSize = size;

    gpuDev.tools.setActiveDevContext();
    cudaError_t res = cudaMalloc((void**)&mem, allocationSize);

    if (res != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaMalloc operation: "
            << "\n   target GPU         " << gpuDev.getIdx()
            << "\n   memory size        " << allocationSize;
        exc << "\n Error: " << cudaGetErrorString(res);

        NEON_THROW(exc);
    }

    return mem;
}


///**
// * Returns the L1 cache line size. This info is not available from querying the device.
// * Information coming from paper and TR on micro-benchmarking
// * @return
// */
//int GpuDev::memory_t::nByteL1cacheLine(){
//
//    NEON_DEV_UNDER_CONSTRUCTION("memory_t");
//}
//
///**
// * Returns the L1 cache line size. This info is not available from querying the device.
// * Information coming from paper and TR on micro-benchmarking
// * @return
// */
//int GpuDev::memory_t::nBytePage(){
//    NEON_DEV_UNDER_CONSTRUCTION("memory_t");
//}

}  // namespace sys
}  // End of namespace Neon