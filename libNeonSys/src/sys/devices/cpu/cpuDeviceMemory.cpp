#include "Neon/core/core.h"
#include "Neon/sys/devices/cpu/CpuDevice.h"


#include "cuda.h"
#include "cuda_runtime.h"

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

void* CpuDev::memory_t::mallocByte(size_t allocationSize)
{
    void* mem = malloc(allocationSize);
    if (nullptr == mem) {
        NeonException exc;
        exc << "Error completing malloc operation: "
            << "\n   memory size        " << allocationSize;
        NEON_THROW(exc);
    }
    return mem;
}

void CpuDev::memory_t::free(void* pointer)
{
    ::free(pointer);
}

void* CpuDev::memory_t::mallocCudaHostByte(size_t allocationSize)
{
    void* mem = nullptr;

    cudaError_t res = cudaHostAlloc((void**)&mem, allocationSize, cudaHostAllocPortable);

    if (res != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaMallocHost operation: "
            << "\n   memory size        " << allocationSize;
        exc << "\n Error: " << cudaGetErrorString(res);
        NEON_THROW(exc);
    }

    return mem;
}

void CpuDev::memory_t::freeCudaHostByte(void* pointer)
{

    cudaError_t res = cudaFreeHost(pointer);

    if (res != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaFreeHost operation: "
            << "\n   Pointer        " << pointer;
        exc << "\n Error: " << cudaGetErrorString(res);
        NEON_THROW(exc);
    }
}

void* CpuDev::memory_t::mallocCudaManagedByte(size_t allocationSize)
{
    void* mem = nullptr;

    cudaError_t res = cudaMallocManaged((void**)&mem, allocationSize);

    if (res != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaMallocManaged operation: "
            << "\n   memory size        " << allocationSize;
        exc << "\n Error: " << cudaGetErrorString(res);
        NEON_THROW(exc);
    }

    return mem;
}

void CpuDev::memory_t::freeCudaManagedByte(void* pointer)
{

    cudaError_t res = cudaFree(pointer);

    if (res != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaFree operation on unified memory: "
            << "\n   Pointer        " << pointer;
        exc << "\n Error: " << cudaGetErrorString(res);
        NEON_THROW(exc);
    }
}
}  // End of namespace sys
}  // End of namespace Neon
