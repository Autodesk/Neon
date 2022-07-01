#include <ctype.h>

#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/GpuDevice.h"
#include "Neon/sys/devices/gpu/GpuTools.h"

#if defined(NEON_OS_WINDOWS)
#include "windows.h"
#elif defined(NEON_OS_LINUX)
#include "sys/sysinfo.h"
#include "sys/types.h"
#else  // defined(NEON_OS_MAC)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/mach_init.h>
#include <mach/mach_types.h>
#include <mach/vm_statistics.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <ctype.h>


namespace Neon {
namespace sys {


/**
 * Follows the order defined by GpuDev::tools_t::arch
 */
std::vector<std::string> deviceToolsArchNames = {"Older",
                                                 "Kepler",
                                                 "Maxwell",
                                                 "Pascal",
                                                 "Volta",
                                                 "Turing",
                                                 "Ampere",
                                                 "NextGen"};

#include <map>
static std::map<int, GpuDevice::tools_t::arch_e> deviceToolsArchCompute_from_local_to_arch = {
    {30, GpuDevice::tools_t::Kepler},
    {35, GpuDevice::tools_t::Kepler},
    {37, GpuDevice::tools_t::Kepler},

    {50, GpuDevice::tools_t::Maxwell},
    {52, GpuDevice::tools_t::Maxwell},
    {53, GpuDevice::tools_t::Maxwell},

    {60, GpuDevice::tools_t::Pascal},
    {61, GpuDevice::tools_t::Pascal},
    {62, GpuDevice::tools_t::Pascal},

    {70, GpuDevice::tools_t::Volta},
    {72, GpuDevice::tools_t::Volta},

    {75, GpuDevice::tools_t::Turing},

    {80, GpuDevice::tools_t::Ampere},
    {86, GpuDevice::tools_t::Ampere},
};
/**
 * Follows the order defined by GpuDev::tools_t::arch
 * https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
 */
std::vector<int> deviceToolsArchCompute = {-1,
                                           30,
                                           50,
                                           60,
                                           70,
                                           70,
                                           80,
                                           -1};

GpuDevice::tools_t::arch_e GpuDevice::tools_t::arch() const
{
    const int majour = majorComputeCapability();
    const int minor = majorComputeCapability();

    const int full = majour * 10 + minor;

    const auto                 it = deviceToolsArchCompute_from_local_to_arch.find(full);
    GpuDevice::tools_t::arch_e res = GpuDevice::tools_t::arch_e::OldGen;

    if (it != deviceToolsArchCompute_from_local_to_arch.end()) {
        res = it->second;
    } else {
        if (full > deviceToolsArchCompute[GpuDevice::tools_t::arch_e::Kepler]) {
            res = GpuDevice::tools_t::arch_e::NextGen;
        } else {
            res = GpuDevice::tools_t::arch_e::OldGen;
        }
    }
    return res;
}

GpuStream GpuDevice::tools_t::stream() const
{
    this->gpuDev.tools.setActiveDevContext();
    ;
    cudaStream_t tmpStream;
    cudaError_t  err = cudaStreamCreate(&tmpStream);
    if (err != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaStreamCreate operation. ";
        exc << "\n Error: " << cudaGetErrorString(err);
        NEON_THROW(exc);
    }
    GpuStream stream(this->gpuDev.idx, tmpStream);

    return stream;
}

void GpuDevice::tools_t::sync() const
{
    this->gpuDev.tools.setActiveDevContext();
    ;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaDeviceSynchronize operation. ";
        exc << "\n Error: " << cudaGetErrorString(err);
        NEON_THROW(exc);
    }
}


GpuEvent GpuDevice::tools_t::event(bool disableTiming) const
{
    this->gpuDev.tools.setActiveDevContext();
    ;
    cudaEvent_t tmpStream;
    cudaError_t err;
    ;
    if (!disableTiming) {
        err = cudaEventCreate(&tmpStream);
    } else {
        err = cudaEventCreateWithFlags(&tmpStream, cudaEventDisableTiming);
    }
    if (err != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaStreamCreate operation. ";
        exc << "\n Error: " << cudaGetErrorString(err);
        NEON_THROW(exc);
    }
    GpuEvent event(this->gpuDev.idx, tmpStream);

    return event;
}

cublasHandle_t GpuDevice::tools_t::cublasHandle(bool device_pointer_mode) const
{
    this->gpuDev.tools.setActiveDevContext();
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        NeonException exc;
        exc << "cuBLAS error creating handle with error:" << Neon::sys::cublasGetErrorString(status);
        NEON_THROW(exc);
    }

    if (device_pointer_mode) {
        status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    } else {
        status = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        NeonException exc;
        exc << "cuBLAS error setting pointer mode with error: " << Neon::sys::cublasGetErrorString(status);
        NEON_THROW(exc);
    }
    return handle;
}

void GpuDevice::tools_t::streamDestroy(GpuStream& stream) const
{
    this->gpuDev.tools.setActiveDevContext();
    ;
    cudaError_t err = cudaStreamDestroy(stream.stream());
    if (err != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaStreamDestroy operation. ";
        exc << "\n Error: " << cudaGetErrorString(err);
        NEON_THROW(exc);
    }
}

void GpuDevice::tools_t::eventDestroy(GpuEvent& event) const
{
    this->gpuDev.tools.setActiveDevContext();
    ;
    cudaError_t err = cudaEventDestroy(event.event());
    if (err != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaEventDestroy operation. ";
        exc << "\n Error: " << cudaGetErrorString(err);
        NEON_THROW(exc);
    }
}


GpuDevice::tools_t::tools_t(GpuDevice& gpuDev)
    : gpuDev(gpuDev) {}

void GpuDevice::tools_t::setActiveDevContext() const
{
    cudaError_t err = cudaSetDevice(this->gpuDev.getIdx().idx());
    if (err != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaSetDevice operation: "
            << "device id " << gpuDev.getIdx();
        exc << "\n Error: " << cudaGetErrorString(err);
        NEON_THROW(exc);
    }
}

int GpuDevice::tools_t::majorComputeCapability() const
{
    this->setActiveDevContext();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuDev.getIdx().idx());
    int majorCompute = prop.major;
    return majorCompute;
}

int GpuDevice::tools_t::minorComputeCapability() const
{
    this->setActiveDevContext();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuDev.getIdx().idx());
    int majorCompute = prop.minor;
    return majorCompute;
}

cudaDeviceProp GpuDevice::tools_t::getDeviceProp() const
{
    this->setActiveDevContext();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuDev.getIdx().idx());
    return prop;
}

std::string GpuDevice::tools_t::getDevName() const
{
    this->setActiveDevContext();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuDev.getIdx().idx());
    std::string name(prop.name);
    name.erase(std::remove(name.begin(), name.end(), ' '),
               name.end());
    //    name.erase(std::remove_if(name.begin(), name.end(), std::isspace), name.end());
    return name;
}

std::string GpuDevice::tools_t::getDevInfo(const std::string& prefix) const
{
    this->setActiveDevContext();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuDev.getIdx().idx());
    std::ostringstream msg;
    const std::string  newLineAndPrefix = "\n" + prefix;
    msg << prefix << "Device Number: " << gpuDev.getIdx();
    msg << newLineAndPrefix << "    Device name:                  " << prop.name;
    std::vector<std::string> cudaComputeModeStrVec = {"Default", "Compute-exclusive-thread", "Compute-prohibited", "Compute-exclusive-process"};
    msg << newLineAndPrefix << "    Compute Mode:                 " << cudaComputeModeStrVec[prop.computeMode];
    msg << newLineAndPrefix << "    Unified addr space with host: " << std::boolalpha << (bool)prop.unifiedAddressing;

    msg << newLineAndPrefix << "    Support for managed mem:      " << std::boolalpha << (bool)prop.managedMemory;
    msg << newLineAndPrefix << "    Memory Clock Rate (KHz):      " << prop.memoryClockRate;
    msg << newLineAndPrefix << "    Memory Bus Width (bits):      " << prop.memoryBusWidth;
    msg << newLineAndPrefix << "    Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6;
    msg << newLineAndPrefix << "    ECC enabled:                  " << std::boolalpha << (bool)prop.ECCEnabled;
    msg << newLineAndPrefix << "    Coherently accessing pageable mem without calling cudaHostRegister: " << std::boolalpha << (bool)prop.pageableMemoryAccess;

    return msg.str();
}

}  // namespace sys
}  // namespace Neon