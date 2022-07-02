#include "Neon/core/core.h"

#include <Neon/sys/devices/gpu/GpuDevice.h>

namespace Neon {
namespace sys {


GpuDevice::GpuDevice()
    : DeviceInterface(DeviceType::CUDA), tools(*this), memory(*this), kernel(*this)
{
}
GpuDevice::GpuDevice(const DeviceID& devIdx)
    : DeviceInterface(DeviceType::CUDA, devIdx), tools(*this), memory(*this), kernel(*this)
{
}
GpuDevice::GpuDevice(GpuDevice&& other)
    : DeviceInterface(DeviceType::CUDA, other.idx.idx()), tools(*this), memory(*this), kernel(*this)
{
}

double GpuDevice::usage() const
{
    size_t      free_t, total_t;
    cudaError_t res = ::cudaMemGetInfo(&free_t, &total_t);  //in bytes
    if (res != cudaSuccess) {
        NeonException exc;
        exc << "CUDA error completing cudaMemGetInfo operation: "
            << "\n\t target GPU         " << this->getIdx();
        exc << "\n\t Error: " << cudaGetErrorString(res);
        NEON_THROW(exc);
    }
    //double free_m = (double)free_t / (double)1048576.0;//in MB
    //double total_m = (double)total_t / (double)1048576.0;//in MB
    //double used_m = total_m - free_m;

    return static_cast<double>(total_t - free_t) / static_cast<double>(total_t);
}
}  // namespace sys
}  // End of namespace Neon
