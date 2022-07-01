#include "Neon/sys/devices/gpu/GpuSys.h"
#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/GpuDevice.h"
#include "Neon/sys/memory/GpuMem.h"

namespace Neon {
namespace sys {


GpuSys::GpuSys()
    : mInit(false)
{
}

void GpuSys::init()
{
    mInit = true;
    this->m_gpuCount = 0;
    cudaError_t res = cudaGetDeviceCount(&this->m_gpuCount);

    if (res != cudaSuccess) {
        NeonException exc("GpuSys_t");
        exc << "CUDA error completing cudaGetNumDevices operation (ret code " << res << ").";
        //NEON_THROW(exc);
    }
    std::ostringstream msg;

    msg << "Loading info on GPU subsystem ";
    if (this->m_gpuCount == 0) {
        msg << "\n    "
            << "No GPU were detected.";
    }
    if (this->m_gpuCount == 1) {
        msg << this->m_gpuCount << " GPU was detected.";
    }
    if (this->m_gpuCount > 1) {
        msg << this->m_gpuCount << " GPUs were detected.";
    }

    m_gpuDevVec.resize(this->m_gpuCount);
    m_gpuMemVec.resize(this->m_gpuCount);
    for (int devId = 0; devId < this->m_gpuCount; devId++) {
        m_gpuDevVec[devId] = std::make_shared<GpuDevice>(devId);
        m_gpuMemVec[devId] = std::make_shared<GpuMem>(*(m_gpuDevVec[devId]));
    }
    NEON_INFO("GpuSys_t: {}", msg.str());
}

const GpuDevice& GpuSys::dev(ComputeID gpuIdx) const
{
    if (this->m_gpuDevVec.empty()) {
        Neon::NeonException exp("GpuSys_t");
        exp << "GPU device set is empty so cannot retrieve reference to Gpu device with idx " << gpuIdx.idx() << "\n";
        exp << "No GPU device were previously detected." << this->m_gpuDevVec.size() - 1;
        NEON_THROW(exp);
    }
    if (gpuIdx.idx() < 0 || size_t(gpuIdx.idx()) >= this->m_gpuDevVec.size()) {
        Neon::NeonException exp("GpuSys_t");
        exp << "Unable to retrieve reference to Gpu device with idx " << gpuIdx.idx() << "\n";
        exp << "Max idx detected was " << this->m_gpuDevVec.size() - 1;
        NEON_THROW(exp);
    }
    const GpuDevice& ret = *(this->m_gpuDevVec[gpuIdx.idx()]);
    return ret;
}

GpuMem& GpuSys::allocator(ComputeID gpuIdx)
{
    if (this->m_gpuMemVec.empty()) {
        Neon::NeonException exp("GpuSys_t");
        exp << "GPU device set is empty so cannot retrieve reference to Gpu device with idx " << gpuIdx.idx() << "\n";
        exp << "No GPU device were previously detected." << this->m_gpuDevVec.size() - 1;
        NEON_THROW(exp);
    }
    if (gpuIdx.idx() < 0 || size_t(gpuIdx.idx()) >= this->m_gpuMemVec.size()) {
        Neon::NeonException exp("GpuSys_t");
        exp << "Unable to retrieve reference to Gpu device with idx " << gpuIdx.idx() << "\n";
        exp << "Max idx detected was" << this->m_gpuDevVec.size() - 1;
        NEON_THROW(exp);
    }

    GpuMem& ret = *(this->m_gpuMemVec[gpuIdx.idx()]);

    return ret;
}

int32_t GpuSys::numDevs() const
{
    return m_gpuCount;
}

bool GpuSys::isInit() const
{
    return mInit;
}

}  // namespace sys
}  // End of namespace Neon
