#include "Neon/sys/devices/cpu/CpuSys.h"
#include "Neon/core/core.h"
#include "Neon/sys/devices/cpu/CpuDevice.h"

namespace Neon {
namespace sys {


CpuSys::CpuSys()
    : mInit(false)
{
}

void CpuSys::init()
{
    mInit = true;
    std::ostringstream msg;

    this->m_cpuDevVec.emplace_back();
    this->m_cpuMemVec.emplace_back(this->m_cpuDevVec[0]);

    NEON_INFO("CpuSys_t: Loading info on CPU subsystem");
}

const CpuDev& CpuSys::dev() const
{
    const CpuDev& ret = this->m_cpuDevVec[0];

    return ret;
}

CpuMem& CpuSys::allocator()
{
    CpuMem& ret = this->m_cpuMemVec[0];

    return ret;
}

int32_t CpuSys::numDevs() const
{
    return static_cast<int32_t>(m_cpuMemVec.size());
}

bool CpuSys::isInit() const
{
    return mInit;
}

}  // namespace sys
}  // End of namespace Neon