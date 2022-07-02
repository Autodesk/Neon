#pragma once


#include "Neon/core/core.h"
#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/devices/gpu/GpuDevice.h"
#include "Neon/sys/devices/memType.h"

#include <vector>

namespace Neon {
namespace sys {

class GpuMem;

class GpuSys
{
   private:
    int32_t                                 m_gpuCount;
    std::vector<std::shared_ptr<GpuDevice>> m_gpuDevVec;  // Devices...
    std::vector<std::shared_ptr<GpuMem>>    m_gpuMemVec;  // Allocators ....
   public:
    /**
     * Constructor
     */
    GpuSys();

    /**
     * Returns a reference to the selected device
     */
    const GpuDevice& dev(ComputeID) const;

    /**
     * Detects the GPU in the system, prints some info and initialize device objects for each device.
     */
    void init();

    /**
     * Returns a reference to the selected device
     */
    GpuMem& allocator(ComputeID);
    /**
     * Returns a reference to the selected device
     */
    int32_t numDevs() const;


    /**
    * Check if the system is initialized 
    */
    bool isInit() const;

   private:
    bool mInit;
};

}  // namespace sys
}  // End of namespace Neon
