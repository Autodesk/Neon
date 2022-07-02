#pragma once


#include "Neon/core/core.h"
#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/devices/cpu/CpuDevice.h"
#include "Neon/sys/devices/memType.h"
#include "Neon/sys/memory/CpuMem.h"

#include <vector>


namespace Neon {
namespace sys {

class CpuMem;


class CpuSys
{
   public:
    std::vector<CpuDev> m_cpuDevVec;  // Devices...
    std::vector<CpuMem> m_cpuMemVec;  // Allocators ....

   public:
    /**
     * Constructor 
     */
    CpuSys();

    /**
     * Returns a reference to the selected device
     */
    const CpuDev& dev() const;

    /**
     * Detect the CPU in the system, prints some info and initialize device objects      
     */
    void init();

    /**
     * Returns a reference to the selected device
     */
    CpuMem& allocator();

    /**
     * Returns number of device 
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
