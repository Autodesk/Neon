#include "Neon/sys/devices/cpu/CpuDevice.h"
#include "Neon/core/core.h"

#if defined(NEON_OS_WINDOWS)
#include "windows.h"
#elif defined(NEON_OS_LINUX)
#include <sys/sysinfo.h>
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
#include "string.h"
namespace Neon {
namespace sys {


CpuDev::CpuDev()
    : DeviceInterface(DeviceType::CPU)
{
}

double CpuDev::usage() const
{
    // TODO@[Max]("find a way to query device load in a platform unified way.")
    return -1;
}


int64_t CpuDev::virtMemory() const
{
// see https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
//-------------------------------------------------------------
#if defined(NEON_OS_MAC)
#endif
//-------------------------------------------------------------
#if defined(NEON_OS_LINUX)
    struct sysinfo memInfo;

    sysinfo(&memInfo);
    int64_t totalVirtualMem = memInfo.totalram;
    //Add other values in next statement to avoid int overflow on right hand side...
    totalVirtualMem += memInfo.totalswap;
    totalVirtualMem *= memInfo.mem_unit;
    return totalVirtualMem;
#endif
//-------------------------------------------------------------
#if defined(NEON_OS_WINDOWS)
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    DWORDLONG totalVirtualMem = memInfo.ullTotalPageFile;
    return totalVirtualMem;
#else
    NEON_WARNING("CpuDev_t: Unable to fully retrieve information on the system memory.");
    return 0;
#endif
}

int64_t CpuDev::usedVirtMemory() const
{
// see https://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
//-------------------------------------------------------------
#if defined(NEON_OS_MAC)
#endif
//-------------------------------------------------------------
#if defined(NEON_OS_LINUX)
    struct sysinfo memInfo;

    sysinfo(&memInfo);
    int64_t virtualMemUsed = memInfo.totalram - memInfo.freeram;
    //Add other values in next statement to avoid int overflow on right hand side...
    virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
    virtualMemUsed *= memInfo.mem_unit;
    return virtualMemUsed;
#endif
//-------------------------------------------------------------
#if defined(NEON_OS_WINDOWS)
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    DWORDLONG virtualMemUsed = memInfo.ullTotalPageFile - memInfo.ullAvailPageFile;
    return virtualMemUsed;
#else
    NEON_WARNING("CpuDev_t: Unable to fully retrieve information on the system memory.");
    return 0;
#endif
}


int64_t CpuDev::physMemory() const
{
//-------------------------------------------------------------
#if defined(NEON_OS_MAC)
    int64_t physical_memory = -1;

    int mib[2];
    mib[0] = CTL_HW;
    mib[1] = HW_MEMSIZE;
    size_t length = sizeof(int64_t);
    sysctl(
        mib, 2, &physical_memory, &length, NULL, 0);

    return physical_memory;
#endif
//-------------------------------------------------------------
#if defined(NEON_OS_LINUX)
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    long long totalPhysMem = memInfo.totalram;
    //Multiply in next statement to avoid int overflow on right hand side...
    totalPhysMem *= memInfo.mem_unit;

    return totalPhysMem;
#endif
//-------------------------------------------------------------
#if defined(NEON_OS_WINDOWS)
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    size_t totalPhysMem = memInfo.ullTotalPhys;
    return totalPhysMem;
#else
    NEON_WARNING("CpuDev_t: Unable to fully retrieve information on the system memory.");
    return 0;
#endif
}

int64_t CpuDev::usedPhysMemory() const
{
//-------------------------------------------------------------
#if defined(NEON_OS_MAC)
    vm_size_t              page_size;
    mach_port_t            mach_port;
    mach_msg_type_number_t count;
    vm_statistics64_data_t vm_stats;

    mach_port = mach_host_self();
    count = sizeof(vm_stats) / sizeof(natural_t);
    if (KERN_SUCCESS == host_page_size(mach_port, &page_size) &&
        KERN_SUCCESS == host_statistics64(mach_port, HOST_VM_INFO,
                                          (host_info64_t)&vm_stats, &count)) {
        //long long free_memory = (int64_t)vm_stats.free_count * (int64_t)page_size;

        long long used_memory = ((int64_t)vm_stats.active_count +
                                 (int64_t)vm_stats.inactive_count +
                                 (int64_t)vm_stats.wire_count) *
                                (int64_t)page_size;
        return used_memory;
    }
    NEON_WARNING("CpuDev_t: Unable to fully retrieve information on the system memory.");
    return 0;
#endif
//-------------------------------------------------------------
#if defined(NEON_OS_LINUX)
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    long long physMemUsed = memInfo.totalram - memInfo.freeram;
    //Multiply in next statement to avoid int overflow on right hand side...
    physMemUsed *= memInfo.mem_unit;

    return physMemUsed;
#endif
//-------------------------------------------------------------
#if defined(NEON_OS_WINDOWS)
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    DWORDLONG physMemUsed = memInfo.ullTotalPhys - memInfo.ullAvailPhys;
    return physMemUsed;
#else
    NEON_WARNING("CpuDev_t: Unable to fully retrieve information on the system memory.");
    return -1;
#endif
}


int64_t CpuDev::processUsedPhysMemory() const
{
//-------------------------------------------------------------
#if defined(NEON_OS_MAC)

    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;

    if (KERN_SUCCESS == task_info(mach_task_self(),
                                  TASK_BASIC_INFO, (task_info_t)&t_info,
                                  &t_info_count)) {
        return t_info.resident_size;
    }
    // resident size is in t_info.resident_size;
    // virtual size is in t_info.virtual_size;
#endif
//-------------------------------------------------------------
#if defined(NEON_OS_LINUX)

    FILE*   file = fopen("/proc/self/status", "r");
    int     result = -1;
    char    line[128];
    int64_t returnVal;

    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            result = this->parseLineGetByte(line);
            break;
        }
    }
    fclose(file);
    returnVal = result;
    returnVal *= 1024;
    return returnVal;
#endif
//-------------------------------------------------------------
#if defined(NEON_OS_WINDOWS)
    //PROCESS_MEMORY_COUNTERS_EX pmc;
    //GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    //int64_t physMemUsedByMe = pmc.WorkingSetSize;
    //return physMemUsedByMe;
#endif
    NEON_WARNING("CpuDev_t: Unable to fully retrieve information on the system memory.");
    return 0;
}

int64_t CpuDev::parseLineGetByte(char* line)
{
    // This assumes that a digit will be found and the line ends in " Kb".
    const size_t i = strlen(line);
    const char*  p = line;
    while (*p < '0' || *p > '9') {
        p++;
    }
    line[i - 3] = '\0';
    return atoi(p) * 1024;
}
}  // End of namespace sys
}  // End of namespace Neon
