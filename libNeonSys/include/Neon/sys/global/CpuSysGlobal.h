#pragma once
#include "Neon/sys/devices/cpu/CpuSys.h"
#include "libneonsys_export.h"

namespace Neon {
namespace sys {
namespace globalSpace {
LIBNEONSYS_EXPORT extern CpuSys cpuSysObjStorage;

CpuSys& cpuSysObj();

}  // namespace globalSpace
}  // namespace sys
}  // namespace Neon
