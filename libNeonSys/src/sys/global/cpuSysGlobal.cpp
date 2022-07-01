#include "Neon/sys/global/CpuSysGlobal.h"
#include "libneonsys_export.h"

namespace Neon {
namespace sys {
namespace globalSpace {

LIBNEONSYS_EXPORT CpuSys cpuSysObjStorage;

CpuSys& cpuSysObj()
{
    return cpuSysObjStorage;
}

}  // namespace globalSpace
}  // namespace sys
}  // End of namespace Neon
