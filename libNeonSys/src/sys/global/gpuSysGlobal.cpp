#include "Neon/sys/global/GpuSysGlobal.h"
#include "libneonsys_export.h"

namespace Neon {
namespace sys {
namespace globalSpace {

LIBNEONSYS_EXPORT GpuSys gpuSysObjStorage;

GpuSys& gpuSysObj()
{
    return gpuSysObjStorage;
}

}  // namespace globalSpace
}  // namespace sys
}  // End of namespace Neon
