#pragma once
#include "Neon/sys/devices/gpu/GpuSys.h"
#include "libneonsys_export.h"

namespace Neon {
namespace sys {
namespace globalSpace {

LIBNEONSYS_EXPORT extern GpuSys gpuSysObjStorage;

GpuSys& gpuSysObj();

}  // namespace globalSpace
}  // End of namespace sys
}  // End of namespace Neon