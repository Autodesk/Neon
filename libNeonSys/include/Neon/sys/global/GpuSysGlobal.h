#pragma once
#include "Neon/sys/devices/gpu/GpuSys.h"
#include "libneonsys_export.h"

namespace Neon::sys::globalSpace {

LIBNEONSYS_EXPORT extern GpuSys gpuSysObjStorage;

GpuSys& gpuSysObj();

}  // namespace Neon::sys::globalSpace