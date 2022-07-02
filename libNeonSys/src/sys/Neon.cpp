#include "Neon/Neon.h"

namespace Neon {
void init()
{
    ::Neon::sys::globalSpace::cpuSysObjStorage.init();
    ::Neon::sys::globalSpace::gpuSysObjStorage.init();
}
}  // namespace Neon
