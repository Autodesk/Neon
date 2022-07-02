#include "Neon/set/GpuEventSet.h"
#include "Neon/set/GpuStreamSet.h"

#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/ComputeID.h"
#include "Neon/sys/global/GpuSysGlobal.h"

#include <vector>

namespace Neon {
namespace set {


GpuEventSet::GpuEventSet(int setCardinality)
{
    eventVec = std::vector<Neon::sys::GpuEvent>(setCardinality);
}

void GpuEventSet::release()
{
    for (auto&& cudaStream : eventVec) {
        cudaStream.release();
    }
}

}  // namespace set
}  // End of namespace Neon
