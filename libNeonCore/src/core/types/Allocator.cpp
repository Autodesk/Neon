#include "Neon/core/types/Allocator.h"
#include "Neon/core/core.h"

#include <string>
#include <vector>

namespace Neon {

static std::vector<const char*> allocTypeStrings = {"ERROR",
                                                    "CUDA_UNIFIED",
                                                    "CUDA_DEVICE",
                                                    "CUDA_HOST",
                                                    "HWLOC",
                                                    "MALLOC",
                                                    "NULL_MEM",
                                                    "MANAGED",
                                                    "MIXED_MEM"};

auto AllocatorUtils::toString(Allocator allocator) -> const char*
{
    return allocTypeStrings[static_cast<unsigned long>(allocator)];
}

auto AllocatorUtils::compatible(Neon::DeviceType devEt, Neon::Allocator type) -> bool
{
    if (type == Neon::Allocator::NULL_MEM) {
        return true;
    }
    switch (devEt) {
        case Neon::DeviceType::OMP:
        case Neon::DeviceType::CPU: {
            switch (type) {
                case Neon::Allocator::MALLOC:
                case Neon::Allocator::CUDA_MEM_HOST:
                case Neon::Allocator::CUDA_MEM_UNIFIED: {
                    return true;
                }
                default: {
                    return false;
                }
            }
        }
        case Neon::DeviceType::CUDA: {
            switch (type) {
                case Neon::Allocator::CUDA_MEM_DEVICE:
                case Neon::Allocator::CUDA_MEM_UNIFIED: {
                    return true;
                }
                default: {
                    return false;
                }
            }
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

auto AllocatorUtils::getDefault(Neon::DeviceType devEt) -> Allocator
{
    switch (devEt) {
        case Neon::DeviceType::OMP:
        case Neon::DeviceType::CPU:
            return Allocator::MALLOC;
        case Neon::DeviceType::CUDA:
            return Allocator::CUDA_MEM_DEVICE;
        case Neon::DeviceType::NONE:
            return Allocator::NULL_MEM;
        case Neon::DeviceType::MPI:
            return Allocator::NULL_MEM;
        default:
            NEON_THROW_UNSUPPORTED_OPERATION("");
    }
}


std::ostream& operator<<(std::ostream& os, Allocator const& m)
{
    return os << std::string(AllocatorUtils::toString(m));
}

}  // End of namespace Neon
