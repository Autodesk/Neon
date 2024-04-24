#pragma once
#if !defined(NEON_WARP_COMPILATION)

 #include <vector>
#include "Neon/set/Transfer.h"

namespace Neon {
namespace set {
class DevSet;
}
/**
 * Abstraction to represent memory options in a io-compute environment.
 */
struct MemoryOptions
{
   public:
    friend class Backend;
    friend class Neon::set::DevSet;

    /**
     * Creation of a non initialized MemoryOptions object.
     * An initialized MemoryOption can be retrieved through the backend.
     */
    MemoryOptions();

    /**
     * Creation of a non initialized MemoryOptions object.
     * An initialized MemoryOption can be retrieved through the backend.
     */
     MemoryOptions(Neon::MemoryLayout order);

    /**
     * Constructor that defines the type of devices, allocators, and memory order
     */
    MemoryOptions(Neon::DeviceType   hostType,
                  Neon::Allocator    hostAllocator,
                  Neon::DeviceType   deviceType,
                  Neon::Allocator    deviceAllocators,
                  Neon::MemoryLayout order);

    /**
     * Returns the compute type.
     */
    auto getDeviceType() const
        -> Neon::DeviceType;

    /**
     * Returns the io device type
     */
    auto getHostType() const
        -> Neon::DeviceType;

    /**
     * Returns the allocator type for compute
     * @return
     */
    auto getDeviceAllocator(Neon::DataUse dataUse = Neon::DataUse::HOST_DEVICE) const
        -> Neon::Allocator;

    /**
     * Returns the allocator type for io
     */
    auto getIOAllocator(Neon::DataUse dataUse = Neon::DataUse::HOST_DEVICE) const
        -> Neon::Allocator;

    /**
     * Returns the defined order
     */
    auto getOrder() const
        -> Neon::MemoryLayout;

    /**
     * Set the layout for the memory
     */
    auto setOrder(Neon::MemoryLayout)
        -> void;

   private:
    /**
     * Helper method to check if the object was initialized by the backend
     * @return
     */
    auto helpWasInitCompleted() const -> bool;

    /**
     * help method that throws an exception if the object was not initialized by the backend
     * @return
     */
    auto helpThrowExceptionIfInitNotCompleted() const -> void;

    /**
     * Private constructor to be used by the backend
     */
    MemoryOptions(Neon::DeviceType   ioType,
                  Neon::DeviceType   computeType,
                  Neon::MemoryLayout order);

    /**
     * Private constructor to be used by the backend
     */
    MemoryOptions(Neon::DeviceType   ioType,
                  Neon::Allocator    ioAllocator,
                  Neon::DeviceType   computeType,
                  Neon::Allocator    computeAllocators[Neon::DeviceTypeUtil::nConfig],
                  Neon::MemoryLayout order);


    Neon::Allocator    mDeviceAllocator = Neon::Allocator::NULL_MEM /** Device allocator type */;
    Neon::DeviceType   mDeviceType = Neon::DeviceType::NONE /** Device device type */;
    Neon::DeviceType   mHostType = Neon::DeviceType::NONE /** Host device type */;
    Neon::Allocator    mHostAllocator = Neon::Allocator::NULL_MEM /** Host allocator type */;
    Neon::MemoryLayout mMemOrder = Neon::MemoryLayout::structOfArrays /** Memory order */;
};
}  // namespace Neon

#endif