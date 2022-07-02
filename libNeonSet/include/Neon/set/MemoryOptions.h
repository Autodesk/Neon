#pragma once
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
     * Constructor that defines the type of devices, allocators, and memory order 
     */
    MemoryOptions(Neon::DeviceType   ioType,
                  Neon::Allocator    ioAllocator,
                  Neon::DeviceType   computeType,
                  Neon::Allocator    computeAllocators,
                  Neon::MemoryLayout order);

    /**
     * Returns the compute type.
     */
    auto getComputeType() const
        -> Neon::DeviceType;

    /**
     * Returns the io device type
     */
    auto getIOType() const
        -> Neon::DeviceType;

    /**
     * Returns the allocator type for compute
     * @return
     */
    auto getComputeAllocator(Neon::DataUse dataUse = Neon::DataUse::IO_COMPUTE) const
        -> Neon::Allocator;

    /**
     * Returns the allocator type for io
     */
    auto getIOAllocator(Neon::DataUse dataUse = Neon::DataUse::IO_COMPUTE) const
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


    Neon::Allocator    mComputeAllocator = Neon::Allocator::NULL_MEM /** Compute allocator type */;
    Neon::DeviceType   mComputeType = Neon::DeviceType::NONE /** Compute device type */;
    Neon::DeviceType   mIOType = Neon::DeviceType::NONE /** IO device type */;
    Neon::Allocator    mIOAllocator = Neon::Allocator::NULL_MEM /** IO allocator type */;
    Neon::MemoryLayout mMemOrder = Neon::MemoryLayout::structOfArrays /** Memory order */;
};
}  // namespace Neon