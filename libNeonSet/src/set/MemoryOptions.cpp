#include "Neon/set/MemoryOptions.h"

namespace Neon {

MemoryOptions::MemoryOptions()
{
}

MemoryOptions::MemoryOptions(Neon::DeviceType   ioType,
                             Neon::DeviceType   computeType,
                             Neon::MemoryLayout order)
{
    mComputeType = computeType;
    mComputeAllocator = Neon::AllocatorUtils::getDefault(computeType);

    mIOType = ioType;
    mIOAllocator = Neon::AllocatorUtils::getDefault(ioType);

    mMemOrder = order;
}

MemoryOptions::MemoryOptions(Neon::DeviceType   ioType,
                             Neon::Allocator    ioAllocator,
                             Neon::DeviceType   computeType,
                             Neon::Allocator    computeAllocators,
                             Neon::MemoryLayout order)
{
    mIOType = ioType;
    mIOAllocator = ioAllocator;

    mComputeType = computeType;
    mComputeAllocator = computeAllocators;

    mMemOrder = order;
}

MemoryOptions::MemoryOptions(Neon::DeviceType   ioType,
                             Neon::Allocator    ioAllocator,
                             Neon::DeviceType   computeType,
                             Neon::Allocator    computeAllocators[Neon::DeviceTypeUtil::nConfig],
                             Neon::MemoryLayout order)
{
    mIOType = ioType;
    mIOAllocator = ioAllocator;

    mComputeType = computeType;
    mComputeAllocator = computeAllocators[Neon::DeviceTypeUtil::toInt(computeType)];

    mMemOrder = order;
}

auto MemoryOptions::getComputeType() const
    -> Neon::DeviceType
{
    helpThrowExceptionIfInitNotCompleted();
    return mComputeType;
}

auto MemoryOptions::getIOType() const
    -> Neon::DeviceType
{
    helpThrowExceptionIfInitNotCompleted();
    return mIOType;
}

auto MemoryOptions::getComputeAllocator(Neon::DataUse dataUse) const
    -> Neon::Allocator
{
    helpThrowExceptionIfInitNotCompleted();

    if (dataUse == Neon::DataUse::IO_POSTPROCESSING) {
        return Neon::Allocator::NULL_MEM;
    }
    if (getComputeType() == Neon::DeviceType::CPU) {
        return Neon::Allocator::NULL_MEM;
    }
    return mComputeAllocator;
}

auto MemoryOptions::getIOAllocator(Neon::DataUse dataUse) const
    -> Neon::Allocator
{
    helpThrowExceptionIfInitNotCompleted();

    if (dataUse == Neon::DataUse::COMPUTE) {
        return Neon::Allocator::NULL_MEM;
    }
    return mIOAllocator;
}

auto MemoryOptions::getOrder() const
    -> Neon::MemoryLayout
{
    helpThrowExceptionIfInitNotCompleted();
    return mMemOrder;
}

auto MemoryOptions::setOrder(Neon::MemoryLayout order)
    -> void
{
    helpThrowExceptionIfInitNotCompleted();
    mMemOrder = order;
}

auto MemoryOptions::helpWasInitCompleted() const -> bool
{
    const bool check1 = mComputeAllocator == Neon::Allocator::NULL_MEM;
    const bool check2 = mComputeType == Neon::DeviceType::NONE;
    const bool check3 = mIOType == Neon::DeviceType::NONE;
    const bool check4 = mIOAllocator == Neon::Allocator::NULL_MEM;

    if (check1 &&
        check2 &&
        check3 &&
        check4) {
        return false;
    }

    return true;
}

auto MemoryOptions::helpThrowExceptionIfInitNotCompleted() const
    -> void
{
    if (!helpWasInitCompleted()) {
        Neon::NeonException exception("MemoryOptions");
        exception << "A MemoryOptions object was used without initialization.";
        NEON_THROW(exception);
    }
}

}  // namespace Neon
