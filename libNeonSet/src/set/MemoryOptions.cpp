#include "Neon/set/MemoryOptions.h"

namespace Neon {

MemoryOptions::MemoryOptions()
{
}

MemoryOptions::MemoryOptions(Neon::DeviceType   ioType,
                             Neon::DeviceType   deviceType,
                             Neon::MemoryLayout order)
{
    mDeviceType = deviceType;
    mDeviceAllocator = Neon::AllocatorUtils::getDefault(deviceType);

    mHostType = ioType;
    mHostAllocator = Neon::AllocatorUtils::getDefault(ioType);

    mMemOrder = order;
}

MemoryOptions::MemoryOptions(Neon::DeviceType   hostType,
                             Neon::Allocator    hostAllocator,
                             Neon::DeviceType   deviceType,
                             Neon::Allocator    deviceAllocators,
                             Neon::MemoryLayout order)
{
    mHostType = hostType;
    mHostAllocator = hostAllocator;

    mDeviceType = deviceType;
    mDeviceAllocator = deviceAllocators;

    mMemOrder = order;
}

MemoryOptions::MemoryOptions(Neon::DeviceType   ioType,
                             Neon::Allocator    ioAllocator,
                             Neon::DeviceType   deviceType,
                             Neon::Allocator    computeAllocators[Neon::DeviceTypeUtil::nConfig],
                             Neon::MemoryLayout order)
{
    mHostType = ioType;
    mHostAllocator = ioAllocator;

    mDeviceType = deviceType;
    mDeviceAllocator = computeAllocators[Neon::DeviceTypeUtil::toInt(deviceType)];

    mMemOrder = order;
}

auto MemoryOptions::getDeviceType() const
    -> Neon::DeviceType
{
    helpThrowExceptionIfInitNotCompleted();
    return mDeviceType;
}

auto MemoryOptions::getHostType() const
    -> Neon::DeviceType
{
    helpThrowExceptionIfInitNotCompleted();
    return mHostType;
}

auto MemoryOptions::getDeviceAllocator(Neon::DataUse dataUse) const
    -> Neon::Allocator
{
    helpThrowExceptionIfInitNotCompleted();

    if (dataUse == Neon::DataUse::HOST) {
        return Neon::Allocator::NULL_MEM;
    }
    if (getDeviceType() == Neon::DeviceType::CPU) {
        return Neon::Allocator::NULL_MEM;
    }
    return mDeviceAllocator;
}

auto MemoryOptions::getIOAllocator(Neon::DataUse dataUse) const
    -> Neon::Allocator
{
    helpThrowExceptionIfInitNotCompleted();

    if (dataUse == Neon::DataUse::DEVICE) {
        return Neon::Allocator::NULL_MEM;
    }
    return mHostAllocator;
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
    const bool check1 = mDeviceAllocator == Neon::Allocator::NULL_MEM;
    const bool check2 = mDeviceType == Neon::DeviceType::NONE;
    const bool check3 = mHostType == Neon::DeviceType::NONE;
    const bool check4 = mHostAllocator == Neon::Allocator::NULL_MEM;

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

MemoryOptions::MemoryOptions(Neon::MemoryLayout order)
{
    mMemOrder = order;
}

}  // namespace Neon
