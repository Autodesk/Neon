#pragma once
#include "Neon/domain/patterns/PatternScalar.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/patterns/BlasSet.h"

namespace Neon {

template <typename T>
PatternScalar<T>::PatternScalar(Neon::Backend               backend,
                                Neon::sys::patterns::Engine engine)
{
    mData = std::make_shared<Data>();
    mData->backend = backend;
    mData->blasSetBoundary = Neon::set::patterns::BlasSet<T>(mData->backend.devSet(), engine);
    mData->blasSetInternal = Neon::set::patterns::BlasSet<T>(mData->backend.devSet(), engine);
    mData->blasSetStandard = Neon::set::patterns::BlasSet<T>(mData->backend.devSet(), engine);
    mData->devType = backend.devType();

    mData->hostTempBoundary = backend.devSet().template newMemDevSet<T>(Neon::DeviceType::CPU,
                                                                        Neon::Allocator::CUDA_MEM_HOST, 1);
    mData->hostTempInternal = backend.devSet().template newMemDevSet<T>(Neon::DeviceType::CPU,
                                                                        Neon::Allocator::CUDA_MEM_HOST, 1);
    mData->hostTempStandard = backend.devSet().template newMemDevSet<T>(Neon::DeviceType::CPU,
                                                                        Neon::Allocator::CUDA_MEM_HOST, 1);
    if (engine == Neon::sys::patterns::Engine::CUB) {
        mData->deviceTempBoundary = backend.devSet().template newMemDevSet<T>(Neon::DeviceType::CUDA,
                                                                              Neon::Allocator::CUDA_MEM_DEVICE, 1);
        mData->deviceTempInternal = backend.devSet().template newMemDevSet<T>(Neon::DeviceType::CUDA,
                                                                              Neon::Allocator::CUDA_MEM_DEVICE, 1);
        mData->deviceTempStandard = backend.devSet().template newMemDevSet<T>(Neon::DeviceType::CUDA,
                                                                              Neon::Allocator::CUDA_MEM_DEVICE, 1);
    }
}

template <typename T>
auto PatternScalar<T>::operator()() -> T&
{
    return standardResult;
}

template <typename T>
auto PatternScalar<T>::operator()() const -> const T&
{
    return standardResult;
}

template <typename T>
auto PatternScalar<T>::uid() const -> Neon::set::MultiDeviceObjectUid
{
    void*                           addr = static_cast<void*>(mData.get());
    Neon::set::MultiDeviceObjectUid uidRes = (size_t)addr;
    return uidRes;
}

template <typename T>
auto PatternScalar<T>::getPartition(
    [[maybe_unused]] const Neon::DeviceType& devType,
    [[maybe_unused]] const Neon::SetIdx&     idx,
    [[maybe_unused]] const Neon::DataView&   dataView) const -> const Partition&
{
    return *this;
}

template <typename T>
auto PatternScalar<T>::getPartition([[maybe_unused]] const DeviceType& devType,
                                    [[maybe_unused]] const SetIdx&     idx,
                                    [[maybe_unused]] const DataView&   dataView) -> PatternScalar::Partition&
{
    return *this;
}

template <typename T>
auto PatternScalar<T>::getPartition([[maybe_unused]] Neon::Execution execution,
                                    [[maybe_unused]] Neon::SetIdx    setIdx,
                                    [[maybe_unused]] const DataView& dataView) const -> const PatternScalar::Partition&
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T>
auto PatternScalar<T>::updateIO([[maybe_unused]] int streamId) -> void
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}
template <typename T>
auto PatternScalar<T>::updateCompute([[maybe_unused]] int streamId) -> void
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T>
auto PatternScalar<T>::getPartition([[maybe_unused]] Neon::Execution execution,
                                    [[maybe_unused]] Neon::SetIdx    setIdx,
                                    [[maybe_unused]] const DataView& dataView)
    -> PatternScalar::Partition&
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T>
auto PatternScalar<T>::operator()(const Neon::DataView& dataView) -> T&
{
    if (dataView == Neon::DataView::STANDARD) {
        return standardResult;

    } else if (dataView == Neon::DataView::INTERNAL) {
        return internalResult;

    } else if (dataView == Neon::DataView::BOUNDARY) {
        return boundaryResult;
    } else {
        NeonException exc("PatternScalar::PatternScalar");
        exc << "Unsupported dataView " << Neon::DataViewUtil::toString(dataView);
        NEON_THROW(exc);
    }
}

template <typename T>
auto PatternScalar<T>::setStream(int                   streamIdx,
                                 const Neon::DataView& dataView) const -> void
{
    auto streams = mData->backend.streamSet(streamIdx);

    if (mData->devType == Neon::DeviceType::CUDA) {
        if (dataView == Neon::DataView::STANDARD) {
            mData->blasSetStandard.setStream(streams);

        } else if (dataView == Neon::DataView::INTERNAL) {
            mData->blasSetInternal.setStream(streams);

        } else if (dataView == Neon::DataView::BOUNDARY) {
            mData->blasSetBoundary.setStream(streams);

        } else {
            NeonException exc("PatternScalar::PatternScalar");
            exc << "Unsupported dataView " << Neon::DataViewUtil::toString(dataView);
            NEON_THROW(exc);
        }
    }
}

template <typename T>
auto PatternScalar<T>::getTempMemory(const Neon::DataView& dataView,
                                     Neon::DeviceType      devType) -> Neon::set::MemDevSet<T>&
{
    if (dataView == Neon::DataView::STANDARD) {
        if (devType == Neon::DeviceType::CPU) {
            return mData->hostTempStandard;
        } else {
            return mData->deviceTempStandard;
        }
    } else if (dataView == Neon::DataView::INTERNAL) {
        if (devType == Neon::DeviceType::CPU) {
            return mData->hostTempInternal;
        } else {
            return mData->deviceTempInternal;
        }

    } else if (dataView == Neon::DataView::BOUNDARY) {
        if (devType == Neon::DeviceType::CPU) {
            return mData->hostTempBoundary;
        } else {
            return mData->deviceTempBoundary;
        }

    } else {
        NeonException exc("PatternScalar::tempMemory");
        exc << "Unsupported dataView " << Neon::DataViewUtil::toString(dataView);
        NEON_THROW(exc);
    }
}

template <typename T>
auto PatternScalar<T>::getBlasSet(const Neon::DataView& dataView) -> Neon::set::patterns::BlasSet<T>&
{
    if (dataView == Neon::DataView::STANDARD) {
        return mData->blasSetStandard;

    } else if (dataView == Neon::DataView::INTERNAL) {
        return mData->blasSetInternal;

    } else if (dataView == Neon::DataView::BOUNDARY) {
        return mData->blasSetBoundary;

    } else {
        NeonException exc("PatternScalar::tempMemory");
        exc << "Unsupported dataView " << Neon::DataViewUtil::toString(dataView);
        NEON_THROW(exc);
    }
}


extern template class PatternScalar<float>;
extern template class PatternScalar<double>;

}  // namespace Neon