#pragma once

#include "Neon/domain/internal/sGrid/sGrid.h"
#include "sPartitionIndexSpace.h"


namespace Neon::domain::internal::sGrid {


NEON_CUDA_HOST_DEVICE
auto sPartitionIndexSpace::nElements() const
    -> int64_t
{
    switch (mDataView) {
        case Neon::DataView::STANDARD: {
            return mGhostOff;
        }
        case Neon::DataView::INTERNAL: {
            return mBdrOff;
        }
        case Neon::DataView::BOUNDARY: {
            return mGhostOff - mBdrOff;
        }
        default: {
#if defined(NEON_PLACE_CUDA_DEVICE)
            return -1;
#else
            NEON_THROW_UNSUPPORTED_OPTION("");
#endif
        }
    }
}


NEON_CUDA_HOST_DEVICE auto
sPartitionIndexSpace::helpApplyDataViewShift(Cell& cell) const
    -> void
{
    switch (mDataView) {
        case Neon::DataView::STANDARD:
        case Neon::DataView::INTERNAL: {
            // Nothing to do
            return;
        }
        case Neon::DataView::BOUNDARY: {
            cell.get() += mBdrOff;
            return;
        }
        default: {
        }
    }
#if defined(NEON_PLACE_CUDA_HOST)
    NEON_THROW_UNSUPPORTED_OPTION();
#else
    printf("Error!!!!\n");
    // Just a way to force a segmentation fault when running on CUDA
    int* error = nullptr;
    error[0] = 0xBAD;
#endif
}


NEON_CUDA_HOST_DEVICE
auto sPartitionIndexSpace::setAndValidate(Cell&                          cell,
                                          const size_t&                  x,
                                          [[maybe_unused]] const size_t& y,
                                          [[maybe_unused]] const size_t& z) const
    -> bool
{
    cell.get() = Cell::Location(x);

#if defined(NEON_PLACE_CUDA_DEVICE)
    bool isValid = false;

    if (cell.get() < this->nElements()) {
        isValid = true;
    }

    helpApplyDataViewShift(cell);
    return isValid;

#else
    helpApplyDataViewShift(cell);
    const bool isValid = true;
    return isValid;
#endif
}


NEON_CUDA_HOST_DEVICE
auto sPartitionIndexSpace::helpGetBoundaryOffset() const
    -> Cell::Offset
{
    return mBdrOff;
}


NEON_CUDA_HOST_DEVICE
auto sPartitionIndexSpace::helpGetGhostOffset() const
    -> Cell::Offset
{
    return mGhostOff;
}


NEON_CUDA_HOST_DEVICE
auto sPartitionIndexSpace::helpGetDataView() const
    -> Neon::DataView
{
    return mDataView;
}

auto sPartitionIndexSpace::helpGetBoundaryOffset()
    -> sCell::Offset&
{
    return mBdrOff;
}

auto sPartitionIndexSpace::helpGetGhostOffset() -> sCell::Offset&
{
    return mGhostOff;
}

auto sPartitionIndexSpace::helpGetDataView() -> Neon::DataView&
{
    return mDataView;
}

}  // namespace Neon::domain::internal::sGrid