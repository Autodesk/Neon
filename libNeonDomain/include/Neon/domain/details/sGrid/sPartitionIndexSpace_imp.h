#pragma once

#include "Neon/domain/details/sGrid/sGrid.h"
#include "sSpan.h"


namespace Neon::domain::details::sGrid {


NEON_CUDA_HOST_DEVICE
auto sSpan::nElements() const
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
sSpan::helpApplyDataViewShift(Cell& cell) const
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
auto sSpan::setAndValidate(Cell&         cell,
                           const size_t& x) const
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
auto sSpan::helpGetBoundaryOffset() const
    -> Cell::Offset
{
    return mBdrOff;
}


NEON_CUDA_HOST_DEVICE
auto sSpan::helpGetGhostOffset() const
    -> Cell::Offset
{
    return mGhostOff;
}


NEON_CUDA_HOST_DEVICE
auto sSpan::helpGetDataView() const
    -> Neon::DataView
{
    return mDataView;
}

auto sSpan::helpGetBoundaryOffset()
    -> sIndex::Offset&
{
    return mBdrOff;
}

auto sSpan::helpGetGhostOffset() -> sIndex::Offset&
{
    return mGhostOff;
}

auto sSpan::helpGetDataView() -> Neon::DataView&
{
    return mDataView;
}

}  // namespace Neon::domain::details::sGrid