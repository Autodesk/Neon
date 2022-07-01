#pragma once

namespace Neon::domain::internal::aGrid {

inline aPartitionIndexSpace::aPartitionIndexSpace()
    : mNElements(0),
      mDataView(DataView::STANDARD)
{
}


inline aPartitionIndexSpace::aPartitionIndexSpace(int            nElements,
                                                  Neon::DataView dataView)
    : mNElements(nElements),
      mDataView(dataView)
{
    // For now we support only standard view
    if (dataView != Neon::DataView::STANDARD) {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }
}


NEON_CUDA_HOST_DEVICE inline auto
aPartitionIndexSpace::numElements() const -> int
{
    switch (mDataView) {
        case Neon::DataView::STANDARD: {
            return mNElements;
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


NEON_CUDA_HOST_DEVICE inline auto
aPartitionIndexSpace::setAndValidate(Cell&                          cell,
                                     const size_t&                  x,
                                     [[maybe_unused]] const size_t& y,
                                     [[maybe_unused]] const size_t& z)
    const
    -> bool
{
    cell.set() = Cell::Location(x);

#if defined(NEON_PLACE_CUDA_DEVICE)
    const bool isValid = (cell.get() < this->numElements()) ? true : false;
    return isValid;
#else
    const bool isValid = true;
    return isValid;
#endif
}

}  // namespace Neon::domain::internal::aGrid