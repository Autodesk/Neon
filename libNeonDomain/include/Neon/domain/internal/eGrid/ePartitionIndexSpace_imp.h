#pragma once
#include "Neon/core/core.h"

namespace Neon::domain::internal::eGrid {


NEON_CUDA_HOST_DEVICE
auto ePartitionIndexSpace::nElements() const -> int64_t
{
    switch (m_dataView) {
        case Neon::DataView::STANDARD: {
            return m_ghostOff[ComDirection_e::COM_DW];
        }
        case Neon::DataView::INTERNAL: {
            return m_bdrOff[ComDirection_e::COM_DW];
        }
        case Neon::DataView::BOUNDARY: {
            return m_ghostOff[ComDirection_e::COM_DW] - m_bdrOff[ComDirection_e::COM_DW];
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
ePartitionIndexSpace::hApplyDataViewShift(Cell& cell)
    const
    -> void
{
    switch (m_dataView) {
        case Neon::DataView::STANDARD:
        case Neon::DataView::INTERNAL: {
            // Nothing to do
            return;
        }
        case Neon::DataView::BOUNDARY: {
            cell.set() += m_bdrOff[ComDirection_e::COM_DW];
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
auto ePartitionIndexSpace::setAndValidate(Cell&                          cell,
                                          const size_t&                  x,
                                          [[maybe_unused]] const size_t& y,
                                          [[maybe_unused]] const size_t& z)
    const
    -> bool
{
    cell.set() = Cell::Location(x);


    bool isValid = false;

    if (cell.get() < this->nElements()) {
        isValid = true;
    }

    hApplyDataViewShift(cell);
    return isValid;


}


NEON_CUDA_HOST_DEVICE
auto ePartitionIndexSpace::hGetBoundaryOffset() -> Cell::Offset*
{
    return m_bdrOff;
}


NEON_CUDA_HOST_DEVICE
auto ePartitionIndexSpace::hgetGhostOffset() -> Cell::Offset*
{
    return m_ghostOff;
}


NEON_CUDA_HOST_DEVICE
auto ePartitionIndexSpace::hGetDataView() -> Neon::DataView&
{
    return m_dataView;
}

}  // namespace Neon::domain::internal::eGrid