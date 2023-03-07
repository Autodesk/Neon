#pragma once
#include "eSpan.h"

namespace Neon::domain::details::eGrid {
NEON_CUDA_HOST_DEVICE
auto eSpan::nElements() const -> int64_t
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
eSpan::helpApplyDataViewShift(Idx& Idx)
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
            Idx.set() += m_bdrOff[ComDirectionUtils::toInt(ComDirection::DW)];
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
auto eSpan::setAndValidate(Idx&          Idx,
                           const uint32_t& x)
    const
    -> bool
{
    Idx.set() = Idx::InternalIdx(x);


    bool isValid = false;

    if (Idx.get() < this->nElements()) {
        isValid = true;
    }

    helpApplyDataViewShift(Idx);
    return isValid;
}


NEON_CUDA_HOST_DEVICE
auto eSpan::helpGetBoundaryOffset() -> Idx::Offset*
{
    return m_bdrOff;
}


NEON_CUDA_HOST_DEVICE
auto eSpan::helpGetGhostOffset() -> Idx::Offset*
{
    return m_ghostOff;
}

NEON_CUDA_HOST_DEVICE
auto eSpan::helpGetDataView() -> Neon::DataView&
{
    return m_dataView;
}
}  // namespace Neon::domain::details::eGrid