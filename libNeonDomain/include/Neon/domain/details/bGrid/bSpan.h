#pragma once

#include "Neon/domain/details/bGrid/bIndex.h"

namespace Neon::domain::details::bGrid {

class bSpan
{
   public:
    bSpan() = default;
    virtual ~bSpan() = default;

    using Idx = bIndex;
    friend class bGrid;

    static constexpr int SpaceDim = 3;

    NEON_CUDA_HOST_DEVICE inline auto setAndValidate(Idx&          cell,
                                                     const size_t& x,
                                                     const size_t& y,
                                                     const size_t& z) const -> bool;

   private:
    NEON_CUDA_HOST_DEVICE inline auto setCell(Idx&                           cell,
                                              [[maybe_unused]] const size_t& x,
                                              [[maybe_unused]] const size_t& y,
                                              [[maybe_unused]] const size_t& z) const -> void;

    Idx::DataBlockCount mDataBlocCount;
    Idx::DataBlockCount mFirstDataBlockOffset;
    Neon::DataView      mDataView;
    uint32_t            mDataBlockSize;
    uint32_t*           mActiveMask;
};
}  // namespace Neon::domain::details::bGrid

#include "Neon/domain/details/bGrid/bSpan_imp.h"