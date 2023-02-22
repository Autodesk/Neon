#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::internal::exp::dGrid {

struct dVoxel
{
    using OuterCell = dVoxel;

    template <typename T,
              int Cardinality>
    friend struct dPartition;
    friend class dSpan;
    friend class dGrid;

    template <typename T,
              int Cardinality>
    friend class dFieldDev;

    // dGrid specific types
    using Offset = int32_t;
    using Location = index_3d;
    using Count = int32_t;

    dVoxel() = default;
    Location mLocation = 0;

   private:

    NEON_CUDA_HOST_DEVICE inline explicit dVoxel(const Location::Integer &x,
                                                const Location::Integer &y,
                                                const Location::Integer &z);

    NEON_CUDA_HOST_DEVICE inline explicit dVoxel(const Location& location);

    NEON_CUDA_HOST_DEVICE inline auto set() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto get() const -> const Location&;
};

// using dCell = dCell<void>;

}  // namespace Neon::domain::dense

#include "Neon/domain/internal/experimental/dGrid/dVoxel_imp.h"
