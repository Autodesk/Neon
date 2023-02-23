#pragma once

#include "Neon/core/core.h"

namespace Neon::domain::internal::exp::dGrid {

class dGrid;
class dSpan;
template <typename T, int C>
class dPartition;

struct dIndex
{
    using OuterCell = dIndex;

    template <typename T, int C>
    friend class dPartition;
    friend dSpan;
    friend dGrid;

    template <typename T,
              int Cardinality>
    friend class dFieldDev;

    // dGrid specific types
    using Offset = int32_t;
    using Location = index_3d;
    using Count = int32_t;

    dIndex() = default;
    Location mLocation = 0;

   private:
    NEON_CUDA_HOST_DEVICE inline explicit dIndex(const Location::Integer& x,
                                                 const Location::Integer& y,
                                                 const Location::Integer& z);

    NEON_CUDA_HOST_DEVICE inline explicit dIndex(const Location& location);

    NEON_CUDA_HOST_DEVICE inline auto set() -> Location&;

    NEON_CUDA_HOST_DEVICE inline auto get() const -> const Location&;
};

// using dCell = dCell<void>;

}  // namespace Neon::domain::internal::exp::dGrid

#include "Neon/domain/internal/experimental/dGrid/dIndex_imp.h"
