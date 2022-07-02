#include "Neon/domain/internal/bGrid/bField.h"
#include "Neon/domain/internal/bGrid/bPartition.h"
#include "Neon/domain/internal/bGrid/bPartitionIndexSpace.h"

#include "Neon/domain/patterns/ReduceKernels.cuh"

namespace Neon::domain::internal::bGrid {

template <typename T, int C>
auto bField<T, C>::dot(Neon::set::patterns::BlasSet<T>& blasSet,
                       const bField<T>&                 input,
                       Neon::set::MemDevSet<T>&         output,
                       const Neon::DataView&            dataView) -> void
{
    //TODO this only works for a  single GPU
    if (dataView != Neon::DataView::STANDARD) {
        NEON_DEV_UNDER_CONSTRUCTION("bField::dot");
    }

    Neon::domain::internal::dotCUB<T,
                                   Cell::sBlockSizeX,
                                   Cell::sBlockSizeY,
                                   Cell::sBlockSizeZ>(blasSet,
                                                      *mData->mGrid,
                                                      *this,
                                                      input,
                                                      output,
                                                      dataView);
}


template <typename T, int C>
auto bField<T, C>::norm2(Neon::set::patterns::BlasSet<T>& blasSet,
                         Neon::set::MemDevSet<T>&         output,
                         const Neon::DataView&            dataView) -> void
{
    //TODO this only works for a  single GPU
    if (dataView != Neon::DataView::STANDARD) {
        NEON_DEV_UNDER_CONSTRUCTION("bField::norm2");
    }

    Neon::domain::internal::norm2CUB<T,
                                     Cell::sBlockSizeX,
                                     Cell::sBlockSizeY,
                                     Cell::sBlockSizeZ>(blasSet,
                                                        *mData->mGrid,
                                                        *this,
                                                        output,
                                                        dataView);
}

template void bField<double, 0>::dot(Neon::set::patterns::BlasSet<double>&,
                                     const bField<double>&,
                                     Neon::set::MemDevSet<double>&,
                                     const Neon::DataView&);

template void bField<float, 0>::dot(Neon::set::patterns::BlasSet<float>&,
                                    const bField<float>&,
                                    Neon::set::MemDevSet<float>&,
                                    const Neon::DataView&);

template void bField<double, 0>::norm2(Neon::set::patterns::BlasSet<double>&,
                                       Neon::set::MemDevSet<double>&,
                                       const Neon::DataView&);

template void bField<float, 0>::norm2(Neon::set::patterns::BlasSet<float>&,
                                      Neon::set::MemDevSet<float>&,
                                      const Neon::DataView&);


}  // namespace Neon::domain::internal::bGrid