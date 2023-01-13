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

    const int blockSize = mData->grid->getBlockSize();

    if (blockSize == 2) {
        Neon::domain::internal::dotCUB<T, 2, 2, 2>(blasSet,
                                                   *mData->grid,
                                                   *this,
                                                   input,
                                                   output,
                                                   dataView);
    } else if (blockSize == 4) {
        Neon::domain::internal::dotCUB<T, 4, 4, 4>(blasSet,
                                                   *mData->grid,
                                                   *this,
                                                   input,
                                                   output,
                                                   dataView);
    } else if (blockSize == 8) {
        Neon::domain::internal::dotCUB<T, 8, 8, 8>(blasSet,
                                                   *mData->grid,
                                                   *this,
                                                   input,
                                                   output,
                                                   dataView);
    } else {
        NeonException exc("bField::dot");
        exc << "block size (" << blockSize << ")is too big.";
        NEON_THROW(exc);
    }
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

    const int blockSize = mData->grid->getBlockSize();

    if (blockSize == 2) {
        Neon::domain::internal::norm2CUB<T, 2, 2, 2>(blasSet,
                                                     *mData->grid,
                                                     *this,
                                                     output,
                                                     dataView);
    } else if (blockSize == 4) {
        Neon::domain::internal::norm2CUB<T, 4, 4, 4>(blasSet,
                                                     *mData->grid,
                                                     *this,
                                                     output,
                                                     dataView);
    } else if (blockSize == 8) {
        Neon::domain::internal::norm2CUB<T, 8, 8, 8>(blasSet,
                                                     *mData->grid,
                                                     *this,
                                                     output,
                                                     dataView);
    } else {
        NeonException exc("bField::norm2");
        exc << "block size (" << blockSize << ")is too big.";
        NEON_THROW(exc);
    }
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