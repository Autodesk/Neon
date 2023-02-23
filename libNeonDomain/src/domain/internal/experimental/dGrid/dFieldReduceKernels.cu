#include "Neon/domain/internal/experimental/dGrid/dGrid.h"
#include "Neon/domain/patterns/ReduceKernels.cuh"

namespace Neon::domain::internal::exp::dGrid {
#if 0
template <typename T, int C>
auto dFieldDev<T, C>::dotCUB(
    Neon::set::patterns::BlasSet<T>& blasSet,
    const dFieldDev<T>&              input,
    Neon::set::MemDevSet<T>&         output,
    const Neon::DataView&            dataView) -> void
{
    Neon::domain::internal::dotCUB<T, 256, 1, 1>(blasSet,
                                                 grid(),
                                                 *this,
                                                 input,
                                                 output,
                                                 dataView);
}


template <typename T, int C>
auto dFieldDev<T, C>::norm2CUB(
    Neon::set::patterns::BlasSet<T>& blasSet,
    Neon::set::MemDevSet<T>&         output,
    const Neon::DataView&            dataView) -> void
{
    Neon::domain::internal::norm2CUB<T, 256, 1, 1>(blasSet,
                                                   grid(),
                                                   *this,
                                                   output,
                                                   dataView);
}

template void dFieldDev<double, 0>::dotCUB(Neon::set::patterns::BlasSet<double>&,
                                           const dFieldDev<double>&,
                                           Neon::set::MemDevSet<double>&,
                                           const Neon::DataView&);

template void dFieldDev<float, 0>::dotCUB(Neon::set::patterns::BlasSet<float>&,
                                          const dFieldDev<float>&,
                                          Neon::set::MemDevSet<float>&,
                                          const Neon::DataView&);

template void dFieldDev<int64_t, 0>::dotCUB(Neon::set::patterns::BlasSet<int64_t>&,
                                            const dFieldDev<int64_t>&,
                                            Neon::set::MemDevSet<int64_t>&,
                                            const Neon::DataView&);

template void dFieldDev<double, 0>::norm2CUB(Neon::set::patterns::BlasSet<double>&,
                                             Neon::set::MemDevSet<double>&,
                                             const Neon::DataView&);

template void dFieldDev<float, 0>::norm2CUB(Neon::set::patterns::BlasSet<float>&,
                                            Neon::set::MemDevSet<float>&,
                                            const Neon::DataView&);

template void dFieldDev<int64_t, 0>::norm2CUB(Neon::set::patterns::BlasSet<int64_t>&,
                                            Neon::set::MemDevSet<int64_t>&,
                                            const Neon::DataView&);
#endif
}  // namespace Neon::domain::internal::dGrid