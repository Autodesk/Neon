//#ifdef NEON_COMPILER_CUDA
//#include <cub/cub.cuh>
//#endif
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/set/DevSet.h"

namespace sk {
template <typename Field>
auto laplacianFilter(const Field& A_g,
                     Field&       C_g) -> Neon::set::Container;

template <typename Field>
auto axpy(const Field&               A_g,
          const typename Field::Type alpha,
          Field&                     C_g) -> Neon::set::Container;

extern template auto laplacianFilter<Neon::domain::internal::eGrid::eField<uint64_t>>(const Neon::domain::internal::eGrid::eField<uint64_t>& A_g,
                                                                                   Neon::domain::internal::eGrid::eField<uint64_t>&       C_g) -> Neon::set::Container;

extern template auto laplacianFilter<Neon::domain::internal::eGrid::eField<double>>(const Neon::domain::internal::eGrid::eField<double>& A_g,
                                                                                 Neon::domain::internal::eGrid::eField<double>&       C_g) -> Neon::set::Container;

extern template auto axpy<Neon::domain::internal::eGrid::eField<uint64_t>>(const Neon::domain::internal::eGrid::eField<uint64_t>& A_g,
                                                                        uint64_t                                            alpha,
                                                                        Neon::domain::internal::eGrid::eField<uint64_t>&       C_g) -> Neon::set::Container;

extern template auto axpy<Neon::domain::internal::eGrid::eField<double>>(const Neon::domain::internal::eGrid::eField<double>& A_g,
                                                                      double                                            alpha,
                                                                      Neon::domain::internal::eGrid::eField<double>&       C_g) -> Neon::set::Container;
}  // namespace sk
