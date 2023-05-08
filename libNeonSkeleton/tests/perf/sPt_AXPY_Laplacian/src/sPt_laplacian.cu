#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/set/DevSet.h"
#include "Neon/skeleton/Skeleton.h"
#include "sPt_laplacian.h"

namespace sk {
template <typename Field>
auto laplacianFilter(const Field& A_g,
                     Field&       C_g) -> Neon::set::Container
{
    auto container = A_g.getGrid().newContainer(
        "laplacianFilter", [&A_g, &C_g](Neon::set::Loader& L) -> auto {
            auto& A = L.load(A_g, Neon::Compute::STENCIL);
            auto& C = L.load(C_g);

            return [A, C] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& cell) mutable {
                typename Field::Type sum = 0;
                for (int nIdx = 0; nIdx < 6; nIdx++) {
                    const auto nInfo = A.getNghData(cell, nIdx, 0, typename Field::Type(0));
                    const auto nVal = nInfo.getData();
                    sum += nVal;
                }
                C(cell, 0) = -sum + 6 * A(cell, 0);
            };
        },
        Neon::Execution::device);
    return container;
}

template <typename Field>
auto axpy(const Field&               A_g,
          const typename Field::Type alpha,
          Field&                     C_g) -> Neon::set::Container
{
    auto container = A_g.getGrid().newContainer(
        "axpy", [&A_g, &C_g, alpha](Neon::set::Loader& L) -> auto {
            auto& A = L.load(A_g);
            auto& C = L.load(C_g);

            return [A, C, alpha] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& cell) mutable {
                C(cell, 0) += alpha * A(cell, 0);
            };
        },
        Neon::Execution::device);
    return container;
}

template auto laplacianFilter<Neon::domain::details::eGrid::eField<int64_t>>(const Neon::domain::details::eGrid::eField<int64_t>& A_g,
                                                                             Neon::domain::details::eGrid::eField<int64_t>&       C_g) -> Neon::set::Container;

template auto laplacianFilter<Neon::domain::details::eGrid::eField<double>>(const Neon::domain::details::eGrid::eField<double>& A_g,
                                                                            Neon::domain::details::eGrid::eField<double>&       C_g) -> Neon::set::Container;

template auto axpy<Neon::domain::details::eGrid::eField<int64_t>>(const Neon::domain::details::eGrid::eField<int64_t>& A_g,
                                                                  int64_t                                              alpha,
                                                                  Neon::domain::details::eGrid::eField<int64_t>&       C_g) -> Neon::set::Container;

template auto axpy<Neon::domain::details::eGrid::eField<double>>(const Neon::domain::details::eGrid::eField<double>& A_g,
                                                                 double                                              alpha,
                                                                 Neon::domain::details::eGrid::eField<double>&       C_g) -> Neon::set::Container;
}  // namespace sk
