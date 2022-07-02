#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/set/DevSet.h"
#include "Neon/skeleton/Skeleton.h"

namespace Test::containers {

template <int CardinalityTest, typename Field>
auto axpy(const Field&               A_g,
          const typename Field::Type alpha,
          Field&                     C_g) -> Neon::set::Container
{
    auto Kontainer = A_g.getGrid().getContainer(
        "axpy", [&A_g, &C_g, alpha ](Neon::set::Loader & L) -> auto{
            auto& A = L.load(A_g);
            auto& C = L.load(C_g);
            if constexpr (CardinalityTest == 1) {
                return [A, C, alpha] NEON_CUDA_HOST_DEVICE(const typename Field::Cell& cell) mutable {
                    C(cell, 0) += alpha * A(cell, 0);
                };
            }else {
                const int CardField = A_g.getCardinality();
                return [A, C, alpha, CardField] NEON_CUDA_HOST_DEVICE(const typename Field::Cell& cell) mutable {
                    for (int card = 0; card < CardField; card++) {
                        C(cell, card) += alpha * A(cell, card);
                    }
                };
            }
        });
    return Kontainer;
}

}  // namespace Test::containers
