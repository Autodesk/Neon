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
    auto c = A_g.getGrid().newContainer(
        "axpy", [&A_g, &C_g, alpha](Neon::set::Loader& L) -> auto {
            auto&     A = L.load(A_g);
            auto&     C = L.load(C_g);
            const int CardField = A_g.getCardinality();
            return [A, C, alpha, CardField] NEON_CUDA_HOST_DEVICE(const typename Field::Idx& cell) mutable {
                if constexpr (CardinalityTest == 1) {
                    C(cell, 0) += alpha * A(cell, 0);
                } else {
                    for (int card = 0; card < CardField; card++) {
                        C(cell, card) += alpha * A(cell, card);
                    }
                }
            };
        });
    return c;
}

}  // namespace Test::containers
