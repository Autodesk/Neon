#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/set/DevSet.h"
#include "Neon/skeleton/Skeleton.h"

namespace Test {
namespace containers {

template <typename Field>
auto sum(const Field& A_g,
         const Field& B_g,
         Field&       C_g) -> Neon::set::Container
{
    auto container = A_g.getGrid().newContainer(
        "sum", [&A_g, &C_g, &B_g](Neon::set::Loader& L) -> auto {
            auto& A = L.load(A_g);
            auto& B = L.load(B_g);
            auto& C = L.load(C_g);

            return [A, C] NEON_CUDA_HOST_DEVICE(const typename Field::Cell& cell) mutable {
                C(cell, 0) += alpha * A(cell, 0);
            };
        });
    return container;
}

}  // namespace containers
}  // namespace Test
