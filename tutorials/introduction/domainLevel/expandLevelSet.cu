#include "Neon/Neon.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"
#include "expandLevelSet.h"

template <typename Field>
auto expandLevelSet(Field& sdf,
                    double expansion)
    -> Neon::set::Container
{
    return sdf.getGrid().getContainer(
        "ExpandLevelSet",
        // Neon Loading Lambda
        [&, expansion](Neon::set::Loader& L) {
            auto& px = L.load(sdf);

            // Neon Compute Lambda
            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename Field::Cell& cell) mutable {
                px(cell, 0) -= expansion;
            };
        });
}

template auto expandLevelSet<Neon::domain::eGrid::Field<double, 0>>(Neon::domain::eGrid::Field<double, 0>& sdf, double expation) -> Neon::set::Container;