#include "Neon/Neon.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"
#include "expandSphere.h"

template <typename Field>
auto expandedLevelSet(Field& sdf,
                 double expantion)
    ->Neon::set::Container
{
    return sdf.getGrid().getContainer(
        "ExpandedLevelSet", [&, expantion](Neon::set::Loader& L) {
            auto& px = L.load(sdf);

            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename Field::Cell& cell) mutable {
                px(cell, 0) -= expantion;
            };
        });
}

template auto expandedLevelSet<Neon::domain::eGrid::Field<double, 0>>(Neon::domain::eGrid::Field<double, 0>& sdf, double expation) -> Neon::set::Container;