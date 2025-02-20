#include "Neon/Neon.h"
#include "Neon/domain/Grids.h"
#include "Neon/skeleton/Skeleton.h"
#include "expandSphere.h"

template <typename Field>
auto expandedLevelSet(Field& sdf,
                 double expantion)
    ->Neon::set::Container
{
    return sdf.getGrid().newContainer(
        "ExpandedLevelSet", [&, expantion](Neon::set::Loader& L) {
            auto& px = L.load(sdf);

            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename Field::Idx& gidx) mutable {
                px(gidx, 0) -= expantion;
            };
        });
}

template auto expandedLevelSet<Neon::dGrid::Field<double, 0>>(Neon::dGrid::Field<double, 0>& sdf, double expation) -> Neon::set::Container;