#pragma once

#include "Neon/Neon.h"
#include "Neon/domain/Grids.h"
#include "Neon/skeleton/Skeleton.h"

template <typename Field>
auto computeGrad(const Field& gidx,
                 Field&       gradField,
                 double       h)
    -> Neon::set::Container;

extern template auto computeGrad<Neon::dGrid::Field<double, 0>>(const Neon::dGrid::Field<double, 0>& gidx,
                                                                Neon::dGrid::Field<double, 0>&       grad,
                                                                double                               h) -> Neon::set::Container;