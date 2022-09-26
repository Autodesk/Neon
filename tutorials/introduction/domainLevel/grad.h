#pragma once

#include "Neon/Neon.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"

template <typename Field>
auto computeGrad(const Field& sdfField,
                 Field&       gradField,
                 double       h)
    -> Neon::set::Container;

extern template auto computeGrad<Neon::domain::eGrid::Field<double, 0>>(const Neon::domain::eGrid::Field<double, 0>& sdf, Neon::domain::eGrid::Field<double, 0>& grad, double h) -> Neon::set::Container;