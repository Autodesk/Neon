#pragma once

#include "Neon/Neon.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"

template <typename Field>
auto expandLevelSet(Field& sdf,
                    double expantion) -> Neon::set::Container;

extern template auto expandLevelSet<Neon::domain::eGrid::Field<double, 0>>(Neon::domain::eGrid::Field<double, 0>& sdf, double expation) -> Neon::set::Container;