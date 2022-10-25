#pragma once

#include "Neon/Neon.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"

template <typename Field>
auto expandedLevelSet(Field& sdf,
                      double expantion) -> Neon::set::Container;

extern template auto expandedLevelSet<Neon::domain::eGrid::Field<double, 0>>(Neon::domain::eGrid::Field<double, 0>& sdf, double expation) -> Neon::set::Container;