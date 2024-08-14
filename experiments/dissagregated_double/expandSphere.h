#pragma once

#include "Neon/Neon.h"
#include "Neon/domain/Grids.h"

template <typename Field>
auto expandedLevelSet(Field& sdf,
                      double expantion) -> Neon::set::Container;

extern template auto expandedLevelSet<Neon::dGrid::Field<double, 0>>(Neon::dGrid::Field<double, 0>& sdf, double expation) -> Neon::set::Container;