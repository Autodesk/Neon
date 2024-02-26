
#pragma once
#include <functional>

#include "Neon/domain/Grids.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/domain/tools/TestData.h"

namespace space_filling_curves {
using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void;

extern template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;


}  // namespace globalIdx
