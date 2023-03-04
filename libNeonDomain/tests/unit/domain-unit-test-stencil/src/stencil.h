
#pragma once
#include <functional>
#include "Neon/domain/details/dGrid/dGrid.h"
#include "Neon/domain/dGrid.h"

#include "Neon/domain/tools/TestData.h"



namespace map {
using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void;

extern template auto run<Neon::domain::details::dGrid::dGrid, int64_t, 0>(TestData<Neon::domain::details::dGrid::dGrid, int64_t, 0>&) -> void;

}  // namespace map
