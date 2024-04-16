
#pragma once
#include <functional>

#include "Neon/domain/Grids.h"
#include "Neon/domain/tools/TestData.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/domain/details/dGridDisg/dGrid.h"


namespace globalIdx {
using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void;

template <typename G, typename T, int C>
auto runTemplate(TestData<G, T, C>& data) -> void;

extern template auto run<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
extern template auto run<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;
extern template auto run<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
extern template auto run<Neon::dGridSoA , int64_t, 0>(TestData<Neon::dGridSoA, int64_t, 0>&) -> void;
extern template auto run<Neon::domain::details::disaggregated::dGrid::dGrid , int64_t, 0>(TestData<Neon::domain::details::disaggregated::dGrid::dGrid, int64_t, 0>&) -> void;
extern template auto run<Neon::bGridMgpu , int64_t, 0>(TestData<Neon::bGridMgpu, int64_t, 0>&) -> void;

extern template auto runTemplate<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
extern template auto runTemplate<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;
extern template auto runTemplate<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
extern template auto runTemplate<Neon::dGridSoA , int64_t, 0>(TestData<Neon::dGridSoA, int64_t, 0>&) -> void;
extern template auto runTemplate<Neon::domain::details::disaggregated::dGrid::dGrid , int64_t, 0>(TestData<Neon::domain::details::disaggregated::dGrid::dGrid, int64_t, 0>&) -> void;
extern template auto runTemplate<Neon::bGridMgpu , int64_t, 0>(TestData<Neon::bGridMgpu, int64_t, 0>&) -> void;

}  // namespace map
