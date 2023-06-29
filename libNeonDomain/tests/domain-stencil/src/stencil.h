
#pragma once
#include <functional>
#include "Neon/domain/Grids.h"

#include "Neon/domain/tools/TestData.h"



namespace map {
using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto runNoTemplate(TestData<G, T, C>& data) -> void;

template <typename G, typename T, int C>
auto runTemplate(TestData<G, T, C>& data) -> void;


extern template auto runNoTemplate<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
extern template auto runNoTemplate<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;
extern template auto runNoTemplate<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
extern template auto runNoTemplate<Neon::dGridSoA , int64_t, 0>(TestData<Neon::dGridSoA, int64_t, 0>&) -> void;

extern template auto runTemplate<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
extern template auto runTemplate<Neon::eGrid, int64_t, 0>(TestData<Neon::eGrid, int64_t, 0>&) -> void;
extern template auto runTemplate<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
extern template auto runTemplate<Neon::dGridSoA , int64_t, 0>(TestData<Neon::dGridSoA, int64_t, 0>&) -> void;

}  // namespace map
