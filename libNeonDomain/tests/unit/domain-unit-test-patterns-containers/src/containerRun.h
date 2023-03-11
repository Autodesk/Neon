
#pragma once
#include <functional>
#include "Neon/domain/details/dGrid/dGrid.h"
#include "Neon/domain/tools/TestData.h"


using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto runContainer(TestData<G, T, C>&                data,
                  const Neon::sys::patterns::Engine eng) -> void;

extern template auto runContainer<Neon::domain::details::dGrid::eGrid, int64_t, 0>(TestData<Neon::domain::details::dGrid::eGrid, int64_t, 0>&,
                                                                                         const Neon::sys::patterns::Engine eng) -> void;
