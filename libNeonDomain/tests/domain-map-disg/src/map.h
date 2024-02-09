
#pragma once
#include <functional>

#include "Neon/domain/Grids.h"
#include "Neon/domain/details/dGridSoA/dGridSoA.h"
#include "Neon/domain/tools/TestData.h"
#include "Neon/domain/details/dGridDisg/dGrid.h"


namespace map {
using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void;

extern template auto run<Neon::bGridDisg, int64_t, 0>(TestData<Neon::bGridDisg, int64_t, 0>&) -> void;
extern template auto run<Neon::bGridMask, int64_t, 0>(TestData<Neon::bGridMask, int64_t, 0>&) -> void;


extern template auto run<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
namespace dataView {

template <typename G, typename T, int C>
auto run(TestData<G, T, C>& data) -> void;

extern template auto run<Neon::bGridDisg, int64_t, 0>(TestData<Neon::bGridDisg, int64_t, 0>&) -> void;
extern template auto run<Neon::bGridMask, int64_t, 0>(TestData<Neon::bGridMask, int64_t, 0>&) -> void;
extern template auto run<Neon::bGrid, int64_t, 0>(TestData<Neon::bGrid, int64_t, 0>&) -> void;
extern template auto run<Neon::dGrid , int64_t, 0>(TestData<Neon::dGrid , int64_t, 0>&) -> void;

}  // namespace dataView

}  // namespace map
