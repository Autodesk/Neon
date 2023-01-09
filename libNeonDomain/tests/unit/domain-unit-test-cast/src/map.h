
#pragma once
#include <functional>
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "cuda_fp16.h"

#include "Neon/domain/tools/TestData.h"


namespace map {
using namespace Neon::domain::tool::testing;

template <typename G, typename T, int C, typename ComputeType>
auto run(TestData<G, T, C>& data) -> void;

extern template auto run<Neon::domain::eGrid, int64_t, 0, double>(TestData<Neon::domain::eGrid, int64_t, 0>&) -> void;
extern template auto run<Neon::domain::dGrid, int64_t, 0, double>(TestData<Neon::domain::dGrid, int64_t, 0>&) -> void;

}  // namespace map
