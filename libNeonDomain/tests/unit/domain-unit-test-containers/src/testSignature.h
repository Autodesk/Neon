
#pragma once
#include <functional>
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"

#include "Neon/domain/tools/TestData.h"


using namespace Neon::domain::tool::testing;

namespace device {
template <typename G, typename T, int C>
auto                 runDevice(TestData<G, T, C>& data) -> void;
extern template auto runDevice<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
}  // namespace device

namespace host {
template <typename G, typename T, int C>
auto                 runHost(TestData<G, T, C>& data) -> void;
extern template auto runHost<Neon::dGrid, int64_t, 0>(TestData<Neon::dGrid, int64_t, 0>&) -> void;
}  // namespace host
