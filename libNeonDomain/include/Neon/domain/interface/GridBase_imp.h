#pragma once
#include "Neon/set/DevSet.h"

namespace Neon::domain::interface {

template <typename T>
auto GridBase::newDataSet() const -> const Neon::set::DataSet<T>
{
    auto dataSet = getBackend().devSet().template newDataSet<T>();
    return dataSet;
}

}  // namespace Neon::domain::interface
