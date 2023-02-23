#pragma once

#include "Neon/set/DevSet.h"

namespace Neon {

template <typename T>
auto Backend::newDataSet()
    const -> Neon::set::DataSet<T>
{
    int  nDevs = getDeviceCount();
    auto result = Neon::set::DataSet<T>(nDevs);
    return result;
}

template <typename T>
auto Backend::newDataSet(T const& val)
    const -> Neon::set::DataSet<T>
{
    int  nDevs = getDeviceCount();
    auto result = Neon::set::DataSet<T>(nDevs, val);
    return result;
}

template <typename T, typename Lambda>
auto Backend::newDataSet(Lambda lambda)
    const -> Neon::set::DataSet<T>
{
    int  nDevs = getDeviceCount();
    auto result = Neon::set::DataSet<T>(nDevs);
    result.forEachSeq(lambda);
    return result;
}
}  // namespace Neon