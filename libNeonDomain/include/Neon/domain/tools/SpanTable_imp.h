#pragma once

#include "Neon/domain/tools/SpanTable.h"

namespace Neon::domain::tool {

template <typename IndexSpace>
SpanTable<IndexSpace>::SpanTable(const Neon::Backend& bk)
{
    for (auto dw : Neon::DataViewUtil::validOptions()) {
        mSpanTable[Neon::DataViewUtil::toInt(dw)] =
            bk.devSet().template newDataSet<IndexSpace>();
    }
    mSetSize = bk.devSet().setCardinality();
}

template <typename IndexSpace>
auto SpanTable<IndexSpace>::
    getSpan(Neon::SetIdx   setIdx,
            Neon::DataView dw)
        -> IndexSpace&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    auto&     output = mSpanTable[dwInt][setIdx.idx()];
    return output;
}

template <typename IndexSpace>
auto SpanTable<IndexSpace>::
    getSpan(Neon::SetIdx   setIdx,
            Neon::DataView dw)
        const -> const IndexSpace&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    auto&     output = mSpanTable[dwInt][setIdx.idx()];
    return output;
}

template <typename IndexSpace>
template <class Lambda>
auto SpanTable<IndexSpace>::forEachConfiguration(const Lambda& lambda)
    -> void
{
    for (auto dw : Neon::DataViewUtil::validOptions()) {
        for (auto setIdx = 0; setIdx < mSetSize; setIdx++) {
            lambda(setIdx, dw, getSpan(setIdx, dw));
        }
    }
}

template <typename IndexSpace>
auto SpanTable<IndexSpace>::getSpan(Neon::DataView dw)
    const -> const Neon::set::DataSet<IndexSpace>&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    auto&     output = mSpanTable[dwInt];
    return output;
}
}  // namespace Neon::domain::tool