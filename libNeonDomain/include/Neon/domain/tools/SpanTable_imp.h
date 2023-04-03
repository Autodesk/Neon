#pragma once

#include "Neon/domain/tools/SpanTable.h"

namespace Neon::domain::tool {

template <typename IndexSpace>
SpanTable<IndexSpace>::SpanTable(const Neon::Backend& bk)
{
    for (auto execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            int const dwInt = Neon::DataViewUtil::toInt(dw);
            int const execInt = Neon::ExecutionUtils::toInt(execution);

            mSpanTable[execInt][dwInt] =
                bk.devSet().template newDataSet<IndexSpace>();
        }
    }
    mSetSize = bk.devSet().setCardinality();
}

template <typename IndexSpace>
auto SpanTable<IndexSpace>::
    getSpan(Neon::Execution execution,
            Neon::SetIdx    setIdx,
            Neon::DataView  dw)
        -> IndexSpace&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    int const execInt = Neon::ExecutionUtils::toInt(execution);
    auto&     output = mSpanTable[execInt][dwInt][setIdx.idx()];
    return output;
}

template <typename IndexSpace>
auto SpanTable<IndexSpace>::
    getSpan(Neon::Execution execution,
            Neon::SetIdx    setIdx,
            Neon::DataView  dw)
        const -> const IndexSpace&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    int const execInt = Neon::ExecutionUtils::toInt(execution);
    auto&     output = mSpanTable[execInt][dwInt][setIdx.idx()];
    return output;
}

template <typename IndexSpace>
template <class Lambda>
auto SpanTable<IndexSpace>::forEachConfiguration(const Lambda& lambda)
    -> void
{
    for (auto execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            for (auto setIdx = 0; setIdx < mSetSize; setIdx++) {
                lambda(execution, setIdx, dw, getSpan(execution, setIdx, dw));
            }
        }
    }
}

template <typename IndexSpace>
auto SpanTable<IndexSpace>::getSpan(Neon::Execution execution,
                                    Neon::DataView  dw)
    const -> const Neon::set::DataSet<IndexSpace>&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    int const execInt = Neon::ExecutionUtils::toInt(execution);
    auto&     output = mSpanTable[execInt][dwInt];
    return output;
}
}  // namespace Neon::domain::tool