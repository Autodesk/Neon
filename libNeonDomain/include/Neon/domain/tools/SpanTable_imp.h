#pragma once

#include "Neon/domain/tools/SpanTable.h"

namespace Neon::domain::tool {

template <typename IndexSpace>
SpanTable<IndexSpace>::SpanTable(const Neon::Backend& bk)
{
    for (auto dw : Neon::DataViewUtil::validOptions()) {
        mIndexSpaceTable[Neon::DataViewUtil::toInt(dw)] =
            bk.devSet().template newDataSet<IndexSpace>();
    }
    mSetSize = bk.devSet().getCardianlity();
}

template <typename IndexSpace>
auto SpanTable<IndexSpace>::
    getIndexSpace(Neon::SetIdx   setIdx,
                  Neon::DataView dw)
        -> IndexSpace&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    auto&     output = mIndexSpaceTable[dwInt][setIdx.idx()];
    return output;
}

template <typename IndexSpace>
auto SpanTable<IndexSpace>::
    getIndexSpace(Neon::SetIdx   setIdx,
                  Neon::DataView dw)
        const -> const IndexSpace&
{
    int const dwInt = Neon::DataViewUtil::toInt(dw);
    auto&     output = mIndexSpaceTable[dwInt][setIdx.idx()];
    return output;
}

template <typename IndexSpace>
template <class Lambda>
auto SpanTable<IndexSpace>::forEachConfiguration(const Lambda& lambda)
    -> void
{
    for (Neon::Execution execution : Neon::ExecutionUtils::getAllOptions()) {
        for (auto dw : Neon::DataViewUtil::validOptions()) {
            for (auto setIdx = 0; setIdx < mSetSize; setIdx++) {
                lambda(setIdx, dw, getIndexSpace(setIdx, dw));
            }
        }
    }
}
}  // namespace Neon::domain::tool