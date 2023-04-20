#pragma once

#include "Neon/set/Backend.h"
#include "Neon/set/DataSet.h"

namespace Neon::domain::tool {

/**
 * A helper class to storage and access IndexSpaces parametrically w.r.t Neon::DataView and Neon::Executions
 */
template <typename Span>
struct SpanTable
{
    SpanTable() = default;

    explicit SpanTable(const Neon::Backend& bk);

    auto init(const Neon::Backend& bk)-> void;


    auto getSpan(Neon::Execution execution,
                 Neon::SetIdx   setIdx,
                 Neon::DataView dw)
        -> Span&;

    auto getSpan(Neon::Execution execution,
                 Neon::SetIdx    setIdx,
                 Neon::DataView  dw)
        const -> const Span&;

    auto getSpan(Neon::Execution execution,
                 Neon::DataView dw)
        const -> const Neon::set::DataSet<Span>&;

    template <class Lambda>
    auto forEachConfiguration(const Lambda& lambda) -> void;

   private:
    using SpanByDevice = Neon::set::DataSet<Span>;
    using SpanByDeviceByDataView = std::array<SpanByDevice, Neon::DataViewUtil::nConfig>;
    using SpanByDeviceByDataViewByExecution = std::array<SpanByDeviceByDataView, Neon::ExecutionUtils::numConfigurations>;

    SpanByDeviceByDataViewByExecution mSpanTable;
    int                               mSetSize = 0;
};


}  // namespace Neon::domain::tool

#include "Neon/domain/tools/SpanTable_imp.h"
