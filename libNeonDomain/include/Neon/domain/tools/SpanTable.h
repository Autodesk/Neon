#pragma once
#include "Neon/set/Backend.h"
#include "Neon/set/DataSet.h"
namespace Neon::domain::tool {

/**
 * A helper class to storage and access IndexSpaces parametrically w.r.t Neon::DataView and Neon::Executions
 */
template <typename IndexSpace>
struct SpanTable
{
    SpanTable() = default;

    explicit SpanTable(const Neon::Backend& bk);

    auto getIndexSpace(Neon::SetIdx   setIdx,
                       Neon::DataView dw)
        -> IndexSpace&;

    auto getIndexSpace(Neon::SetIdx   setIdx,
                       Neon::DataView dw)
        const -> const IndexSpace&;

    template <class Lambda>
    auto forEachConfiguration(const Lambda& lambda) -> void;

   private:
    using SpanByDevice = Neon::set::DataSet<IndexSpace>;
    using SpanByDeviceByDataView = std::array<SpanByDevice, Neon::DataViewUtil::nConfig>;

    SpanByDeviceByDataView mIndexSpaceTable;
    int                    mSetSize = 0;
};

}  // namespace Neon::domain::tool
