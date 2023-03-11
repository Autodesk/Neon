#pragma once

#include <string>

#include "Neon/core/core.h"

#include "Neon/set/Backend.h"
#include "Neon/set/DataSet.h"
#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/GridBase.h"

namespace Neon::domain::interface {

template <typename IndexT>
struct IndexProperties
{
   public:
    using Index = IndexT;

    IndexProperties() = default;

    auto init(Neon::SetIdx, Neon::DataView)
        -> void;

    auto init(Neon::SetIdx, Neon::DataView, typename Index::OuterIndex const&)
        -> void;

    auto setIsInside(bool isInside)
        -> void;

    auto isInside() const
        -> bool;

    /**
     * Returns the relative index of the device owning the Index.
     * @return
     */
    auto getSetIdx() const
        -> Neon::SetIdx;

    /**
     * Returns DataView information for the Index.
     * Possible values are {internal or Boundary}
     *
     * @return
     */
    auto getDataView() const
        -> Neon::DataView;

    /**
     * Returns the OuterIndex abstraction for the Index.
     * OuterIndex abstractions are used by sGrid to convert sGrid local indexing to
     * a OuterGrid Index handle. The handle may be of the same type as the OuterIndex Index type
     * or different.
     *
     * For example, eIndex define a handle for eGrid that is just an int32_t.
     * eGrid defines its OuterIndex as alias it eIndex.
     *
     * On the other hand, dIndex is wrapper for index_3d type. For a more efficient storage use of sGrid,
     * dGrid could define its OuterGrid as simple int32_t.
     */
    auto getOuterIndex() const
        -> typename Index::OuterIndex const&;

   private:
    bool                       mIsInsise;
    Neon::SetIdx               mSetIdx;
    DataView                   mDataView;
    typename Index::OuterIndex mOuterIndex;
};

template <typename IndexT>
auto IndexProperties<IndexT>::init(Neon::SetIdx   setIdx,
                                   Neon::DataView dataView) -> void
{
    mSetIdx = setIdx;
    mDataView = dataView;
}

template <typename IndexT>
auto IndexProperties<IndexT>::init(Neon::SetIdx                       setIdx,
                                   Neon::DataView                     dataView,
                                   typename IndexT::OuterIndex const& outerIndex) -> void
{
    mSetIdx = setIdx;
    mDataView = dataView;
    mOuterIndex = outerIndex;
}

template <typename IndexT>
auto IndexProperties<IndexT>::setIsInside(bool isInside) -> void
{
    mIsInsise = isInside;
}

template <typename IndexT>
auto IndexProperties<IndexT>::isInside() const -> bool
{
    return mIsInsise;
}

template <typename IndexT>
auto IndexProperties<IndexT>::getSetIdx() const -> Neon::SetIdx
{
    return mSetIdx;
}

template <typename IndexT>
auto IndexProperties<IndexT>::getDataView() const -> Neon::DataView
{
    return mDataView;
}
template <typename IndexT>
auto IndexProperties<IndexT>::getOuterIndex() const -> typename Index::OuterIndex const&
{
    return mOuterIndex;
}


}  // namespace Neon::domain::interface
