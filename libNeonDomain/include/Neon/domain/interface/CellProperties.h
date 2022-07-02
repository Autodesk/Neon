#pragma once

#include <string>

#include "Neon/core/core.h"

#include "Neon/set/Backend.h"
#include "Neon/set/DataSet.h"
#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/GridBase.h"

namespace Neon::domain::interface {

template <typename CellT>
struct CellProperties
{
   public:
    using Cell = CellT;

    CellProperties() = default;

    auto init(Neon::SetIdx, Neon::DataView)
        -> void;

    auto init(Neon::SetIdx, Neon::DataView, typename Cell::OuterCell const&)
        -> void;

    auto setIsInside(bool isInside)
        -> void;

    auto isInside() const
        -> bool;

    /**
     * Returns the relative index of the device owning the cell.
     * @return
     */
    auto getSetIdx() const
        -> Neon::SetIdx;

    /**
     * Returns DataView information for the cell.
     * Possible values are {internal or Boundary}
     *
     * @return
     */
    auto getDataView() const
        -> Neon::DataView;

    /**
     * Returns the OuterCell abstraction for the cell.
     * OuterCell abstractions are used by sGrid to convert sGrid local indexing to
     * a OuterGrid cell handle. The handle may be of the same type as the OuterCell Cell type
     * or different.
     *
     * For example, eCell define a handle for eGrid that is just an int32_t.
     * eGrid defines its OuterCell as alias it eCell.
     *
     * On the other hand, dCell is wrapper for index_3d type. For a more efficient storage use of sGrid,
     * dGrid could define its OuterGrid as simple int32_t.
     */
    auto getOuterCell() const
        -> typename Cell::OuterCell const&;

   private:
    bool                     mIsInsise;
    Neon::SetIdx             mSetIdx;
    DataView                 mDataView;
    typename Cell::OuterCell mOuterCell;
};

template <typename CellT>
auto CellProperties<CellT>::init(Neon::SetIdx   setIdx,
                                 Neon::DataView dataView) -> void
{
    mSetIdx = setIdx;
    mDataView = dataView;
}

template <typename CellT>
auto CellProperties<CellT>::init(Neon::SetIdx                     setIdx,
                                 Neon::DataView                   dataView,
                                 typename CellT::OuterCell const& outerCell) -> void
{
    mSetIdx = setIdx;
    mDataView = dataView;
    mOuterCell = outerCell;
}

template <typename CellT>
auto CellProperties<CellT>::setIsInside(bool isInside) -> void
{
    mIsInsise = isInside;
}

template <typename CellT>
auto CellProperties<CellT>::isInside() const -> bool
{
    return mIsInsise;
}

template <typename CellT>
auto CellProperties<CellT>::getSetIdx() const -> Neon::SetIdx
{
    return mSetIdx;
}

template <typename CellT>
auto CellProperties<CellT>::getDataView() const -> Neon::DataView
{
    return mDataView;
}
template <typename CellT>
auto CellProperties<CellT>::getOuterCell() const -> typename Cell::OuterCell const&
{
    return mOuterCell;
}


}  // namespace Neon::domain::interface
