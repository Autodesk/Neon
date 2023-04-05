#include "Neon/domain/details/bGrid/bGrid.h"

namespace Neon::domain::details::bGrid {

auto bGrid::
    helpGetDataBlockSize()
        const -> int
{
    return mData->dataBlockSize;
}

auto bGrid::
    helpGetBlockViewGrid()
        const -> BlockViewGrid&
{
    return mData->blockViewGrid;
}

auto bGrid::
    helpGetActiveBitMask()
        const -> BlockViewGrid::Field<uint64_t, 0>&
{
    return mData->activeBitMask;
}

auto bGrid::
    helpGetBlockConnectivity()
        const -> BlockViewGrid::Field<BlockIdx, 27>&
{
    return mData->blockConnectivity;
}
auto bGrid::
    helpGetDataBlockOriginField()
        const -> Neon::aGrid::Field<index_3d, 0>&
{
    return mData->mDataBlockOriginField;
}
auto bGrid::getSpan(Neon::Execution  execution,
                    SetIdx           setIdx,
                    Neon::DataView   dataView) -> const bGrid::Span&
{
    return mData->spanTable.getSpan(execution, setIdx, dataView);
}

}  // namespace Neon::domain::details::bGrid