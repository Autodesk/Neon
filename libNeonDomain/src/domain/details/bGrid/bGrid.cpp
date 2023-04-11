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
auto bGrid::getSpan(Neon::Execution execution,
                    SetIdx          setIdx,
                    Neon::DataView  dataView) -> const bGrid::Span&
{
    return mData->spanTable.getSpan(execution, setIdx, dataView);
}

bGrid::~bGrid()
{
}
auto bGrid::getSetIdx(const index_3d& idx) const -> int32_t
{
    auto const& decomposition = mData->partitioner1D.getDecomposition();
    decomposition.getSetIdx(const index_3d& idx);
}
auto bGrid::getLaunchParameters(Neon::DataView dataView,
                                const index_3d&,
                                const size_t& sharedMem) const -> Neon::set::LaunchParameters
{
    auto res = mData->launchParameters[Neon::DataViewUtil::toInt(dataView)];
    res.forEachSeq([&](SetIdx const& setIdx, Neon::set::LaunchParameters::launchInfo_e& launchParams) -> void {
        launchParams.setShm(sharedMem)
    });
    return mData->launchParameters[Neon::DataViewUtil::toInt(dataView)];
}

}  // namespace Neon::domain::details::bGrid