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
    GridBaseTemplate::CellProperties cellProperties;

    cellProperties.setIsInside(this->isInsideDomain(idx));
    if (!cellProperties.isInside()) {
        return -1;
    }
    Neon::SetIdx setIdx = cellProperties.getSetIdx();
    return setIdx;
}
auto bGrid::getLaunchParameters(Neon::DataView dataView,
                                const index_3d&,
                                const size_t& sharedMem) const -> Neon::set::LaunchParameters
{
    auto res = mData->launchParametersTable.get(dataView);
    res.forEachSeq([&](SetIdx const& /*setIdx*/,
                       Neon::set::LaunchParameters::launchInfo_e& launchParams) -> void {
        launchParams.setShm(sharedMem);
    });
    return res;
}

auto bGrid::
    helpGetStencilIdTo3dOffset()
        const -> Neon::set::MemSet<Neon::int8_3d>&
{
    return mData->stencilIdTo3dOffset;
}

auto bGrid::isInsideDomain(const index_3d& idx) const -> bool
{
    // 1. check if the block is active
    const index_3d blockIdx3d = idx / this->mData->dataBlockSize;
    auto           blockProperties = mData->blockViewGrid.getProperties(blockIdx3d);

    if (!blockProperties.isInside()) {
        return false;
    }
    // 2. The block is active, check the element on the block
    uint32_t              wordCardinality;
    Span::BitMaskWordType mask;
    Span::getMaskAndWordIdforBlockBitMask(idx.x % this->mData->dataBlockSize,
                                          idx.y % this->mData->dataBlockSize,
                                          idx.z % this->mData->dataBlockSize,
                                          this->mData->dataBlockSize,
                                          NEON_OUT mask,
                                          NEON_OUT wordCardinality);
    auto activeBits = mData->activeBitMask.getReference(blockIdx3d, int(wordCardinality));
    return (activeBits & mask) != 0;
}

auto bGrid::getProperties(const index_3d& idx) const -> GridBaseTemplate::CellProperties
{
    GridBaseTemplate::CellProperties cellProperties;

    cellProperties.setIsInside(this->isInsideDomain(idx));
    if (!cellProperties.isInside()) {
        return cellProperties;
    }

    if (this->getDevSet().setCardinality() == 1) {
        cellProperties.init(0, DataView::INTERNAL);
    } else {
        const index_3d blockIdx3d = idx / this->mData->dataBlockSize;
        auto           blockViewProperty = mData->blockViewGrid.getProperties(blockIdx3d);

        cellProperties.init(blockViewProperty.getSetIdx(),
                            blockViewProperty.getDataView());
    }
    return cellProperties;
}

auto bGrid::helpGetSetIdxAndGridIdx(Neon::index_3d idx)
    const -> std::tuple<Neon::SetIdx, Idx>
{
    const index_3d blockIdx3d = idx / this->mData->dataBlockSize;
    auto [setIdx, bvGridIdx] = mData->blockViewGrid.helpGetSetIdxAndGridIdx(blockIdx3d);
    Idx bIdx;
    bIdx.mDataBlockIdx = bvGridIdx.helpGet();
    bIdx.mInDataBlockIdx.x = idx.x % this->mData->dataBlockSize;
    bIdx.mInDataBlockIdx.y = idx.y % this->mData->dataBlockSize;
    bIdx.mInDataBlockIdx.z = idx.z % this->mData->dataBlockSize;

    return {setIdx, bIdx};
}

}  // namespace Neon::domain::details::bGrid