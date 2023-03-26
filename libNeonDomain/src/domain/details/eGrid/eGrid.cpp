#include "Neon/domain/details/eGrid/eGrid.h"

namespace Neon::domain::details::eGrid {


eGrid::eGrid()
{
    mData = std::make_shared<Data>();
}

eGrid::Data::Data(Neon::Backend const& backend)
{
    spanTable = Neon::domain::tool::SpanTable<eSpan>(backend);
    elementsPerPartition = Neon::domain::tool::SpanTable<int>(backend);

    halo = index_3d(0, 0, 0);
    reduceEngine = Neon::sys::patterns::Engine::cuBlas;
}


auto eGrid::getMemoryGrid() -> Neon::aGrid&
{
    return mData->memoryGrid;
}

auto eGrid::getSpan(SetIdx         setIdx,
                    Neon::DataView dataView)
    const -> const Span&
{
    return mData->spanTable.getSpan(setIdx, dataView);
}

auto eGrid::setReduceEngine(Neon::sys::patterns::Engine eng)
    -> void
{
    mData->reduceEngine = eng;
}

auto eGrid::getLaunchParameters(const Neon::DataView  dataView,
                                const Neon::index_3d& blockSize,
                                const size_t&         shareMem) const -> Neon::set::LaunchParameters
{
    Neon::set::LaunchParameters ret = getBackend().devSet().newLaunchParameters();

    auto dimsByDataView = getBackend().devSet().newDataSet<index_3d>([&](Neon::SetIdx const& setIdx,
                                                                         auto&               value) {
        value.x = getSpan(setIdx, dataView).mCount;
        value.y = 1;
        value.z = 1;
    });

    ret.set(Neon::sys::GpuLaunchInfo::domainGridMode,
            dimsByDataView,
            blockSize,
            shareMem);
    return ret;
}

auto eGrid::convertToNghIdx(const std::vector<Neon::index_3d>& stencilOffsets)
    const -> std::vector<NghIdx>
{
    std::vector<NghIdx> res;
    for (const auto& offset : stencilOffsets) {
        NghIdx newItem = convertToNghIdx(offset);
        res.push_back(newItem);
    }
    return res;
}

auto eGrid::convertToNghIdx(Neon::index_3d const& offset)
    const -> NghIdx
{
    int  i = 0;
    bool found = false;
    for (auto const& ngh : mData->stencil.neighbours()) {
        if (ngh == offset) {
            found = true;
            break;
        }
        i++;
    }
    if (!found) {
        NeonException exception("eGrid");
        exception << "Requested stencil point was not included in the grid initialization";
        NEON_THROW(exception);
    }
    NghIdx newItem = NghIdx(i);
    return newItem;
}

auto eGrid::isInsideDomain(const index_3d& idx) const -> bool
{
    auto const& metaInfo = mData->denseMeta.get(idx);
    return metaInfo.isValid();
}

auto eGrid::getSetIdx(const Neon::index_3d& idx) const -> int32_t
{
    auto prop = getProperties(idx);
    if (!prop.isInside()) {
        return -1;
    }
    return prop.getSetIdx();
}

auto eGrid::getProperties(const index_3d& idx) const -> GridBaseTemplate::CellProperties
{
    GridBaseTemplate::CellProperties cellProperties;
    cellProperties.setIsInside(isInsideDomain(idx));
    if (!cellProperties.isInside()) {
        return cellProperties;
    }

    if (this->getDevSet().setCardinality() == 1) {
        cellProperties.init(0, DataView::INTERNAL);
    } else {
        auto const& metaInfo = mData->denseMeta.get(idx);
        cellProperties.init(metaInfo.setIdx, metaInfo.dw);
    }
    return cellProperties;
}

auto eGrid::getConnectivityField() -> Neon::aGrid::Field<int32_t, 0>
{
    return mData->mConnectivityAField;
}

auto eGrid::getGlobalMappingField() -> Neon::aGrid::Field<index_3d, 0>
{
    return mData->mGlobalMappingAField;
}

auto eGrid::getStencil3dTo1dOffset() -> Neon::set::MemSet<int8_t>
{
    return mData->mStencil3dTo1dOffset;
}
auto eGrid::getPartitioner() -> const tool::Partitioner1D&
{
    return mData->partitioner1D;
}

auto eGrid::helpGetData() -> eGrid::Data&
{
    return *mData.get();
}


}  // namespace Neon::domain::details::eGrid