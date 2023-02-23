#include "Neon/domain/internal/experimental/dGrid/dGrid.h"

namespace Neon::domain::internal::exp::dGrid {

dGrid::dGrid()
{
    mData = std::make_shared<Data>();
}

dGrid::Data::Data(const Neon::Backend& backend)
{
    partitionDims = backend.devSet().newDataSet<index_3d>({0, 0, 0});
    firstZIndex = backend.devSet().newDataSet<index_t>(0);
    spanTable = Neon::domain::tool::SpanTable<dSpan>(backend);
    elementsPerPartition = Neon::domain::tool::SpanTable<int>(backend);

    halo = index_3d(0, 0, 0);
    reduceEngine = Neon::sys::patterns::Engine::cuBlas;
}

auto dGrid::getSpan(SetIdx         setIdx,
                    Neon::DataView dataView)
    const -> const Span&
{
   return mData->spanTable.getSpan(setIdx, dataView);
}


auto dGrid::helpGetPartitionDim()
    const -> const Neon::set::DataSet<index_3d>
{
    return mData->partitionDims;
}

auto dGrid::helpIdexPerPartition(Neon::DataView dataView)
    const -> const Neon::set::DataSet<int>
{
    return mData->elementsPerPartition.getSpan(dataView);
}

auto dGrid::setReduceEngine(Neon::sys::patterns::Engine eng)
    -> void
{
    mData->reduceEngine = eng;
}

auto dGrid::getLaunchParameters(const Neon::DataView  dataView,
                                const Neon::index_3d& blockSize,
                                const size_t&         shareMem) const -> Neon::set::LaunchParameters
{
    Neon::set::LaunchParameters ret = getBackend().devSet().newLaunchParameters();

    auto dimsByDataView = getBackend().devSet().newDataSet<index_3d>([&](Neon::SetIdx const& setIdx,
                                                                         auto&               value) {
        value = getSpan(setIdx, dataView).mDim;
    });

    ret.set(Neon::sys::GpuLaunchInfo::domainGridMode,
            dimsByDataView,
            blockSize,
            shareMem);
    return ret;
}

auto dGrid::convertToNghIdx(const std::vector<Neon::index_3d>& stencilOffsets)
    const -> std::vector<NghIdx>
{
    std::vector<NghIdx> res;
    for (const auto& offset : stencilOffsets) {
        res.push_back(offset.template newType<int8_t>());
    }
    return res;
}

auto dGrid::convertToNghIdx(Neon::index_3d const& stencilOffsets)
    const -> NghIdx
{
    return stencilOffsets.template newType<int8_t>();
}

auto dGrid::isInsideDomain(const index_3d& idx) const -> bool
{
    bool isPositive = idx >= 0;
    bool isLover = idx < this->getDimension();
    return isLover && isPositive;
}

auto dGrid::getProperties(const index_3d& idx) const -> GridBaseTemplate::CellProperties
{
    GridBaseTemplate::CellProperties cellProperties;
    cellProperties.setIsInside(isInsideDomain(idx));
    if (!cellProperties.isInside()) {
        return cellProperties;
    }

    if (this->getDevSet().setCardinality() == 1) {
        cellProperties.init(0, DataView::INTERNAL);
    } else {
        int            zCounter = 0;
        int            zCounterPrevious = 0;
        Neon::SetIdx   setIdx;
        Neon::DataView dataView = DataView::BOUNDARY;
        for (int i = 0; i < this->getDevSet().setCardinality(); i++) {
            zCounter += mData->partitionDims[i].z;
            if (idx.z < zCounter) {
                setIdx = i;
            }
            if ((zCounterPrevious + mData->halo.z >= idx.z) &&
                (zCounter - mData->halo.z < idx.z)) {
                dataView = Neon::DataView::INTERNAL;
            }
            zCounterPrevious = zCounter;
        }
        cellProperties.init(setIdx, dataView);
    }
    return cellProperties;
}
auto dGrid::helpGetFirstZindex() const -> const Neon::set::DataSet<int32_t>&
{
    return mData->firstZIndex;
}
}  // namespace Neon::domain::internal::exp::dGrid