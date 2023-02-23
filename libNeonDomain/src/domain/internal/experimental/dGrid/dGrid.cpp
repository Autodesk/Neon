#include "Neon/domain/internal/experimental/dGrid/dGrid.h"

namespace Neon::domain::internal::exp::dGrid {

dGrid::dGrid()
{
    mData = std::make_shared<Data>();
}


auto dGrid::getSpan(SetIdx         setIdx,
                    Neon::DataView dataView)
    const -> const Span&
{
    mData->spanTable.getSpan(setIdx, dataView);
}


auto dGrid::helpGetPartitionDim()
    const -> const Neon::set::DataSet<index_3d>
{
    return mData->partitionDims;
}

auto dGrid::helpPointsPerPartition(Neon::DataView dataView)
    const -> const Neon::set::DataSet<size_t>
{
    return flattenedPartitions(dataView);
}

auto dGrid::setReduceEngine(Neon::sys::patterns::Engine eng)
    -> void
{
    mData->reduceEngine = eng;
}

auto dGrid::flattenedPartitions(Neon::DataView dataView)
    const -> const Neon::set::DataSet<size_t>
{
    Neon::set::DataSet<size_t> flat_parts(mData->partitionDims.cardinality());
    switch (dataView) {
        case Neon::DataView::STANDARD: {
            for (int i = 0; i < flat_parts.cardinality(); ++i) {
                flat_parts[i] = mData->partitionDims[i].rMulTyped<size_t>();
            }
            return flat_parts;
        }
        case Neon::DataView::INTERNAL: {
            for (int i = 0; i < flat_parts.cardinality(); ++i) {
                flat_parts[i] = mData->partitionDims[i].rMulTyped<size_t>() -
                                2 * size_t(mData->halo.z) *
                                    size_t(mData->partitionDims[i].y) *
                                    size_t(mData->partitionDims[i].x);
            }
            return flat_parts;
        }
        case Neon::DataView::BOUNDARY: {
            for (int i = 0; i < flat_parts.cardinality(); ++i) {
                flat_parts[i] = 2 *
                                size_t(mData->halo.z) *
                                size_t(mData->partitionDims[i].y) *
                                size_t(mData->partitionDims[i].x);
            }
            return flat_parts;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPERATION("");
        }
    }
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
    -> std::vector<NghIdx>
{
    std::vector<NghIdx> res;
    for (const auto& offset : stencilOffsets) {
        res.push_back(offset.template newType<int8_t>());
    }
    return res;
}

auto dGrid::convertToNghIdx(const Neon::index_3d stencilOffsets)
    -> NghIdx
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