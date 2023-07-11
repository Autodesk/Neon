#include "Neon/domain/details/eGrid/eGrid.h"

namespace Neon::domain::details::eGrid {


eGrid::eGrid(const Backend&                     backend,
             const int32_3d&                    dimension,
             Neon::domain::tool::Partitioner1D& partitioner,
             const Stencil&                     stencil,
             const Vec_3d<double>&              spacing,
             const Vec_3d<double>&              origin)
{
    mData = std::make_shared<Data>(backend);
    mData->stencil = stencil;
    const index_3d defaultBlockSize(256, 1, 1);

    {
        auto nElementsPerPartition = backend.devSet().template newDataSet<size_t>(0);
        // We do an initialization with nElementsPerPartition to zero,
        // then we reset to the computed number.
        eGrid::GridBase::init("eGrid",
                              backend,
                              dimension,
                              stencil,
                              nElementsPerPartition,
                              Neon::index_3d(256, 1, 1),
                              spacing,
                              origin);
    }


    mData->partitioner1D = partitioner;

    mData->mConnectivityAField = mData->partitioner1D.getConnectivity();
    mData->mGlobalMappingAField = mData->partitioner1D.getGlobalMapping();
    mData->mStencil3dTo1dOffset = mData->partitioner1D.getStencil3dTo1dOffset();
    mData->memoryGrid = mData->partitioner1D.getMemoryGrid();
    //mData->partitioner1D.getDenseMeta(mData->denseMeta);

    const int32_t numDevices = getBackend().devSet().setCardinality();

    if (numDevices > 1 && getDimension().z < numDevices) {
        NeonException exc("dGrid_t");
        exc << "The grid size in the z-direction (" << getDimension().z << ") is less the number of devices (" << numDevices
            << "). It is ambiguous how to distribute the gird";
        NEON_THROW(exc);
    }


    {
        // Initialization of the SPAN table
        mData->spanTable.forEachConfiguration([&](Neon::Execution /*execution*/,
                                                  Neon::SetIdx   setIdx,
                                                  Neon::DataView dw,
                                                  eSpan&         span) {
            span.mDataView = dw;
            switch (dw) {
                case Neon::DataView::STANDARD: {
                    int countPerPartition = 0;
                    countPerPartition += mData->partitioner1D.getSpanClassifier().countInternal(setIdx);
                    countPerPartition += mData->partitioner1D.getSpanClassifier().countBoundary(setIdx);

                    span.mCount = countPerPartition;
                    span.mFirstIndexOffset = 0;
                    span.mDataView = dw;

                    break;
                }
                case Neon::DataView::BOUNDARY: {
                    int countPerPartition = 0;
                    countPerPartition += mData->partitioner1D.getSpanClassifier().countBoundary(setIdx);

                    span.mCount = countPerPartition;
                    span.mFirstIndexOffset = mData->partitioner1D.getSpanClassifier().countInternal(setIdx);
                    span.mDataView = dw;

                    break;
                }
                case Neon::DataView::INTERNAL: {
                    int countPerPartition = 0;
                    countPerPartition += mData->partitioner1D.getSpanClassifier().countInternal(setIdx);

                    span.mCount = countPerPartition;
                    span.mFirstIndexOffset = 0;
                    span.mDataView = dw;
                    break;
                }
                default: {
                    NeonException exc("dFieldDev");
                    NEON_THROW(exc);
                }
            }
        });

        mData->elementsPerPartition.forEachConfiguration([&](Neon::Execution execution,
                                                             Neon::SetIdx    setIdx,
                                                             Neon::DataView  dw,
                                                             int&            count) {
            count = mData->spanTable.getSpan(execution, setIdx, dw).mCount;
        });
    }

    {  // Init base class information
        Neon::set::DataSet<size_t> nElementsPerPartition = mData->partitioner1D.getStandardCount().template newType<size_t>();
        eGrid::GridBase::init("eGrid",
                              backend,
                              dimension,
                              stencil,
                              nElementsPerPartition,
                              defaultBlockSize,
                              spacing,
                              origin);
    }
}


eGrid::eGrid()
{
    mData = std::make_shared<Data>();
}

eGrid::Data::Data(Neon::Backend const& backend)
{
    spanTable = Neon::domain::tool::SpanTable<eSpan>(backend);
    elementsPerPartition = Neon::domain::tool::SpanTable<int>(backend);
    reduceEngine = Neon::sys::patterns::Engine::cuBlas;
}


auto eGrid::getMemoryGrid() -> Neon::aGrid&
{
    return mData->memoryGrid;
}

auto eGrid::getSpan(Neon::Execution execution,
                    SetIdx          setIdx,
                    Neon::DataView  dataView)
    const -> const Span&
{
    return mData->spanTable.getSpan(execution, setIdx, dataView);
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
        value.x = getSpan(Neon::Execution::host, setIdx, dataView).mCount;
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
    //auto const& metaInfo = mData->denseMeta.get(idx);
    auto const& metaInfo = mData->partitioner1D.getDenseMeta().get(idx);
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
        //auto const& metaInfo = mData->denseMeta.get(idx);
        auto const& metaInfo = mData->partitioner1D.getDenseMeta().get(idx);
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

auto eGrid::helpGetSetIdxAndGridIdx(Neon::index_3d idx) const -> std::tuple<Neon::SetIdx, eIndex>
{
    eIndex eIndex;
    SetIdx setIdx;
    setIdx.invalidate();
    {
        //auto const& meta = mData->denseMeta.get(idx);
        auto const& meta = mData->partitioner1D.getDenseMeta().get(idx);
        if (meta.isValid()) {
            auto const& span = getSpan(Execution::host, meta.setIdx, Neon::DataView::STANDARD);
            span.setAndValidate(eIndex, meta.index);
            setIdx = meta.setIdx;
        }
    }
    return {setIdx, eIndex};
}


}  // namespace Neon::domain::details::eGrid