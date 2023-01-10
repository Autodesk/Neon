#include "Neon/domain/internal/aGrid/aGrid.h"

namespace Neon::domain::internal::aGrid {

aGrid::aGrid()
    : Neon::domain::interface::GridBaseTemplate<aGrid, aCell>()
{
    mStorage = std::make_shared<Storage>();
};

aGrid::aGrid(const Neon::Backend&  backend,
             const Neon::int32_3d& dimension,
             const int32_3d&       blockDim,
             const Vec_3d<double>& spacingData,
             const Vec_3d<double>& origin)
{
    if (dimension.y != 1 || dimension.z != 1) {
        NEON_THROW_UNSUPPORTED_OPTION("aGrid only support 1D grids.");
    }
    Neon::set::DataSet<size_t> lenghts = backend.devSet().newDataSet<size_t>();
    for (int idx = 0; idx < backend.devSet().setCardinality(); idx++) {
        size_t count = dimension.x / backend.devSet().setCardinality();
        size_t reminder = dimension.x % backend.devSet().setCardinality();
        if (reminder > static_cast<size_t>(idx)) {
            count++;
        }
        lenghts[idx] = count;
    }

    this->init(backend, dimension, lenghts, blockDim, spacingData, origin);
}


aGrid::aGrid(const Neon::Backend&              backend,
             const Neon::set::DataSet<size_t>& lenghts,
             const int32_3d&                   blockDim,
             const Vec_3d<double>&             spacingData,
             const Vec_3d<double>&             origin)
{
    Neon::int32_3d dimension(0, 0, 0);

    for (int idx = 0; idx < backend.devSet().setCardinality(); idx++) {
        dimension.x += int(lenghts[idx]);
    }

    init(backend, dimension, lenghts,
         blockDim, spacingData, origin);
}

auto aGrid::init(const Neon::Backend&              backend,
                 const Neon::int32_3d&             dimension,
                 const Neon::set::DataSet<size_t>& lenghts,
                 const int32_3d&                   blockDim,
                 const Vec_3d<double>&             spacingData,
                 const Vec_3d<double>&             origin)
    -> void
{
    aGrid::GridBase::init("aGrid",
                          backend,
                          dimension,
                          Neon::domain::Stencil(),
                          lenghts,
                          blockDim,
                          spacingData,
                          origin);

    mStorage = std::make_shared<Storage>();

    setDefaultBlock(blockDim);

    mStorage->partitionSpaceSet = newDataSet<PartitionIndexSpace>();
    for (int setIdx = 0; setIdx < mStorage->partitionSpaceSet.cardinality(); setIdx++) {
        mStorage->partitionSpaceSet[setIdx] =
            PartitionIndexSpace(int(getNumActiveCellsPerPartition()[setIdx]), Neon::DataView::STANDARD);
    }

    for (int i = 0; i < getDevSet().setCardinality(); i++) {
        for (auto indexing : {Neon::DataView::STANDARD}) {
            getDefaultLaunchParameters(indexing) = getLaunchParameters(indexing, blockDim, 0);
        }
    }

    mStorage->firstIdxPerPartition = newDataSet<size_t>();
    mStorage->firstIdxPerPartition[0] = 0;
    size_t tmpCount = 0;
    for (int i = 1; i < getDevSet().setCardinality(); i++) {
        tmpCount += this->getNumActiveCellsPerPartition()[i - 1];
        mStorage->firstIdxPerPartition[i] = tmpCount;
    }
}

auto aGrid::getLaunchParameters(Neon::DataView        dataView,
                                const Neon::index_3d& blockDim,
                                size_t                shareMem) const -> Neon::set::LaunchParameters
{
    auto launchParameters = this->getBackend().devSet().newLaunchParameters();
    if (blockDim.y != 1 || blockDim.z != 1) {
        NeonException exc("eGrid");
        exc << "CUDA block size should be 1D\n";
        NEON_THROW(exc);
    }

    if (dataView != Neon::DataView::STANDARD) {
        NEON_WARNING("aGrid: Request of non standard view.");
    }

    for (int i = 0; i < getDevSet().setCardinality(); i++) {

        auto gridMode = Neon::sys::GpuLaunchInfo::mode_e::domainGridMode;
        auto grid1DLength = getNumActiveCellsPerPartition()[i];
        launchParameters[i].set(gridMode,
                                grid1DLength,
                                blockDim,
                                shareMem);
    }
    return launchParameters;
}

auto aGrid::getFirstIdxPerPartition() const
    -> const Neon::set::DataSet<size_t>&
{
    return mStorage->firstIdxPerPartition;
}

auto aGrid::getPartitionIndexSpace(Neon::DeviceType /* devE */,
                                   SetIdx         setIdx,
                                   Neon::DataView dataView) const
    -> const aGrid::PartitionIndexSpace&
{
    if (dataView != Neon::DataView::STANDARD) {
        NEON_THROW_UNSUPPORTED_OPERATION("aGrid only support STANDARD view");
    }
    return mStorage->partitionSpaceSet[setIdx];
}

auto aGrid::setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig) const -> void
{
    if (gridKernelConfig.runtime() != Neon::Runtime::system) {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }
    gridKernelConfig.expertSetLaunchParameters([&](Neon::set::LaunchParameters& launchParameters) {
        launchParameters = getDefaultLaunchParameters(gridKernelConfig.dataView());
    });
    gridKernelConfig.expertSetBackend(getBackend());

    return;
}

auto aGrid::getKernelConfig(int,
                            Neon::DataView) -> Neon::set::KernelConfig
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}
auto aGrid::isInsideDomain(const index_3d& idx) const -> bool
{
    if (idx.y != 0 || idx.z != 0) {
        return false;
    }

    if (idx >= 0 && idx < this->getDimension()) {
        return true;
    }
    return false;
}
auto aGrid::getProperties(const index_3d& cell3dIdx) const -> GridBaseTemplate::CellProperties
{
    GridBaseTemplate::CellProperties cellProperties;

    cellProperties.setIsInside(isInsideDomain(cell3dIdx));
    if (!cellProperties.isInside()) {
        return cellProperties;
    }

    Neon::SetIdx targetSetIdx;
    for (int setIdx = 0; setIdx < this->getDevSet().setCardinality(); setIdx++) {
        auto firstId = mStorage->firstIdxPerPartition[setIdx];
        if (cell3dIdx.x >= int(firstId)) {
            targetSetIdx = setIdx;
            break;
        }
    }
    cellProperties.init(targetSetIdx, Neon::DataView::STANDARD);
    return cellProperties;
}

#define AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION(TYPE)                                \
    template auto aGrid::newField<TYPE, 0>(const std::string   fieldUserName,       \
                                           int                 cardinality,         \
                                           TYPE                inactiveValue,       \
                                           Neon::DataUse       dataUse,             \
                                           Neon::MemoryOptions memoryOptions) const \
        ->Field<TYPE, 0>;

AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION(double)
AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION(float)
AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION(int32_t)
AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION(int64_t)

#undef AGRID_NEW_FIELD_EXPLICIT_INSTANTIATION

template class aField<int, 0>;
template class aField<double, 0>;


}  // namespace Neon::domain::internal::aGrid
