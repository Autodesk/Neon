#include "Neon/domain/internal/dGrid/dGrid.h"

namespace Neon::domain::internal::dGrid {

dGrid::dGrid()
{
    m_data = std::make_shared<data_t>();
}

auto dGrid::partitions() const -> const Neon::set::DataSet<index_3d>
{
    return m_data->partitionDims;
}

auto dGrid::flattenedLengthSet(Neon::DataView dataView) const -> const Neon::set::DataSet<size_t>
{
    return flattenedPartitions(dataView);
}

auto dGrid::setReduceEngine(Neon::sys::patterns::Engine eng) -> void
{
    m_data->reduceEngine = eng;
}

auto dGrid::flattenedPartitions(Neon::DataView dataView) const -> const Neon::set::DataSet<size_t>
{
    Neon::set::DataSet<size_t> flat_parts(m_data->partitionDims.cardinality());
    switch (dataView) {
        case Neon::DataView::STANDARD: {
            for (int i = 0; i < flat_parts.cardinality(); ++i) {
                flat_parts[i] = m_data->partitionDims[i].rMulTyped<size_t>();
            }
            return flat_parts;
        }
        case Neon::DataView::INTERNAL: {
            for (int i = 0; i < flat_parts.cardinality(); ++i) {
                flat_parts[i] = m_data->partitionDims[i].rMulTyped<size_t>() -
                                2 * size_t(m_data->halo.z) *
                                    size_t(m_data->partitionDims[i].y) *
                                    size_t(m_data->partitionDims[i].x);
            }
            return flat_parts;
        }
        case Neon::DataView::BOUNDARY: {
            for (int i = 0; i < flat_parts.cardinality(); ++i) {
                flat_parts[i] = 2 *
                                size_t(m_data->halo.z) *
                                size_t(m_data->partitionDims[i].y) *
                                size_t(m_data->partitionDims[i].x);
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
    int                         m_zBoundaryRadius = m_data->halo.z;

    switch (dataView) {
        case Neon::DataView::STANDARD: {
            auto dims = getDevSet().newDataSet<index_3d>();
            // Only works z partitions.
            assert(m_data->halo.x == 0 && m_data->halo.y == 0);

            for (int32_t i = 0; i < dims.size(); ++i) {
                dims[i] = m_data->partitionDims[i];
            }
            ret.set(Neon::sys::GpuLaunchInfo::domainGridMode,
                    dims,
                    blockSize,
                    shareMem);
            return ret;
        }
        case Neon::DataView::BOUNDARY: {
            auto dims = getDevSet().newDataSet<index_3d>();
            // Only works z partitions.
            assert(m_data->halo.x == 0 && m_data->halo.y == 0);

            for (int32_t i = 0; i < dims.size(); ++i) {
                dims[i] = m_data->partitionDims[i];
                dims[i].z = m_zBoundaryRadius * 2;
            }

            ret.set(Neon::sys::GpuLaunchInfo::domainGridMode,
                    dims,
                    blockSize,
                    shareMem);
            return ret;
        }
        case Neon::DataView::INTERNAL: {
            auto dims = getDevSet().newDataSet<index_3d>();
            // Only works z partitions.
            assert(m_data->halo.x == 0 && m_data->halo.y == 0);

            for (int32_t i = 0; i < dims.size(); ++i) {
                dims[i] = m_data->partitionDims[i];
                dims[i].z = dims[i].z - m_zBoundaryRadius * 2;
                if (dims[i].z <= 0 && dims.size() > 1) {
                    NeonException exp("dGrid");
                    exp << "The grid size is too small to support the data view model correctly";
                    NEON_THROW(exp);
                }
            }

            ret.set(Neon::sys::GpuLaunchInfo::domainGridMode,
                    dims,
                    blockSize,
                    shareMem);
            return ret;
        }
        default: {
            NeonException exc("dFieldDev");
            NEON_THROW(exc);
        }
    }
}

auto dGrid::getPartitionIndexSpace(Neon::DeviceType devE,
                                   SetIdx           setIdx,
                                   Neon::DataView   dataView)
    -> const PartitionIndexSpace&
{
    return m_data->partitionIndexSpaceVec.at(static_cast<int>(dataView)).local(devE, setIdx, dataView);
}

auto dGrid::newGpuLaunchParameters() const -> Neon::set::LaunchParameters
{

    return getBackend().devSet().newLaunchParameters();
}

auto dGrid::convertToNgh(const std::vector<Neon::index_3d>& stencilOffsets) -> std::vector<ngh_idx>
{
    std::vector<ngh_idx> res;
    for (const auto& offset : stencilOffsets) {
        res.push_back(offset.template newType<int8_t>());
    }
    return res;
}

auto dGrid::convertToNgh(const Neon::index_3d stencilOffsets) -> ngh_idx
{
    return stencilOffsets.template newType<int8_t>();
}

auto dGrid::getKernelConfig(int            streamIdx,
                            Neon::DataView dataView)
    -> Neon::set::KernelConfig
{
    Neon::domain::KernelConfig kernelConfig(streamIdx, dataView);
    if (kernelConfig.runtime() != Neon::Runtime::system) {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

    Neon::set::LaunchParameters launchInfoSet = getLaunchParameters(dataView,
                                                                    getDefaultBlock(), 0);

    kernelConfig.expertSetLaunchParameters(launchInfoSet);
    kernelConfig.expertSetBackend(getBackend());

    return kernelConfig;
}

auto dGrid::setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig) const
    -> void
{
    if (gridKernelConfig.runtime() != Neon::Runtime::system) {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

    // LaunchParameters is generated for the standard view
    Neon::set::LaunchParameters launchInfoSet(int(m_data->partitionDims.size()));
    launchInfoSet.set(Neon::sys::GpuLaunchInfo::domainGridMode,
                      m_data->partitionDims, getDefaultBlock(), 0);


    gridKernelConfig.expertSetLaunchParameters(launchInfoSet);
    gridKernelConfig.expertSetBackend(getBackend());
}

auto dGrid::isInsideDomain(const index_3d& idx, [[maybe_unused]] int level) const -> bool
{
    bool isPositive = idx >= 0;
    bool isLover = idx < this->getDimension();
    return isLover && isPositive;
}

auto dGrid::getProperties(const index_3d& idx, [[maybe_unused]] int level) const -> GridBaseTemplate::CellProperties
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
            zCounter += m_data->partitionDims[i].z;
            if (idx.z < zCounter) {
                setIdx = i;
            }
            if ((zCounterPrevious + m_data->halo.z >= idx.z) &&
                (zCounter - m_data->halo.z < idx.z)) {
                dataView = Neon::DataView::INTERNAL;
            }
            zCounterPrevious = zCounter;
        }
        cellProperties.init(setIdx, dataView);
    }
    return cellProperties;
}
}  // namespace Neon::domain::internal::dGrid