#include "Neon/domain/internal/eGrid/eGrid.h"

namespace Neon::domain::internal::eGrid {

eGrid::eGrid()
{
    m_ds = std::make_shared<eStorage>();
}

auto eGrid::newLaunchParameters() const
    -> Neon::set::LaunchParameters
{
    return getDevSet().newLaunchParameters();
}

auto eGrid::getPartitionIndexSpace(Neon::DeviceType devE,
                                   SetIdx           setIdx,
                                   Neon::DataView   dataView) -> const PartitionIndexSpace&
{
    return m_ds->getPartitionIndexSpace(dataView).local(devE, setIdx, dataView);
}

const internals::dsFrame_t* eGrid::frame() const
{
    return helpGetMds().builder.frame().get();
}

auto eGrid::convertToNgh(const std::vector<Neon::index_3d>& stencilOffsets)
    -> std::vector<ngh_idx>
{
    std::vector<ngh_idx> res;
    for (const auto& offset : stencilOffsets) {
        res.push_back(convertToNgh(offset));
    }
    return res;
}

auto eGrid::convertToNgh(const Neon::index_3d& stencilOffset)
    -> Self::ngh_idx
{
    Self::ngh_idx id;
    id = static_cast<Self::ngh_idx>(getStencil().find(stencilOffset));
    return id;
}

auto eGrid::helpGetMds() -> eStorage&
{
    return *(m_ds.get());
}

auto eGrid::helpGetMds() const -> const eStorage&
{
    return *(m_ds.get());
}

auto eGrid::getKernelConfig(int            streamIdx,
                            Neon::DataView dataView) -> Neon::set::KernelConfig
{
    Neon::domain::KernelConfig kernelConfig(streamIdx, dataView);
    if (kernelConfig.runtime() != Neon::Runtime::system) {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

    // LAUNCH INFO
    Neon::set::LaunchParameters launchInfos;
    {  // computing launch info
        if (kernelConfig.blockConfig().blockMode() == Neon::set::BlockConfig::system) {
            launchInfos = getDefaultLaunchParameters(dataView);
        } else {
            auto indexing = kernelConfig.dataView();
            auto blockSize = kernelConfig.blockConfig().blockSize();
            auto sharedMem = kernelConfig.blockConfig().sharedMemory(blockSize);
            launchInfos = getLaunchParameters(indexing, blockSize, sharedMem);
        }
    }

    kernelConfig.expertSetLaunchParameters(launchInfos);
    kernelConfig.expertSetBackend(getBackend());
    kernelConfig.expertSetStream(streamIdx);

    return kernelConfig;
}

auto eGrid::setKernelConfig(Neon::set::KernelConfig& kernelConfig) const -> void
{
    // Using default value;
    Neon::domain::KernelConfig gridKernelConfig;

    if (gridKernelConfig.runtime() != Neon::Runtime::system) {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

    // LAUNCH INFO
    Neon::set::LaunchParameters launchInfos;
    {  // computing launch info
        if (gridKernelConfig.blockConfig().blockMode() == Neon::set::BlockConfig::system) {
            launchInfos = getDefaultLaunchParameters(kernelConfig.dataView());
        } else {
            auto indexing = kernelConfig.dataView();
            auto blockSize = gridKernelConfig.blockConfig().blockSize();
            auto sharedMem = gridKernelConfig.blockConfig().sharedMemory(blockSize);
            launchInfos = getLaunchParameters(indexing, blockSize, sharedMem);
        }
    }

    kernelConfig.expertSetLaunchParameters(launchInfos);
    kernelConfig.expertSetBackend(getBackend());
    return;
}

auto eGrid::setKernelConfig(Neon::domain::KernelConfig& gridKernelConfig) const -> void
{

    if (gridKernelConfig.runtime() != Neon::Runtime::system) {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

    // LAUNCH INFO
    {  // computing launch info
        if (gridKernelConfig.blockConfig().blockMode() == Neon::set::BlockConfig::system) {
            gridKernelConfig.expertSetLaunchParameters([&](Neon::set::LaunchParameters& launchInfo) {
                launchInfo = getDefaultLaunchParameters(gridKernelConfig.dataView());
            });
        } else {
            gridKernelConfig.expertSetLaunchParameters([&](Neon::set::LaunchParameters& launchInfo) {
                auto indexingPolicy = gridKernelConfig.dataView();
                auto blockDim = gridKernelConfig.blockConfig().blockSize();
                auto sharedMem = gridKernelConfig.blockConfig().sharedMemory(blockDim);

                for (int i = 0; i < getDevSet().setCardinality(); i++) {
                    const auto    gridMode = Neon::sys::GpuLaunchInfo::mode_e::domainGridMode;
                    const count_t nElements = m_ds->getCount(indexingPolicy)[i];
                    launchInfo[i].set(gridMode, nElements, blockDim, sharedMem);
                }
            });
        }
    }
    gridKernelConfig.expertSetBackend(getBackend());

    return;
}

auto eGrid::helpSetDefaultBlock()
    -> void
{
    if (getDefaultBlock().y != 1 || getDefaultBlock().z != 1) {
        NeonException exc("eGrid");
        exc << "CUDA block size should be 1D\n";
        NEON_THROW(exc);
    }

    for (int i = 0; i < getDevSet().setCardinality(); i++) {
        for (auto indexing : DataViewUtil::validOptions()) {

            auto gridMode = Neon::sys::GpuLaunchInfo::mode_e::domainGridMode;
            auto gridDim = m_ds->getCount(indexing)[i];
            getDefaultLaunchParameters(indexing)[i].set(gridMode, gridDim, getDefaultBlock(), 0);
        }
    };
}

auto eGrid::isInsideDomain(const index_3d& idx, [[maybe_unused]] int level) const -> bool
{
    auto gridBox = this->getDimension();
    bool isInsideBox = true;
    isInsideBox = (idx.x < gridBox.x) && isInsideBox;
    isInsideBox = (idx.y < gridBox.y) && isInsideBox;
    isInsideBox = (idx.z < gridBox.z) && isInsideBox;
    isInsideBox = (idx.x >= 0) && isInsideBox;
    isInsideBox = (idx.y >= 0) && isInsideBox;
    isInsideBox = (idx.z >= 0) && isInsideBox;
    if (!isInsideBox) {
        return false;
    }
    const auto& GtoL = frame()->globalToLocal();
    const auto& info = GtoL.elRef(idx);
    return info.isActive();
}

auto eGrid::getProperties(const index_3d& idx, [[maybe_unused]] int level) const -> GridBaseTemplate::CellProperties
{
    GridBaseTemplate::CellProperties cellProperties;
    const auto&                      GtoL = frame()->globalToLocal();
    const auto&                      info = GtoL.elRef(idx);

    cellProperties.setIsInside(info.isActive());
    if (!cellProperties.isInside()) {
        return cellProperties;
    }

    auto            setIdx = info.getPrtIdx();
    auto            dataview = info.getDataView();
    Cell::OuterCell outerCell;
    outerCell.set() = Neon::domain::internal::eGrid::eCell::Location(info.getLocalIdx());
    cellProperties.init(setIdx, dataview, outerCell);

    return cellProperties;
}

auto eGrid::getLaunchParameters(Neon::DataView  dataView,
                                const index_3d& blockDim,
                                const size_t&   shareMem) const -> Neon::set::LaunchParameters
{
    if (blockDim.y != 1 || blockDim.z != 1) {
        NeonException exc("eGrid");
        exc << "CUDA block size should be 1D\n";
        NEON_THROW(exc);
    }

    auto newLaunchParameters = getDevSet().newLaunchParameters();

    for (int i = 0; i < getDevSet().setCardinality(); i++) {

        auto gridMode = Neon::sys::GpuLaunchInfo::mode_e::domainGridMode;
        auto gridDim = m_ds->getCount(dataView)[i];
        newLaunchParameters[i].set(gridMode, gridDim, blockDim, shareMem);
    }
    return newLaunchParameters;
}

}  // namespace Neon::domain::internal::eGrid
