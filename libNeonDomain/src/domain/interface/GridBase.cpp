#include "Neon/domain/interface/GridBase.h"
#include "Neon/domain/tools/IOGridVTK.h"

namespace Neon::domain::interface {

auto GridBase::init(const std::string&                gridImplementationName,
                    const Neon::Backend&              backend,
                    const Neon::index_3d&             dimension,
                    const Neon::domain::Stencil&      stencil,
                    const Neon::set::DataSet<size_t>& nPartitionElements,
                    const Neon::index_3d&             defaultBlockSize,
                    const Vec_3d<double>&             spacingData,
                    const Vec_3d<double>&             origin) -> void
{
    mStorage->backend = backend;
    mStorage->dimension = dimension;
    mStorage->stencil = stencil;
    mStorage->nPartitionElements = nPartitionElements.clone();
    mStorage->spacing = spacingData;
    mStorage->origin = origin;
    mStorage->gridImplementationName = gridImplementationName;

    for (const auto& dw : DataViewUtil::validOptions()) {
        mStorage->defaults.launchParameters[DataViewUtil::toInt(dw)] = backend.devSet().newLaunchParameters();
    }
    mStorage->defaults.blockDim = defaultBlockSize;
}

GridBase::GridBase()
    : mStorage(std::make_shared<GridBase::Storage>())
{
}

GridBase::GridBase(const std::string&                gridImplementationName,
                   const Neon::Backend&              backend,
                   const Neon::index_3d&             dimension,
                   const Neon::domain::Stencil&      stencil,
                   const Neon::set::DataSet<size_t>& nPartitionElements,
                   const Neon::index_3d&             defaultBlockSize,
                   const Vec_3d<double>&             spacingData,
                   const Vec_3d<double>&             origin)
    : mStorage(std::make_shared<GridBase::Storage>())
{
    init(gridImplementationName,
         backend,
         dimension,
         stencil,
         nPartitionElements,
         defaultBlockSize,
         spacingData,
         origin);
}

auto GridBase::getDimension() const -> const Neon::index_3d&
{
    return mStorage->dimension;
}

auto GridBase::getStencil() const -> const Neon::domain::Stencil&
{
    return mStorage->stencil;
}

auto GridBase::getSpacing() const -> const Vec_3d<double>&
{
    return mStorage->spacing;
}

auto GridBase::getOrigin() const -> const Vec_3d<double>&
{
    return mStorage->origin;
}

auto GridBase::getNumAllCells() const
    -> size_t
{
    return mStorage->dimension.rMulTyped<size_t>();
}

auto GridBase::getNumActiveCells() const
    -> size_t
{
    size_t count = 0;
    for (int idx = 0; idx < mStorage->backend.devSet().setCardinality(); idx++) {
        count += mStorage->nPartitionElements[idx];
    }
    return count;
}

auto GridBase::getBackend() const
    -> const Backend&
{
    return mStorage->backend;
}

auto GridBase::getBackend()
    -> Backend&
{
    return mStorage->backend;
}

auto GridBase::getDevSet() const
    -> const Neon::set::DevSet&
{
    return mStorage->backend.devSet();
}

auto GridBase::getDefaultBlock() const
    -> const Neon::index_3d&
{
    return mStorage->defaults.blockDim;
}

auto GridBase::setDefaultBlock(const Neon::index_3d& blockDim)
    -> void
{
    mStorage->defaults.blockDim = blockDim;
}

auto GridBase::flattenedLengthSet() const
    -> const Neon::set::DataSet<size_t>&
{
    return getNumActiveCellsPerPartition();
}

auto GridBase::getNumActiveCellsPerPartition() const
    -> const Neon::set::DataSet<size_t>&
{
    return mStorage->nPartitionElements;
}

auto GridBase::getDefaultLaunchParameters(Neon::DataView dataView)
    -> Neon::set::LaunchParameters&
{
    return mStorage->defaults.launchParameters[Neon::DataViewUtil::toInt(dataView)];
}

auto GridBase::getDefaultLaunchParameters(Neon::DataView dataView) const
    -> const Neon::set::LaunchParameters&
{
    return mStorage->defaults.launchParameters[Neon::DataViewUtil::toInt(dataView)];
}

auto GridBase::getImplementationName() const
    -> const std::string&
{
    return mStorage->gridImplementationName;
}

auto GridBase::toString() const -> std::string
{
    std::stringstream s;
    s << "[Domain Grid]:{" << this->getImplementationName() << "}"
      << ", [Background Grid]:{" << this->getDimension() << "}"
      << ", [Active Cells]:{" << this->getNumActiveCells() << "}"
      << ", [Cell Distribution]:{" << [&] {
             std::stringstream tmp;
             tmp << "(";
             const int nPartitions = int(this->getNumActiveCellsPerPartition().size());
             for (int i = 0; i < nPartitions; i++) {
                 tmp << this->getNumActiveCellsPerPartition()[i];
                 if (i < nPartitions - 1) {
                     tmp << ",";
                 }
             }
             tmp << ")";
             return tmp.str();
         }()
      << "}"
      << ", [Backend]:{" << getBackend().toString() << "}";

    return s.str();
}

auto GridBase::getGridUID() const -> size_t
{
    return size_t(mStorage.get());
}

auto GridBase::toReport(Neon::Report& report,
                        bool          includeBackendInfo) const -> void
{
    auto subdoc = report.getSubdoc();

    report.addMember("ImplementationName", getImplementationName(), &subdoc);
    report.addMember("ActiveCells", getNumActiveCells(), &subdoc);
    report.addMember("GridUID", getGridUID(), &subdoc);

    report.addMember(
        "Dimension",
        [&] {
            std::stringstream list;
            list << "[";
            list << getDimension().x << " "
                 << getDimension().y << " "
                 << getDimension().z << "]";
            return list.str();
        }(),
        &subdoc);

    report.addMember(
        "Stencil",
        [&] {
            std::stringstream list;
            list << "[";
            bool isFirst = true;
            auto stencil = getStencil();
            auto stencilPoints = stencil.points();
            for (auto& point : stencilPoints) {
                if (!isFirst) {
                    list << " ";
                }
                list << "(";
                list << point.x << " "
                     << point.y << " "
                     << point.z << ")";
                isFirst = false;
            }
            return list.str();
        }(),
        &subdoc);

    report.addMember(
        "ActiveCellsPerPartition",
        [&] {
            std::stringstream list;
            list << "[";
            int i = 0;
            for (auto& nCells : getNumActiveCellsPerPartition().vec()) {
                if (i != 0) {
                    list << " ";
                }
                list << nCells;
                i = 1;
            }
            list << "]";
            return list.str();
        }(),
        &subdoc);

    if (includeBackendInfo)
        getBackend().toReport(report, &subdoc);

    report.addSubdoc("Grid", subdoc);
}

}  // namespace Neon::domain::interface