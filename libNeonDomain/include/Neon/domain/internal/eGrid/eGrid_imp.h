#pragma once

#include "eGridStorage.h"
namespace Neon::domain::internal::eGrid {

template <typename ActiveCellLambda>
eGrid::eGrid(const Neon::Backend&         backend,
             const Neon::index_3d&        cellDomain,
             const ActiveCellLambda&      activeCellLambda,
             const Neon::domain::Stencil& stencil,
             const Vec_3d<double>&        spacingData,
             const Vec_3d<double>&        origin,
             bool                         includeInveseMappingField)
{
    auto                 nElementsPerPartition = backend.devSet().template newDataSet<size_t>(0);
    const Neon::index_3d defaultsBlockDim(512, 1, 1);

    // We do an initialization with nElementsPerPartition to zero,
    // then we reset to the computed number.
    eGrid::GridBaseTemplate::init("eGrid",
                                  backend,
                                  cellDomain,
                                  stencil,
                                  nElementsPerPartition,
                                  defaultsBlockDim,
                                  spacingData,
                                  origin);

    namespace eInternals = Neon::domain::internal::eGrid::internals;
    m_ds = std::make_shared<eStorage>();

    m_ds->inverseMappingEnabled = includeInveseMappingField;

    m_ds->builder = eInternals::dsBuilder_t(getDevSet(),
                                            cellDomain,
                                            activeCellLambda,
                                            getDevSet().setCardinality(),
                                            stencil);

    auto initCellClassification = [this, &nElementsPerPartition]() -> void {
        m_ds->getCount(Neon::DataView::STANDARD) = getDevSet().newDataSet<count_t>();
        m_ds->getCount(Neon::DataView::INTERNAL) = getDevSet().newDataSet<count_t>();
        m_ds->getCount(Neon::DataView::BOUNDARY) = getDevSet().newDataSet<count_t>();

        for (int i = 0; i < getDevSet().setCardinality(); i++) {
            m_ds->getCountPerDevice(Neon::DataView::STANDARD, i) = m_ds->builder.frame()->localIndexingInfo(i).nElements(false);
            m_ds->getCountPerDevice(Neon::DataView::INTERNAL, i) = m_ds->builder.frame()->localIndexingInfo(i).internalCount();
            m_ds->getCountPerDevice(Neon::DataView::BOUNDARY, i) = m_ds->builder.frame()->localIndexingInfo(i).bdrCount();

            nElementsPerPartition[i] = m_ds->getCountPerDevice(Neon::DataView::STANDARD, i);
        }
    };

    auto initDefaultLaunchParameters = [this]() -> void {
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
        }
    };

    auto inverseMappingField = [this]() -> void {
        Neon::Backend bk(getDevSet(), Neon::Runtime::openmp);
        if (getDevSet().type() == Neon::DeviceType::CUDA) {
            bk = Neon::Backend(getDevSet(), Neon::Runtime::stream);
        }

        if (m_ds->inverseMappingEnabled) {
            //            const int cardinality = 3;
            //            m_ds->inverseMappingFieldMirror = this->newField<index_t, cardinality>({bk,
            //                                                                                    Neon::DataUse::IO_COMPUTE},
            //                                                                                   haloStatus_et::OFF,
            //                                                                                   cardinality, -1);
            //
            //            m_ds->inverseMappingFieldMirror.cpu().forEachActive([](const index_3d& idx3d, const int card, index_t& val) {
            //                val = idx3d.v[card];
            //            });
            //            m_ds->inverseMappingFieldMirror.updateCompute(getDevSet().defaultStreamSet());
            NEON_DEV_UNDER_CONSTRUCTION("");
        }
    };

    auto initPartitionIndexSpace = [this]() {
        for (auto& dw : Neon::DataViewUtil::validOptions()) {
            m_ds->getPartitionIndexSpace(dw) = this->getDevSet().newDataSet<ePartitionIndexSpace>();

            for (int gpuIdx = 0; gpuIdx < this->getDevSet().setCardinality(); gpuIdx++) {
                const auto& indexingInfo = m_ds->builder.frame()->localIndexingInfo(gpuIdx);

                std::array<Cell::Offset, ComDirection_e::COM_NUM> bdrOff = {indexingInfo.bdrOff(ComDirection_e::COM_DW),
                                                                            indexingInfo.bdrOff(ComDirection_e::COM_UP)};
                std::array<Cell::Offset, ComDirection_e::COM_NUM> ghostOff = {indexingInfo.ghostOff(ComDirection_e::COM_DW),
                                                                              indexingInfo.ghostOff(ComDirection_e::COM_UP)};

                m_ds->getPartitionIndexSpace(dw)[gpuIdx].hGetBoundaryOffset()[ComDirection_e::COM_UP] = bdrOff[ComDirection_e::COM_UP];
                m_ds->getPartitionIndexSpace(dw)[gpuIdx].hGetBoundaryOffset()[ComDirection_e::COM_DW] = bdrOff[ComDirection_e::COM_DW];

                m_ds->getPartitionIndexSpace(dw)[gpuIdx].hgetGhostOffset()[ComDirection_e::COM_UP] = ghostOff[ComDirection_e::COM_UP];
                m_ds->getPartitionIndexSpace(dw)[gpuIdx].hgetGhostOffset()[ComDirection_e::COM_DW] = ghostOff[ComDirection_e::COM_DW];

                m_ds->getPartitionIndexSpace(dw)[gpuIdx].hGetDataView() = dw;
            }
        }
    };

    initCellClassification();
    initDefaultLaunchParameters();
    inverseMappingField();
    initPartitionIndexSpace();

    eGrid::GridBaseTemplate::init("eGrid",
                                  backend,
                                  cellDomain,
                                  stencil,
                                  nElementsPerPartition,
                                  defaultsBlockDim,
                                  spacingData,
                                  origin);
}


template <typename T, int C>
auto eGrid::newField(const std::string&  fieldUserName,
                     int                 cardinality,
                     T                   inactiveValue,
                     Neon::DataUse       dataUse,
                     Neon::MemoryOptions memoryOptions) const
    -> Field<T, C>
{
    memoryOptions = getDevSet().sanitizeMemoryOption(memoryOptions);

    if (C != 0 && cardinality != C) {
        NeonException exception("Dynamic and static setCardinality do not match.");
        NEON_THROW(exception);
    }

    auto helpNewFieldDev = [this](Neon::sys::memConf_t           memConf,
                                  int                            cardinality,
                                  T                              inactiveValue,
                                  Neon::domain::haloStatus_et::e haloStatus) {
        eFieldDevice_t<T, C> field(*this,
                                   memConf,
                                   getDevSet(),
                                   cardinality,
                                   inactiveValue,
                                   m_ds->builder.frame(),
                                   haloStatus);
        return field;
    };
    switch (getBackend().devType()) {
        case Neon::DeviceType::OMP:
        case Neon::DeviceType::CPU: {
            // For CPU we have the same configuration for COMPUTE, IO_COMPUTE, IO_POST

            Neon::sys::MemAlignment alignment;

            Neon::DeviceType devC = Neon::DeviceType::NONE;
            Neon::DeviceType devIO = Neon::DeviceType::CPU;

            // Neon::Allocator allocTypeC = memoryOptions.getComputeAllocator();

            Neon::Allocator allocTypeC = Neon::Allocator::NULL_MEM;
            Neon::Allocator allocTypeIO = Neon::Allocator::MALLOC;

            Neon::sys::memConf_t confC(devC,
                                       allocTypeC,
                                       Neon::memLayout_et::convert(memoryOptions.getOrder()),
                                       alignment,
                                       Neon::memLayout_et::padding_e::OFF);

            Neon::sys::memConf_t confIO(devIO,
                                        allocTypeIO,
                                        Neon::memLayout_et::convert(memoryOptions.getOrder()),
                                        alignment,
                                        Neon::memLayout_et::padding_e::OFF);


            const Neon::set::DataConfig dataConfig(getBackend(), dataUse,
                                                   confIO, confC);

            eFieldDevice_t<T, C> gpu = helpNewFieldDev(dataConfig.memConfig(Neon::DeviceType::CUDA),
                                                       cardinality,
                                                       inactiveValue,
                                                       Neon::domain::haloStatus_et::e::ON);

            eFieldDevice_t<T, C> cpu = helpNewFieldDev(dataConfig.memConfig(Neon::DeviceType::CPU),
                                                       cardinality,
                                                       inactiveValue,
                                                       Neon::domain::haloStatus_et::e::ON);


            eField<T, C> mirror(fieldUserName,
                                cardinality,
                                inactiveValue,
                                dataUse,
                                memoryOptions,
                                Neon::domain::haloStatus_et::e::ON,
                                dataConfig, cpu, gpu);
            return mirror;
        }
        case Neon::DeviceType::CUDA: {

            // Compute is the CUDA
            Neon::sys::MemAlignment tmp;
            auto                    exp = tmp.expAlign(Neon::DeviceType::CUDA, 0);
            auto                    alignmentC = Neon::sys::MemAlignment(Neon::sys::memAlignment_et::user, exp);
            auto                    alignmentIO = Neon::sys::MemAlignment(Neon::sys::memAlignment_et::user, exp);

            Neon::Allocator allocTypeC = memoryOptions.getComputeAllocator();
            Neon::Allocator allocTypeIO = dataUse == Neon::DataUse::COMPUTE ? Neon::Allocator::NULL_MEM : memoryOptions.getIOAllocator();

            Neon::DeviceType devC = getBackend().devType();
            Neon::DeviceType devIO = getBackend().devType() == Neon::DeviceType::CUDA ? Neon::DeviceType::CPU : Neon::DeviceType::CUDA;

            Neon::sys::memConf_t confC(devC,
                                       allocTypeC,
                                       Neon::memLayout_et::convert(memoryOptions.getOrder()),
                                       alignmentC,
                                       Neon::memLayout_et::padding_e::OFF);

            Neon::sys::memConf_t confIO(devIO,
                                        allocTypeIO,
                                        Neon::memLayout_et::convert(memoryOptions.getOrder()),
                                        alignmentIO,
                                        Neon::memLayout_et::padding_e::OFF);


            const Neon::set::DataConfig dataConfig(getBackend(), dataUse,
                                                   confIO,
                                                   confC);

            eFieldDevice_t<T, C> gpu = helpNewFieldDev(dataConfig.memConfig(Neon::DeviceType::CUDA),
                                                       cardinality,
                                                       inactiveValue,
                                                       Neon::domain::haloStatus_et::e::ON);

            eFieldDevice_t<T, C> cpu = helpNewFieldDev(dataConfig.memConfig(Neon::DeviceType::CPU),
                                                       cardinality,
                                                       inactiveValue,
                                                       Neon::domain::haloStatus_et::e::ON);


            eField<T, C> mirror(fieldUserName,
                                cardinality,
                                inactiveValue,
                                dataUse,
                                memoryOptions,
                                Neon::domain::haloStatus_et::e::ON,
                                dataConfig, cpu, gpu);
            return mirror;
        }
        default: {
            NEON_DEV_UNDER_CONSTRUCTION("");
        }
    }
}

template <typename LoadingLambda>
auto eGrid::getContainer(const std::string& name,
                         LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
    Neon::domain::KernelConfig kernelConfig(0);

    const Neon::index_3d& defaultBlockSize = getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
                                                                     Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                     *this,
                                                                     lambda,
                                                                     defaultBlockSize,
                                                                     [](const Neon::index_3d&) { return size_t(0); });
    return kContainer;
}

template <typename LoadingLambda>
auto eGrid::getContainer(const std::string& name,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
    Neon::domain::KernelConfig kernelConfig(0);

    const Neon::index_3d& defaultBlockSize = getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
                                                                     Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                     *this,
                                                                     lambda,
                                                                     blockSize,
                                                                     [sharedMem](const Neon::index_3d&) { return sharedMem; });
    return kContainer;
}

template <typename T>
auto eGrid::newPatternScalar() const
    -> Neon::template PatternScalar<T>
{
    return Neon::PatternScalar<T>(getBackend());
}

template <typename T, int C>
auto eGrid::dot(const std::string& /*name*/,
                eField<T, C>& /*input1*/,
                eField<T, C>& /*input2*/,
                Neon::template PatternScalar<T>& /*scalar*/) const
    -> Neon::set::Container
{
    NEON_THROW_UNSUPPORTED_OPERATION("Patterns on eGrid");
}

template <typename T, int C>
auto eGrid::norm2(const std::string& /*name*/,
                  eField<T, C>& /*input*/,
                  Neon::template PatternScalar<T>& /*scalar*/) const
    -> Neon::set::Container
{
    NEON_THROW_UNSUPPORTED_OPERATION("Patterns on eGrid");
}
};  // namespace Neon::domain::internal::eGrid
