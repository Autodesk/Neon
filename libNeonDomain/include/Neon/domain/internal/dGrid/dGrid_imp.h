#pragma once


#include "dGrid.h"

namespace Neon::domain::internal::dGrid {

template <typename ActiveCellLambda>
dGrid::dGrid(const Neon::Backend&                    backend,
             const Neon::int32_3d&                   dimension,
             [[maybe_unused]] const ActiveCellLambda activeCellLambda,
             const Neon::domain::Stencil&            stencil,
             const Vec_3d<double>&                   spacingData,
             const Vec_3d<double>&                   origin)

{

    {
        auto nElementsPerPartition = backend.devSet().template newDataSet<size_t>(0);
        // We do an initialization with nElementsPerPartition to zero,
        // then we reset to the computed number.
        dGrid::GridBase::init("dGrid",
                              backend,
                              dimension,
                              stencil,
                              nElementsPerPartition,
                              Neon::index_3d(256, 1, 1),
                              spacingData,
                              origin);
    }
    m_data = std::make_shared<data_t>();

    m_data->reduceEngine = Neon::sys::patterns::Engine::cuBlas;

    const int32_t num_devices = getBackend().devSet().setCardinality();
    if (num_devices == 1) {
        // Single device
        m_data->partitionDims = Neon::set::DataSet<index_3d>(1, getDimension());
    } else if (getDimension().z < num_devices) {
        NeonException exc("dGrid_t");
        exc << "The grid size in the z-direction (" << getDimension().z << ") is less the number of devices (" << num_devices
            << "). It is ambiguous how to distribute the gird";
        NEON_THROW(exc);
    } else {
        // we only partition along the z-direction. Each partition has uniform_z
        // along the z-direction. The rest is distribute to make the partitions
        // as equal as possible
        int32_t uniform_z = getDimension().z / num_devices;
        int32_t reminder = getDimension().z % num_devices;
        m_data->partitionDims = Neon::set::DataSet<index_3d>(num_devices, getDimension());

        for (int32_t i = 0; i < num_devices; ++i) {
            m_data->partitionDims[i].x = getDimension().x;
            m_data->partitionDims[i].y = getDimension().y;
            if (i < reminder) {
                m_data->partitionDims[i].z = uniform_z + 1;
            } else {
                m_data->partitionDims[i].z = uniform_z;
            }
        }
    }


    // we partition along z so we only need halo along z
    m_data->halo.x = m_data->halo.y = m_data->halo.z = 0;
    for (const auto& ngh : stencil.neighbours()) {
        m_data->halo.z = std::max(m_data->halo.z, std::abs(ngh.z));
    }

    const index_3d defaultBlockSize(256, 1, 1);

    for (const auto& dw : DataViewUtil::validOptions()) {
        getDefaultLaunchParameters(dw) = getLaunchParameters(dw, defaultBlockSize, 0);
    }
    m_data->partitionIndexSpaceVec = std::vector<Neon::set::DataSet<PartitionIndexSpace>>(3);

    for (auto& i : {Neon::DataView::STANDARD,
                    Neon::DataView::INTERNAL,
                    Neon::DataView::BOUNDARY}) {
        if (static_cast<int>(i) > 2) {
            NeonException exp("dGrid");
            exp << "Inconsistent enumeration for DataView_t";
            NEON_THROW(exp);
        }
        m_data->partitionIndexSpaceVec[static_cast<int>(i)] = getDevSet().newDataSet<PartitionIndexSpace>();
        int setCardinality = getDevSet().setCardinality();
        for (int gpuIdx = 0; gpuIdx < setCardinality; gpuIdx++) {


            m_data->partitionIndexSpaceVec[static_cast<int>(i)][gpuIdx].m_dataView = i;
            m_data->partitionIndexSpaceVec[static_cast<int>(i)][gpuIdx].m_zHaloRadius = setCardinality == 1 ? 0 : m_data->halo.z;
            m_data->partitionIndexSpaceVec[static_cast<int>(i)][gpuIdx].m_zBoundaryRadius = m_data->halo.z;
            m_data->partitionIndexSpaceVec[static_cast<int>(i)][gpuIdx].m_dim = m_data->partitionDims[gpuIdx];
        }
    }

    Neon::set::DataSet<size_t> nElementsPerPartition = backend.devSet().template newDataSet<size_t>([this](Neon::SetIdx idx, size_t& size) {
        size = m_data->partitionDims[idx.idx()].template rMulTyped<size_t>();
    });

    dGrid::GridBase::init("dGrid",
                          backend,
                          dimension,
                          stencil,
                          nElementsPerPartition,
                          defaultBlockSize,
                          spacingData,
                          origin);
}


template <typename T, int C>
auto dGrid::newField(const std::string   fieldUserName,
                     int                 cardinality,
                     [[maybe_unused]] T  inactiveValue,
                     Neon::DataUse       dataUse,
                     Neon::MemoryOptions memoryOptions) const
    -> dField<T, C>
{
    memoryOptions = getDevSet().sanitizeMemoryOption(memoryOptions);

    const auto haloStatus = Neon::domain::haloStatus_et::ON;

    if (C != 0 && cardinality != C) {
        NeonException exception("dGrid::newField Dynamic and static cardinality do not match.");
        NEON_THROW(exception);
    }

    dField<T, C> field(fieldUserName,
                       dataUse,
                       memoryOptions,
                       *this,
                       m_data->partitionDims,
                       m_data->halo.z,
                       haloStatus,
                       cardinality);

    return field;
}

template <typename T, int C>
auto dGrid::newFieldDev(Neon::sys::memConf_t           memConf,
                        int                            cardinality,
                        [[maybe_unused]] T             inactiveValue,
                        Neon::domain::haloStatus_et::e haloStatus)
    -> dFieldDev<T, C>
{
    haloStatus = Neon::domain::haloStatus_et::ON;
    if (memConf.padding() == Neon::memLayout_et::padding_e::ON) {
        NEON_DEV_UNDER_CONSTRUCTION("TODO: waiting for refactoring of memory options");
    }
    Neon::index_3d halo = (haloStatus == Neon::domain::haloStatus_et::ON) ? m_data->halo : Neon::index_3d(0, 0, 0);
    return dFieldDev<T, C>(*this, getDimension(),
                           halo, memConf, cardinality);
}

template <typename LoadingLambda>
auto dGrid::getContainer(const std::string& name,
                         LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
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
auto dGrid::getContainer(const std::string& name,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
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
auto dGrid::newPatternScalar() const -> Neon::template PatternScalar<T>
{
    auto pattern = Neon::PatternScalar<T>(getBackend(), m_data->reduceEngine);

    if (m_data->reduceEngine == Neon::sys::patterns::Engine::CUB) {
        for (auto& dataview : {Neon::DataView::STANDARD,
                               Neon::DataView::INTERNAL,
                               Neon::DataView::BOUNDARY}) {
            auto launchParam = getLaunchParameters(dataview, getDefaultBlock(), 0);
            for (SetIdx id = 0; id < launchParam.cardinality(); id++) {
                uint32_t numBlocks = launchParam[id].cudaGrid().x *
                                     launchParam[id].cudaGrid().y *
                                     launchParam[id].cudaGrid().z;
                pattern.getBlasSet(dataview).getBlas(id.idx()).setNumBlocks(numBlocks);
            }
        }
    }
    return pattern;
}

template <typename T>
auto dGrid::dot(const std::string&               name,
                dField<T>&                       input1,
                dField<T>&                       input2,
                Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container
{
    if (m_data->reduceEngine == Neon::sys::patterns::Engine::cuBlas || getBackend().devType() == Neon::DeviceType::CPU) {
        return Neon::set::Container::factoryOldManaged(
            name,
            Neon::set::internal::ContainerAPI::DataViewSupport::on,
            *this, [&](Neon::set::Loader& loader) {
                loader.load(input1);
                if (input1.getUid() != input2.getUid()) {
                    loader.load(input2);
                }
                loader.load(scalar);
                return [&](int streamIdx, Neon::DataView dataView) mutable -> void {
                    if (dataView != Neon::DataView::STANDARD && getBackend().devSet().setCardinality() == 1) {
                        NeonException exc("dGrid_t");
                        exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                        NEON_THROW(exc);
                    }
                    scalar.setStream(streamIdx, dataView);
                    scalar(dataView) = input1.dot(scalar.getBlasSet(dataView),
                                                  input2, scalar.getTempMemory(dataView), dataView);
                    if (dataView == Neon::DataView::BOUNDARY) {
                        scalar(Neon::DataView::STANDARD) =
                            scalar(Neon::DataView::BOUNDARY) + scalar(Neon::DataView::INTERNAL);
                    }
                };
            });
    } else if (m_data->reduceEngine == Neon::sys::patterns::Engine::CUB) {

        return Neon::set::Container::factoryOldManaged(
            name,
            Neon::set::internal::ContainerAPI::DataViewSupport::on,
            *this, [&](Neon::set::Loader& loader) {
                loader.load(input1);
                if (input1.getUid() != input2.getUid()) {
                    loader.load(input2);
                }
                loader.load(scalar);

                return [&](int streamIdx, Neon::DataView dataView) mutable -> void {
                    if (dataView != Neon::DataView::STANDARD && getBackend().devSet().setCardinality() == 1) {
                        NeonException exc("dGrid_t");
                        exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                        NEON_THROW(exc);
                    }
                    scalar.setStream(streamIdx, dataView);

                    // calc dot product and store results on device
                    input1.dotCUB(scalar.getBlasSet(dataView),
                                  input2,
                                  scalar.getTempMemory(dataView, Neon::DeviceType::CUDA),
                                  dataView);

                    // move to results to host
                    scalar.getTempMemory(dataView,
                                         Neon::DeviceType::CPU)
                        .template updateFrom<Neon::run_et::et::async>(
                            scalar.getBlasSet(dataView).getStream(),
                            scalar.getTempMemory(dataView, Neon::DeviceType::CUDA));

                    // sync
                    scalar.getBlasSet(dataView).getStream().sync();

                    // read the results
                    scalar(dataView) = 0;
                    int nGpus = getBackend().devSet().setCardinality();
                    for (int idx = 0; idx < nGpus; idx++) {
                        scalar(dataView) += scalar.getTempMemory(dataView, Neon::DeviceType::CPU).elRef(idx, 0, 0);
                    }

                    if (dataView == Neon::DataView::BOUNDARY) {
                        scalar(Neon::DataView::STANDARD) =
                            scalar(Neon::DataView::BOUNDARY) + scalar(Neon::DataView::INTERNAL);
                    }
                };
            });


    } else {
        NeonException exc("dGrid_t");
        exc << "Unsupported reduction engine";
        NEON_THROW(exc);
    }
}

template <typename T>
auto dGrid::norm2(const std::string&               name,
                  dField<T>&                       input,
                  Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container
{
    if (m_data->reduceEngine == Neon::sys::patterns::Engine::cuBlas || getBackend().devType() == Neon::DeviceType::CPU) {
        return Neon::set::Container::factoryOldManaged(
            name,
            Neon::set::internal::ContainerAPI::DataViewSupport::on,
            *this, [&](Neon::set::Loader& loader) {
                loader.load(input);

                return [&](int streamIdx, Neon::DataView dataView) mutable -> void {
                    if (dataView != Neon::DataView::STANDARD && getBackend().devSet().setCardinality() == 1) {
                        NeonException exc("dGrid_t");
                        exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                        NEON_THROW(exc);
                    }
                    scalar.setStream(streamIdx, dataView);
                    scalar(dataView) = input.norm2(scalar.getBlasSet(dataView),
                                                   scalar.getTempMemory(dataView), dataView);
                    if (dataView == Neon::DataView::BOUNDARY) {
                        scalar(Neon::DataView::STANDARD) =
                            std::sqrt(scalar(Neon::DataView::BOUNDARY) * scalar(Neon::DataView::BOUNDARY) +
                                      scalar(Neon::DataView::INTERNAL) * scalar(Neon::DataView::INTERNAL));
                    }
                };
            });
    } else if (m_data->reduceEngine == Neon::sys::patterns::Engine::CUB) {
        return Neon::set::Container::factoryOldManaged(
            name,
            Neon::set::internal::ContainerAPI::DataViewSupport::on,
            *this, [&](Neon::set::Loader& loader) {
                loader.load(input);

                return [&](int streamIdx, Neon::DataView dataView) mutable -> void {
                    if (dataView != Neon::DataView::STANDARD && getBackend().devSet().setCardinality() == 1) {
                        NeonException exc("dGrid_t");
                        exc << "Reduction operation can only run on standard data view when the number of partitions/GPUs is 1";
                        NEON_THROW(exc);
                    }
                    scalar.setStream(streamIdx, dataView);

                    // calc dot product and store results on device
                    input.norm2CUB(scalar.getBlasSet(dataView),
                                   scalar.getTempMemory(dataView, Neon::DeviceType::CUDA),
                                   dataView);

                    // move to results to host
                    scalar.getTempMemory(dataView,
                                         Neon::DeviceType::CPU)
                        .template updateFrom<Neon::run_et::et::async>(
                            scalar.getBlasSet(dataView).getStream(),
                            scalar.getTempMemory(dataView, Neon::DeviceType::CUDA));

                    // sync
                    scalar.getBlasSet(dataView).getStream().sync();

                    // read the results
                    scalar(dataView) = 0;
                    int nGpus = getBackend().devSet().setCardinality();
                    for (int idx = 0; idx < nGpus; idx++) {
                        scalar(dataView) += scalar.getTempMemory(dataView, Neon::DeviceType::CPU).elRef(idx, 0, 0);
                    }
                    scalar(dataView) = std::sqrt(scalar());

                    if (dataView == Neon::DataView::BOUNDARY) {
                        scalar(Neon::DataView::STANDARD) =
                            std::sqrt(scalar(Neon::DataView::BOUNDARY) * scalar(Neon::DataView::BOUNDARY) +
                                      scalar(Neon::DataView::INTERNAL) * scalar(Neon::DataView::INTERNAL));
                    }
                };
            });

    } else {
        NeonException exc("dGrid_t");
        exc << "Unsupported reduction engine";
        NEON_THROW(exc);
    }
}

}  // namespace Neon::domain::internal::dGrid