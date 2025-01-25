#pragma once
#include "ncclGrid.h"

namespace Neon::domain::details::ncclGrid {


template <typename ActiveCellLambda>
ncclGrid::ncclGrid(const Neon::Backend&  backend,
                   const Neon::int32_3d& dimension,
                   const ActiveCellLambda& /*activeCellLambda*/,
                   const Neon::domain::Stencil&                 stencil,
                   const Vec_3d<double>&                        spacing,
                   const Vec_3d<double>&                        origin,
                   Neon::domain::tool::spaceCurves::EncoderType encoderType)
{
    mData = std::make_shared<Data>(backend);
    const index_3d defaultBlockSize(256, 1, 1);
    if (encoderType != Neon::domain::tool::spaceCurves::EncoderType::sweep) {
        NeonException exce("ncclGrid");
        exce << "ncclGrid only supports sweep space filling curves";
        NEON_THROW(exce);
    }

    {
        auto nElementsPerPartition = backend.newRankData<size_t>(0);
        // We do an initialization with nElementsPerPartition to zero,
        // then we reset to the computed number.
        ncclGrid::GridBase::init("ncclGrid",
                                 backend,
                                 dimension,
                                 stencil,
                                 nElementsPerPartition,
                                 Neon::index_3d(256, 1, 1),
                                 spacing,
                                 origin,
                                 Neon::domain::tool::spaceCurves::EncoderType::sweep,
                                 {0, 0, 0});
    }

    if (!getBackend().isDistributed()) {
        NeonException exc("ncclGrid_t");
        exc << "The grid can work only with distributed backends";
        NEON_THROW(exc);
    }
    const int32_t worldSize = backend.getNccl().getWorldSize();
    const int32_t myRank = backend.getNccl().getWorldRank();
    const int32_t numDevices = backend.deviceCount();
    assert(numDevices == 1);

    // we only partition along the z-direction. Each partition has uniform_z
    // along the z-direction. The rest is distribute to make the partitions
    // as equal as possible
    int32_t uniform_z = getDimension().z / worldSize;
    int32_t reminder = getDimension().z % worldSize;

    mData->world.firstZIndex[0] = 0;
    backend.forEachMPIRank([&](const int& rank_idx) {
        mData->world.partitionDims[rank_idx].x = getDimension().x;
        mData->world.partitionDims[rank_idx].y = getDimension().y;
        if (rank_idx < reminder) {
            mData->world.partitionDims[rank_idx].z = uniform_z + 1;
        } else {
            mData->world.partitionDims[rank_idx].z = uniform_z;
        }
        if (rank_idx > 0) {
            mData->world.firstZIndex[rank_idx] = mData->world.firstZIndex[rank_idx - 1] +
                                                 mData->world.partitionDims[rank_idx - 1].z;
        }
    });


    {  // Computing halo size
        // we partition along z so we only need halo along z
        mData->halo = Neon::index_3d(0, 0, 0);
        for (const auto& ngh : stencil.neighbours()) {
            mData->halo.z = std::max(mData->halo.z, std::abs(ngh.z));
        }
    }

    {  // Computing halo size
        for (const auto& dw : DataViewUtil::validOptions()) {
            getDefaultLaunchParameters(dw) = getLaunchParameters(dw, defaultBlockSize, 0);
        }
    }

    {  // Initialization of the span table
        mData->spanTable.forEachConfiguration([&](Neon::Execution,
                                                  Neon::SetIdx   setIdx,
                                                  Neon::DataView dw,
                                                  ncclSpan&      span) {
            span.mDataView = dw;
            span.mZghostRadius = mData->halo.z;
            span.mZboundaryRadius = mData->halo.z;
            span.mMaxZInDomain = mData->world.partitionDims[myRank].z;

            switch (dw) {
                case Neon::DataView::STANDARD: {
                    // Only works z partitions.
                    assert(mData->halo.x == 0 && mData->halo.y == 0);
                    span.mSpanDim = mData->world.partitionDims[myRank];
                    break;
                }
                case Neon::DataView::BOUNDARY: {
                    // Only works z partitions.
                    assert(mData->halo.x == 0 && mData->halo.y == 0);
                    span.mSpanDim = mData->world.partitionDims[myRank];
                    span.mSpanDim.z = span.mZboundaryRadius * 2;
                    break;
                }
                case Neon::DataView::INTERNAL: {
                    auto dims = getDevSet().newDataSet<index_3d>();
                    // Only works z partitions.
                    assert(mData->halo.x == 0 && mData->halo.y == 0);

                    span.mSpanDim = mData->world.partitionDims[myRank];
                    span.mSpanDim.z = span.mSpanDim.z - span.mZboundaryRadius * 2;
                    if (span.mSpanDim.z <= 0) {
                        NeonException exp("ncclGrid");
                        exp << "The grid size is too small to support the data view model correctly \n";
                        exp << span.mSpanDim << " for setIdx " << setIdx << " and device " << getDevSet().devId(setIdx);
                        NEON_THROW(exp);
                    }

                    break;
                }
                default: {
                    NeonException exc("ncclFieldDev");
                    NEON_THROW(exc);
                }
            }
        });

        mData->elementsPerPartition.forEachConfiguration([&](Neon::Execution execution,
                                                             Neon::SetIdx    setIdx,
                                                             Neon::DataView  dw,
                                                             int&            count) {
            if (Execution::host == execution) {
                count = mData->spanTable.getSpan(Neon::Execution::host, setIdx, dw).mSpanDim.rMul();
            }
        });
    }


    {  // a Grid allocation
        Neon::set::DataSet<size_t> elementPerPartition = backend.devSet().template newDataSet<size_t>(
            [&](Neon::SetIdx setIdx, size_t& count) {
                assert(setIdx.idx() == 0);
                size_3d dim = mData->world.partitionDims[myRank].newType<size_t>();
                dim.z += mData->halo.z * 2;
                count = dim.rMul();
            });
        mData->memoryGrid = Neon::aGrid(backend, elementPerPartition);
    }

    {  // Stencil Idx to 3d offset
        auto nPoints = backend.devSet().newDataSet<uint64_t>(stencil.nNeighbours());
        mData->stencilIdTo3dOffset = backend.devSet().template newMemSet<int8_3d>(Neon::DataUse::HOST_DEVICE,
                                                                                  1,
                                                                                  backend.getMemoryOptions(),
                                                                                  nPoints);
        for (int i = 0; i < stencil.nNeighbours(); ++i) {
            for (int devIdx = 0; devIdx < backend.devSet().setCardinality(); devIdx++) {
                index_3d      pLong = stencil.neighbours()[i];
                Neon::int8_3d pShort = pLong.newType<int8_t>();
                mData->stencilIdTo3dOffset.eRef(devIdx, i) = pShort;
            }
        }
        mData->stencilIdTo3dOffset.updateDeviceData(backend, Neon::Backend::mainStreamIdx);
    }

    {  // Init base class information
        Neon::set::DataSet<size_t> nElementsPerPartition = backend.devSet().template newDataSet<size_t>([this](Neon::SetIdx idx, size_t& size) {
            size = mData->world.partitionDims[idx.idx()].template rMulTyped<size_t>();
        });
        ncclGrid::GridBase::init("ncclGrid",
                                 backend,
                                 dimension,
                                 stencil,
                                 nElementsPerPartition,
                                 defaultBlockSize,
                                 spacing,
                                 origin,
                                 Neon::domain::tool::spaceCurves::EncoderType::sweep,
                                 {0, 0, 0});
    }
}


template <typename T, int C>
auto ncclGrid::newField(const std::string&  fieldUserName,
                        int                 cardinality,
                        [[maybe_unused]] T  inactiveValue,
                        Neon::DataUse       dataUse,
                        Neon::MemoryOptions memoryOptions) const
    -> ncclField<T, C>
{
    memoryOptions = getDevSet().sanitizeMemoryOption(memoryOptions);

    const auto haloStatus = Neon::domain::haloStatus_et::ON;

    if (C != 0 && cardinality != C) {
        NeonException exception("ncclGrid::newField Dynamic and static cardinality do not match.");
        NEON_THROW(exception);
    }

    ncclField<T, C> field(fieldUserName,
                          dataUse,
                          memoryOptions,
                          *this,
                          mData->world.partitionDims,
                          mData->halo.z,
                          haloStatus,
                          cardinality,
                          mData->stencilIdTo3dOffset);

    return field;
}

template <Neon::Execution execution,
          typename LoadingLambda>
auto ncclGrid::newContainer(const std::string& name,
                            LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
    const Neon::index_3d& defaultBlockSize = getDefaultBlock();
    Neon::set::Container  c = Neon::set::Container::factory<execution>(name,
                                                                       Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                       *this,
                                                                       lambda,
                                                                       defaultBlockSize,
                                                                       [](const Neon::index_3d&) { return 0; });
    return c;
}

template <Neon::Execution execution,
          typename LoadingLambda>
auto ncclGrid::newContainer(const std::string& name,
                            index_3d           blockSize,
                            size_t             sharedMem,
                            LoadingLambda      lambda)
    const
    -> Neon::set::Container
{
    Neon::set::Container c = Neon::set::Container::factory<execution>(name,
                                                                      Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                      *this,
                                                                      lambda,
                                                                      blockSize,
                                                                      [sharedMem](const Neon::index_3d&) { return sharedMem; });
    return c;
}

template <typename T>
auto ncclGrid::newPatternScalar() const -> Neon::template PatternScalar<T>
{
    auto pattern = Neon::PatternScalar<T>(getBackend(), mData->reduceEngine);

    if (mData->reduceEngine == Neon::sys::patterns::Engine::CUB) {
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
auto ncclGrid::dot([[maybe_unused]] const std::string&               name,
                   [[maybe_unused]] ncclField<T>&                    input1,
                   [[maybe_unused]] ncclField<T>&                    input2,
                   [[maybe_unused]] Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T>
auto ncclGrid::norm2([[maybe_unused]] const std::string&               name,
                     [[maybe_unused]] ncclField<T>&                    input,
                     [[maybe_unused]] Neon::template PatternScalar<T>& scalar,
                     [[maybe_unused]] Neon::Execution                  execution) const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

}  // namespace Neon::domain::details::ncclGrid