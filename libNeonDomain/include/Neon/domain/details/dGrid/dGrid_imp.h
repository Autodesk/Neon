#pragma once
#include "dGrid.h"

namespace Neon::domain::details::dGrid {


template <Neon::domain::SparsityPattern ActiveCellLambda>
dGrid::dGrid(const Neon::Backend&  backend,
             const Neon::int32_3d& dimension,
             const ActiveCellLambda& /*activeCellLambda*/,
             const Neon::domain::Stencil& stencil,
             const Vec_3d<double>&        spacing,
             const Vec_3d<double>&        origin)
{
    mData = std::make_shared<Data>(backend);
    const index_3d defaultBlockSize(256, 1, 1);

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
                              spacing,
                              origin);
    }

    const int32_t numDevices = getBackend().devSet().setCardinality();
    if (numDevices == 1) {
        // Single device
        mData->partitionDims[0] = getDimension();
        mData->firstZIndex[0] = 0;
    } else if (getDimension().z < numDevices) {
        NeonException exc("dGrid_t");
        exc << "The grid size in the z-direction (" << getDimension().z << ") is less the number of devices (" << numDevices
            << "). It is ambiguous how to distribute the gird";
        NEON_THROW(exc);
    } else {
        // we only partition along the z-direction. Each partition has uniform_z
        // along the z-direction. The rest is distribute to make the partitions
        // as equal as possible
        int32_t uniform_z = getDimension().z / numDevices;
        int32_t reminder = getDimension().z % numDevices;

        mData->firstZIndex[0] = 0;
        backend.forEachDeviceSeq([&](const Neon::SetIdx& setIdx) {
            mData->partitionDims[setIdx].x = getDimension().x;
            mData->partitionDims[setIdx].y = getDimension().y;
            if (setIdx < reminder) {
                mData->partitionDims[setIdx].z = uniform_z + 1;
            } else {
                mData->partitionDims[setIdx].z = uniform_z;
            }
            if (setIdx.idx() > 0) {
                mData->firstZIndex[setIdx] = mData->firstZIndex[setIdx - 1] +
                                             mData->partitionDims[setIdx - 1].z;
            }
        });
    }

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
        const int setCardinality = getDevSet().setCardinality();
        mData->spanTable.forEachConfiguration([&](Neon::SetIdx   setIdx,
                                                  Neon::DataView dw,
                                                  dSpan&         span) {
            span.mDataView = dw;
            span.mZHaloRadius = setCardinality == 1 ? 0 : mData->halo.z;
            span.mZBoundaryRadius = mData->halo.z;

            switch (dw) {
                case Neon::DataView::STANDARD: {
                    // Only works z partitions.
                    assert(mData->halo.x == 0 && mData->halo.y == 0);

                    span.mDim = mData->partitionDims[setIdx];
                    break;
                }
                case Neon::DataView::BOUNDARY: {
                    auto dims = getDevSet().newDataSet<index_3d>();
                    // Only works z partitions.
                    assert(mData->halo.x == 0 && mData->halo.y == 0);

                    span.mDim = mData->partitionDims[setIdx];
                    span.mDim.z = span.mZBoundaryRadius * 2;

                    break;
                }
                case Neon::DataView::INTERNAL: {
                    auto dims = getDevSet().newDataSet<index_3d>();
                    // Only works z partitions.
                    assert(mData->halo.x == 0 && mData->halo.y == 0);

                    span.mDim = mData->partitionDims[setIdx];
                    span.mDim.z = span.mDim.z - span.mZBoundaryRadius * 2;
                    if (span.mDim.z <= 0 && setCardinality > 1) {
                        NeonException exp("dGrid");
                        exp << "The grid size is too small to support the data view model correctly \n";
                        exp << span.mDim << " for setIdx " << setIdx << " and device " << getDevSet().devId(setIdx);
                        NEON_THROW(exp);
                    }

                    break;
                }
                default: {
                    NeonException exc("dFieldDev");
                    NEON_THROW(exc);
                }
            }
        });

        mData->elementsPerPartition.forEachConfiguration([&](Neon::SetIdx   setIdx,
                                                             Neon::DataView dw,
                                                             int&           count) {
            count = mData->spanTable.getSpan(setIdx, dw).mDim.rMul();
        });
    }


    {  // a Grid allocation
        auto haloStatus = Neon::domain::haloStatus_et::ON;
        haloStatus = (backend.devSet().setCardinality() == 1) ? haloStatus_et::e::OFF : haloStatus;
        auto                       partitionMemoryDim = mData->partitionDims;
        Neon::set::DataSet<size_t> elementPerPartition = backend.devSet().template newDataSet<size_t>(
            [&](Neon::SetIdx setIdx, size_t& count) {
                size_3d dim = partitionMemoryDim[setIdx.idx()].newType<size_t>();
                if (haloStatus == Neon::domain::haloStatus_et::ON) {
                    dim.z += mData->halo.z * 2;
                }
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
                Neon::int8_3d pShort(pLong.x, pLong.y, pLong.z);
                mData->stencilIdTo3dOffset.eRef(devIdx, i) = pShort;
            }
        }
        // mData->stencilIdTo3dOffset.updateCompute(backend, Neon::Backend::mainStreamIdx);
    }

    {  // Init base class information
        Neon::set::DataSet<size_t> nElementsPerPartition = backend.devSet().template newDataSet<size_t>([this](Neon::SetIdx idx, size_t& size) {
            size = mData->partitionDims[idx.idx()].template rMulTyped<size_t>();
        });
        dGrid::GridBase::init("dGrid",
                              backend,
                              dimension,
                              stencil,
                              nElementsPerPartition,
                              defaultBlockSize,
                              spacing,
                              origin);
    }
}


template <typename T, int C>
auto dGrid::newField(const std::string&  fieldUserName,
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
                       mData->partitionDims,
                       mData->halo.z,
                       haloStatus,
                       cardinality,
                       mData->stencilIdTo3dOffset);

    return field;
}

template <typename LoadingLambda>
auto dGrid::newContainer(const std::string& name,
                         LoadingLambda      lambda,
                         Neon::Execution    execution)
    const
    -> Neon::set::Container
{
    const Neon::index_3d& defaultBlockSize = getDefaultBlock();
    Neon::set::Container  kContainer = Neon::set::Container::factory(name,
                                                                     execution,
                                                                     Neon::set::internal::ContainerAPI::DataViewSupport::on,
                                                                     *this,
                                                                     lambda,
                                                                     defaultBlockSize,
                                                                     [](const Neon::index_3d&) { return size_t(0); });
    return kContainer;
}

template <typename LoadingLambda>
auto dGrid::newContainer(const std::string& name,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda,
                         Neon::Execution    execution)
    const
    -> Neon::set::Container
{
    Neon::set::Container kContainer = Neon::set::Container::factory(name,
                                                                    execution,
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
auto dGrid::dot([[maybe_unused]] const std::string&               name,
                [[maybe_unused]] dField<T>&                       input1,
                [[maybe_unused]] dField<T>&                       input2,
                [[maybe_unused]] Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T>
auto dGrid::norm2([[maybe_unused]] const std::string&               name,
                  [[maybe_unused]] dField<T>&                       input,
                  [[maybe_unused]] Neon::template PatternScalar<T>& scalar,
                  [[maybe_unused]] Neon::Execution                  execution) const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

}  // namespace Neon::domain::details::dGrid