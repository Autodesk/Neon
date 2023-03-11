#pragma once
#include "eGrid.h"

namespace Neon::domain::details::eGrid {


template <Neon::domain::SparsityPattern ActiveCellLambda>
eGrid::eGrid(const Neon::Backend&         backend,
             const Neon::int32_3d&        dimension,
             const ActiveCellLambda&      activeCellLambda,
             const Neon::domain::Stencil& stencil,
             const Vec_3d<double>&        spacing,
             const Vec_3d<double>&        origin)
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


    mData->partitioner1D = Neon::domain::tool::Partitioner1D(
        backend,
        activeCellLambda,
        [](Neon::index_3d idx) { return false; },
        1,
        dimension,
        1);


    mData->mConnectivityAField = mData->partitioner1D.getConnectivity();
    mData->mGlobalMappingAField = mData->partitioner1D.getGlobalMapping();
    mData->mStencil3dTo1dOffset = mData->partitioner1D.getStencil3dTo1dOffset();
    mData->memoryGrid  = mData->partitioner1D.getMemoryGrid();

    const int32_t numDevices = getBackend().devSet().setCardinality();

    if (numDevices > 1 && getDimension().z < numDevices) {
        NeonException exc("dGrid_t");
        exc << "The grid size in the z-direction (" << getDimension().z << ") is less the number of devices (" << numDevices
            << "). It is ambiguous how to distribute the gird";
        NEON_THROW(exc);
    }


    {
        // Initialization of the SPAN table
        const int setCardinality = getDevSet().setCardinality();
        mData->spanTable.forEachConfiguration([&](Neon::SetIdx   setIdx,
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

        mData->elementsPerPartition.forEachConfiguration([&](Neon::SetIdx   setIdx,
                                                             Neon::DataView dw,
                                                             int&           count) {
            count = mData->spanTable.getSpan(setIdx, dw).mCount;
        });
    }


    //
    //    {  // Stencil Idx to 3d offset
    //        auto nPoints = backend.devSet().newDataSet<uint64_t>(stencil.nNeighbours());
    //        mData->stencilIdTo3dOffset = backend.devSet().template newMemSet<int8_3d>(Neon::DataUse::HOST_DEVICE,
    //                                                                                  1,
    //                                                                                  backend.getMemoryOptions(),
    //                                                                                  nPoints);
    //        for (int i = 0; i < stencil.nNeighbours(); ++i) {
    //            for (int devIdx = 0; devIdx < backend.devSet().setCardinality(); devIdx++) {
    //                index_3d      pLong = stencil.neighbours()[i];
    //                Neon::int8_3d pShort(pLong.x, pLong.y, pLong.z);
    //                mData->stencilIdTo3dOffset.eRef(devIdx, i) = pShort;
    //            }
    //        }
    //        // mData->stencilIdTo3dOffset.updateCompute(backend, Neon::Backend::mainStreamIdx);
    //    }

    {  // Init base class information
        Neon::set::DataSet<size_t> nElementsPerPartition = backend.devSet().template newDataSet<size_t>([this](Neon::SetIdx idx, size_t& size) {
            size = mData->partitionDims[idx.idx()].template rMulTyped<size_t>();
        });
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


template <typename T, int C>
auto eGrid::newField(const std::string&  fieldUserName,
                     int                 cardinality,
                     [[maybe_unused]] T  inactiveValue,
                     Neon::DataUse       dataUse,
                     Neon::MemoryOptions memoryOptions) const
    -> eField<T, C>
{
    memoryOptions = getDevSet().sanitizeMemoryOption(memoryOptions);

    const auto haloStatus = Neon::domain::haloStatus_et::ON;

    if (C != 0 && cardinality != C) {
        NeonException exception("dGrid::newField Dynamic and static cardinality do not match.");
        NEON_THROW(exception);
    }

    eField<T, C> field(fieldUserName,
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
auto eGrid::newContainer(const std::string& name,
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
auto eGrid::newContainer(const std::string& name,
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
auto eGrid::newPatternScalar() const -> Neon::template PatternScalar<T>
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
auto eGrid::dot([[maybe_unused]] const std::string&               name,
                [[maybe_unused]] eField<T>&                       input1,
                [[maybe_unused]] eField<T>&                       input2,
                [[maybe_unused]] Neon::template PatternScalar<T>& scalar) const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

template <typename T>
auto eGrid::norm2([[maybe_unused]] const std::string&               name,
                  [[maybe_unused]] eField<T>&                       input,
                  [[maybe_unused]] Neon::template PatternScalar<T>& scalar,
                  [[maybe_unused]] Neon::Execution                  execution) const -> Neon::set::Container
{
    NEON_DEV_UNDER_CONSTRUCTION("");
}

}  // namespace Neon::domain::details::eGrid