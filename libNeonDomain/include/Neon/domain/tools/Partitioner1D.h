#pragma once

#include "Neon/domain/aGrid.h"
#include "Neon/domain/tools/partitioning/Cassifications.h"
#include "Neon/domain/tools/partitioning/SpanClassifier.h"
#include "Neon/domain/tools/partitioning/SpanDecomposition.h"
#include "Neon/domain/tools/partitioning/SpanLayout.h"

namespace Neon::domain::tool {

class Partitioner1D
{
   public:
    Partitioner1D() = default;

    class DenseMeta
    {
       public:
        class Meta
        {
           public:
            Meta() = default;
            Meta(int p, int i, Neon::DataView d)
            {
                setIdx = p;
                index = i;
                dw = d;
            }

            auto isValid()
                const -> bool
            {
                return setIdx != -1;
            }

            int            setIdx = -1;
            int            index = -1;
            Neon::DataView dw = Neon::DataView::STANDARD;
        };

        DenseMeta() = default;
        DenseMeta(Neon::index_3d const& d)
        {
            dim = d;
            index = std::vector<int32_t>(dim.rMulTyped<size_t>(), -1);
            invalidMeta.setIdx = -1;
            invalidMeta.index = -1;
            invalidMeta.dw = Neon::DataView::STANDARD;
        }

        auto get(Neon::int32_3d idx)
            -> Meta const&
        {
            size_t  pitch = idx.mPitch(dim);
            int32_t dataIdx = index[pitch];
            if (dataIdx == -1) {
                return invalidMeta;
            }
            Meta& valid = data[dataIdx];
            return valid;
        }

        auto add(Neon::int32_3d idx, int partId, int offset, Neon::DataView dw)
            -> void
        {
            size_t pitch = idx.mPitch(dim);
            data.emplace_back(partId, offset, dw);
            index[pitch] = int32_t(data.size() - 1);
        }

       private:
        std::vector<Meta>    data;
        std::vector<int32_t> index;
        Neon::index_3d       dim;
        Meta                 invalidMeta;
    };

    template <typename ActiveCellLambda,
              typename BcLambda>
    Partitioner1D(const Neon::Backend&        backend,
                  const ActiveCellLambda&     activeCellLambda,
                  const BcLambda&             bcLambda,
                  const int&                  dataBlockSize,
                  const Neon::int32_3d&       domainSize,
                  const Neon::domain::Stencil stencil,
                  const int&                  discreteVoxelSpacing = 1)
    {
        mData = std::make_shared<Data>();

        mData->mDataBlockSize = dataBlockSize;
        mData->mDiscreteVoxelSpacing = discreteVoxelSpacing;
        mData->mStencil = stencil;
        mData->mDomainSize = domainSize;

        Neon::int32_3d block3DSpan(NEON_DIVIDE_UP(domainSize.x, dataBlockSize),
                                   NEON_DIVIDE_UP(domainSize.y, dataBlockSize),
                                   NEON_DIVIDE_UP(domainSize.z, dataBlockSize));

        mData->block3DSpan = block3DSpan;

        std::vector<int> nBlockProjectedToZ(block3DSpan.z);

        auto block3dIdxToBlockOrigin = [&](Neon::int32_3d const& block3dIdx) {
            Neon::int32_3d blockOrigin(block3dIdx.x * dataBlockSize * discreteVoxelSpacing,
                                       block3dIdx.y * dataBlockSize * discreteVoxelSpacing,
                                       block3dIdx.z * dataBlockSize * discreteVoxelSpacing);
            return blockOrigin;
        };

        auto getVoxelAbsolute3DIdx = [&](Neon::int32_3d const& blockOrigin,
                                         Neon::int32_3d const& voxelRelative3DIdx) {
            const Neon::int32_3d id(blockOrigin.x + voxelRelative3DIdx.x * discreteVoxelSpacing,
                                    blockOrigin.y + voxelRelative3DIdx.y * discreteVoxelSpacing,
                                    blockOrigin.z + voxelRelative3DIdx.z * discreteVoxelSpacing);
            return id;
        };

        mData->spanDecomposition = std::make_shared<partitioning::SpanDecomposition>(
            backend,
            activeCellLambda,
            block3dIdxToBlockOrigin,
            getVoxelAbsolute3DIdx,
            block3DSpan,
            dataBlockSize,
            domainSize,
            discreteVoxelSpacing);

        mData->mSpanClassifier = std::make_shared<partitioning::SpanClassifier>(
            backend,
            activeCellLambda,
            bcLambda,
            block3dIdxToBlockOrigin,
            getVoxelAbsolute3DIdx,
            block3DSpan,
            dataBlockSize,
            domainSize,
            stencil,
            discreteVoxelSpacing,
            mData->spanDecomposition);

        mData->mSpanLayout = std::make_shared<partitioning::SpanLayout>(
            backend,
            mData->spanDecomposition,
            mData->mSpanClassifier);

        mData->mTopologyWithGhost = aGrid(backend,
                                          mData->mSpanLayout->getStandardAndGhostCount().typedClone<size_t>(), {251, 1, 1});
    }

    auto getBlockSpan() -> Neon::int32_3d
    {
        return mData->block3DSpan;
    }
    auto getMemoryGrid() -> Neon::aGrid&
    {
        return mData->mTopologyWithGhost;
    }

    auto getStandardAndGhostCount()
        const -> const Neon::set::DataSet<int32_t>&
    {
        return mData->mSpanLayout->getStandardAndGhostCount();
    }

    auto getStandardCount()
        const -> const Neon::set::DataSet<int32_t>&
    {
        return mData->mSpanLayout->getStandardCount();
    }

    auto getSpanClassifier()
        const -> partitioning::SpanClassifier const&;

    auto getSpanLayout()
        const -> partitioning::SpanLayout const&;

    auto getDecomposition()
        const -> partitioning::SpanDecomposition const&;

    auto getGlobalMapping()
        -> Neon::aGrid::Field<Neon::int32_3d, 0>&
    {
        if (!mData->globalMappingInit) {
            mData->globalMapping = mData->mTopologyWithGhost.template newField<Neon::int32_3d, 0>("GlobalMapping",
                                                                                                  1,
                                                                                                  Neon::int32_3d(0),
                                                                                                  Neon::DataUse::HOST_DEVICE);

            mData->mTopologyWithGhost.getBackend().forEachDeviceSeq([&](Neon::SetIdx const& setIdx) {
                int count = 0;
                using namespace partitioning;

                auto partition = mData->globalMapping.getPartition(Execution::host, setIdx);

                for (auto byPartition : {ByPartition::internal, ByPartition::boundary}) {
                    for (auto byDirection : {ByDirection::up, ByDirection::down}) {
                        if (byPartition == ByPartition::internal && byDirection == ByDirection::down) {
                            continue;
                        }
                        for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                            auto const& mapperVec = mData->mSpanClassifier->getMapper1Dto3D(
                                setIdx,
                                byPartition,
                                byDirection,
                                byDomain);
                            for (uint64_t j = 0; j < mapperVec.size(); j++) {

                                aGrid::Cell    idx(count);
                                Neon::int32_3d point3d = mapperVec[j];
                                point3d = point3d * mData->mDiscreteVoxelSpacing * mData->mDataBlockSize;
                                partition(idx, 0) = point3d;
                                count++;
                            }
                        }
                    }
                }
            });

            mData->globalMapping.updateDeviceData(Neon::Backend::mainStreamIdx);
            mData->globalMappingInit = true;
        }
        return mData->globalMapping;
    }

    template <typename Lambda>
    auto forEachSeq(Neon::SetIdx setIdx, const Lambda& lambda)
        const -> void
    {
        int count = 0;
        using namespace partitioning;

        for (auto byPartition : {ByPartition::internal, ByPartition::boundary}) {
            for (auto byDirection : {ByDirection::up, ByDirection::down}) {
                if (byPartition == ByPartition::internal && byDirection == ByDirection::down) {
                    continue;
                }
                for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                    auto const& mapperVec = mData->mSpanClassifier->getMapper1Dto3D(
                        setIdx,
                        byPartition,
                        byDirection,
                        byDomain);
                    for (const auto& point3d : mapperVec) {
                        lambda(count,
                               point3d,
                               byPartition == ByPartition::internal
                                   ? Neon::DataView::INTERNAL
                                   : Neon::DataView::BOUNDARY);
                        count++;
                    }
                }
            }
        }
    }

    auto getDenseMeta(DenseMeta& denseMeta) const
    {
        denseMeta = DenseMeta(mData->mDomainSize);
        auto const& backend = mData->mTopologyWithGhost.getBackend();
        backend.forEachDeviceSeq(
            [&](Neon::SetIdx setIdx) {
                forEachSeq(
                    setIdx,
                    [&](int offset, Neon::int32_3d const& idx3d, Neon::DataView dw) {
                        denseMeta.add(idx3d, setIdx, offset, dw);
                    });
            });
    }

    auto getStencil3dTo1dOffset()
        const -> Neon::set::MemSet<int8_t>&
    {
        if (!mData->getStencil3dTo1dOffsetInit) {
            const Backend& backend = mData->mTopologyWithGhost.getBackend();
            ;

            int32_t radius = mData->mStencil.getRadius();
            int     countElement = (2 * radius + 1);
            countElement = countElement * countElement * countElement;

            auto memSize = backend.devSet().template newDataSet<uint64_t>(countElement);

            mData->stencil3dTo1dOffset = backend.devSet().template newMemSet<int8_t>(
                Neon::DataUse::HOST_DEVICE,
                1,
                Neon::MemoryOptions(),
                memSize);

            backend.forEachDeviceSeq([&]([[maybe_unused]] SetIdx setIdx) {
                int stencilIdx = 0;
                for (int i = 0; i < 27; i++) {
                    mData->stencil3dTo1dOffset.eRef(setIdx, i, 0) = -33;
                }
                for (auto ngh : mData->mStencil.neighbours()) {
                    int     yPitch = (2 * radius + 1);
                    int     zPitch = yPitch * yPitch;
                    auto    nghShifted = ngh + radius;
                    int32_t offset = nghShifted.x + nghShifted.y * yPitch + nghShifted.z * zPitch;
                    assert(offset < countElement);
                    assert(offset >= 0);
                    // std::cout << ngh << " -> " << stencilIdx << " -> Pitch "  << offset<< std::endl;
                    mData->stencil3dTo1dOffset.eRef(setIdx, offset, 0) = stencilIdx;
                    stencilIdx++;
                }
            });

            mData->stencil3dTo1dOffset.updateDeviceData(backend, Neon::Backend::mainStreamIdx);
            mData->getStencil3dTo1dOffsetInit = true;
        }
        return mData->stencil3dTo1dOffset;
    }

    auto getConnectivity()
        -> Neon::aGrid::Field<int32_t, 0>
    {
        if (!mData->connectivityInit) {
            mData->connectivity = mData->mTopologyWithGhost.template newField<int32_t, 0>("GlobalMapping",
                                                                                          mData->mStencil.nPoints(),
                                                                                          0,
                                                                                          Neon::DataUse::HOST_DEVICE);

            mData->mTopologyWithGhost.getBackend().forEachDeviceSeq(
                [&](Neon::SetIdx const& setIdx) {
                    auto& partition = mData->connectivity.getPartition(Neon::Execution::host, setIdx);
                    using namespace partitioning;

                    // Internal voxels will read only non ghost data
                    for (auto byPartition : {ByPartition::internal}) {
                        const auto byDirection = ByDirection::up;
                        for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                            auto const& mapperVec = mData->mSpanClassifier->getMapper1Dto3D(
                                setIdx,
                                byPartition,
                                byDirection,
                                byDomain);
                            auto const start = mData->mSpanLayout->getBoundsInternal(setIdx, byDomain).first;
                            for (uint64_t blockIdx = 0; blockIdx < mapperVec.size(); blockIdx++) {
                                auto const& point3d = mapperVec[blockIdx];
                                for (int s = 0; s < mData->mStencil.nPoints(); s++) {

                                    auto const offset = mData->mStencil.neighbours()[s];

                                    auto findings = mData->mSpanLayout->findNeighbourOfInternalPoint(
                                        setIdx,
                                        point3d, offset);

                                    uint32_t const noNeighbour = std::numeric_limits<uint32_t>::max();
                                    uint32_t       targetNgh = noNeighbour;
                                    if (findings.first) {
                                        targetNgh = findings.second;
                                    }
                                    aGrid::Cell aIdx(start + blockIdx);
                                    partition(aIdx, s) = targetNgh;
                                }
                            }
                        }
                    }
                    for (auto byPartition : {ByPartition::boundary}) {
                        for (auto byDirection : {ByDirection::up, ByDirection::down}) {

                            for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                                auto const& mapperVec = mData->mSpanClassifier->getMapper1Dto3D(
                                    setIdx,
                                    byPartition,
                                    byDirection,
                                    byDomain);

                                auto const start = mData->mSpanLayout->getBoundsBoundary(setIdx, byDirection, byDomain).first;
                                for (int64_t blockIdx = 0; blockIdx < int64_t(mapperVec.size()); blockIdx++) {
                                    auto const& point3d = mapperVec[blockIdx];
                                    for (int s = 0; s < mData->mStencil.nPoints(); s++) {


                                        auto const offset = mData->mStencil.neighbours()[s];

                                        auto findings = mData->mSpanLayout->findNeighbourOfBoundaryPoint(
                                            setIdx,
                                            point3d,
                                            offset.newType<int32_t>());

                                        uint32_t const noNeighbour = std::numeric_limits<uint32_t>::max();
                                        uint32_t       targetNgh = noNeighbour;
                                        if (findings.first) {
                                            targetNgh = findings.second;
                                        }
                                        aGrid::Cell aIdx(start + blockIdx);
                                        partition(aIdx, s) = targetNgh;
                                    }
                                }
                            }
                        }
                    }
                });
            mData->connectivity.updateDeviceData(Neon::Backend::mainStreamIdx);
            mData->connectivityInit = true;
        }
        return mData->connectivity;
    }

   private:
    class Data
    {
       public:
        int                                   mDataBlockSize = 0;
        int                                   mDiscreteVoxelSpacing = 0;
        Neon::domain::Stencil                 mStencil;
        Neon::index_3d                        mDomainSize;
        Neon::int32_3d                        block3DSpan;
        bool                                  globalMappingInit = false;
        Neon::aGrid::Field<Neon::int32_3d, 0> globalMapping;

        bool                      getStencil3dTo1dOffsetInit = false;
        Neon::set::MemSet<int8_t> stencil3dTo1dOffset;

        bool                           connectivityInit = false;
        Neon::aGrid::Field<int32_t, 0> connectivity;

        std::shared_ptr<partitioning::SpanDecomposition> spanDecomposition;
        std::shared_ptr<partitioning::SpanClassifier>    mSpanClassifier;
        std::shared_ptr<partitioning::SpanLayout>        mSpanLayout;

        Neon::aGrid mTopologyWithGhost;
    };
    std::shared_ptr<Data> mData;
};

}  // namespace Neon::domain::tool