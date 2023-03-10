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

    template <typename ActiveCellLambda,
              typename BcLambda>
    Partitioner1D(const Neon::Backend&        backend,
                  const ActiveCellLambda&     activeCellLambda,
                  const BcLambda&             bcLambda,
                  const int&                  blockSize,
                  const Neon::int32_3d&       domainSize,
                  const Neon::domain::Stencil stencil,
                  const int&                  discreteVoxelSpacing = 1)
    {
        mBlockSize = blockSize;
        mDiscreteVoxelSpacing = discreteVoxelSpacing;
        mStencil = stencil;

        Neon::int32_3d block3DSpan(NEON_DIVIDE_UP(domainSize.x, blockSize),
                                   NEON_DIVIDE_UP(domainSize.y, blockSize),
                                   NEON_DIVIDE_UP(domainSize.z, blockSize));

        std::vector<int> nBlockProjectedToZ(block3DSpan.z);

        auto constexpr block3dIdxToBlockOrigin = [&](Neon::int32_3d const& block3dIdx) {
            Neon::int32_3d blockOrigin(block3dIdx.x * blockSize * discreteVoxelSpacing,
                                       block3dIdx.y * blockSize * discreteVoxelSpacing,
                                       block3dIdx.z * blockSize * discreteVoxelSpacing);
            return blockOrigin;
        };

        auto constexpr getVoxelAbsolute3DIdx = [&](Neon::int32_3d const& blockOrigin,
                                                   Neon::int32_3d const& voxelRelative3DIdx) {
            const Neon::int32_3d id(blockOrigin.x + voxelRelative3DIdx.x * discreteVoxelSpacing,
                                    blockOrigin.y + voxelRelative3DIdx.y * discreteVoxelSpacing,
                                    blockOrigin.z + voxelRelative3DIdx.z * discreteVoxelSpacing);
            return id;
        };

        mSpanPartitioner = partitioning::SpanDecomposition(
            backend,
            activeCellLambda,
            block3dIdxToBlockOrigin,
            getVoxelAbsolute3DIdx,
            block3DSpan,
            blockSize,
            domainSize,
            stencil,
            discreteVoxelSpacing);

        mSpanClassifier = partitioning::SpanClassifier(
            backend,
            activeCellLambda,
            bcLambda,
            block3dIdxToBlockOrigin,
            getVoxelAbsolute3DIdx,
            block3DSpan,
            blockSize,
            domainSize,
            discreteVoxelSpacing,
            mSpanPartitioner);

        mSpanLayout = partitioning::SpanLayout(
            backend,
            mSpanPartitioner,
            mSpanClassifier);


        mTopologyWithGhost = aGrid(backend, mSpanLayout.getStandardAndGhostCount().typedClone<size_t>(), {251, 1, 1});
    }

    auto getMemoryGrid() -> Neon::aGrid&{
        return mTopologyWithGhost;
    }

    auto getSpanClassifier()
        const -> partitioning::SpanClassifier const&;

    auto getSpanLayout()
        const -> partitioning::SpanLayout const&;

    auto getGlobalMapping() -> Neon::aGrid::Field<Neon::int32_3d, 0>
    {
        auto result = mTopologyWithGhost.template newField<Neon::int32_3d, 0>("GlobalMapping",
                                                                              1,
                                                                              Neon::int32_3d(0),
                                                                              Neon::DataUse::HOST_DEVICE);

        mTopologyWithGhost.getBackend().forEachDeviceSeq([&](Neon::SetIdx const& setIdx) {
            int count = 0;
            using namespace partitioning;

            auto partition = result.getPartition(Execution::host, setIdx);

            for (auto byPartition : {ByPartition::internal, ByPartition::boundary}) {
                for (auto byDirection : {ByDirection::up, ByDirection::down}) {
                    if (byPartition == ByPartition::internal && byDirection == ByDirection::down) {
                        continue;
                    }
                    for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                        auto const& mapperVec = mSpanClassifier.getMapper1Dto3D(
                            setIdx,
                            byPartition,
                            byDirection,
                            byDomain);
                        for (uint64_t j = 0; j < mapperVec.size(); j++) {

                            aGrid::Cell idx(count);
                            auto const& point3d = mapperVec[j];
                            partition(idx, 0) = point3d;
                            count++;
                        }
                    }
                }
            }
        });

        result.updateDeviceData(Neon::Backend::mainStreamIdx);

        return result;
    }

    auto getStencil3dTo1dOffset() const -> Neon::set::MemSet<int8_3d>
    {
        const Backend& backend = mTopologyWithGhost.getBackend();

        auto stencilNghSize = backend.devSet().template newDataSet<uint64_t>(
            mStencil.neighbours().size());

        Neon::set::MemSet<int8_3d> stencilNghIndex = backend.devSet().template newMemSet<int8_3d>(
            Neon::DataUse::HOST_DEVICE,
            1,
            Neon::MemoryOptions(),
            stencilNghSize);

        backend.forEachDeviceSeq([&](SetIdx setIdx) {
            for (int64_t s = 0; s < int64_t(mStencil.neighbours().size()); ++s) {
                stencilNghIndex.eRef(setIdx, s, 0).x = static_cast<int8_3d::Integer>(mStencil.neighbours()[s].x);
                stencilNghIndex.eRef(setIdx, s, 0).y = static_cast<int8_3d::Integer>(mStencil.neighbours()[s].y);
                stencilNghIndex.eRef(setIdx, s, 0).z = static_cast<int8_3d::Integer>(mStencil.neighbours()[s].z);
            }
        });

        stencilNghIndex.updateDeviceData(backend, Neon::Backend::mainStreamIdx);
        return stencilNghIndex;
    }

    auto getStencil1dTo3dOffset(
        const Backend&               backend,
        int                          stream,
        const Neon::domain::Stencil& stencil) const -> Neon::set::MemSet<int32_t>
    {

        auto stencilNghSize = backend.devSet().template newDataSet<uint64_t>(
            stencil.neighbours().size());

        int32_t radius = stencil.getRadius();
        int     countElement = (2 * radius + 1);
        countElement = countElement * countElement * countElement;

        Neon::set::MemSet<int32_t> stencilNghIndex = backend.devSet().template newMemSet<int32_t>(
            Neon::DataUse::HOST_DEVICE,
            countElement,
            Neon::MemoryOptions(),
            stencilNghSize);

        backend.forEachDeviceSeq([&](SetIdx setIdx) {
            int stencilIdx = 0;
            for (auto ngh : stencil.neighbours()) {
                int yPitch = countElement;
                int zPitch = countElement * countElement;

                int32_t offset = ngh.x + ngh.y * yPitch + ngh.z * zPitch;
                stencilNghIndex.eRef(setIdx, offset, 0) = stencilIdx;
                stencilIdx++;
            }
        });

        stencilNghIndex.updateDeviceData(backend, stream);
        return stencilNghIndex;
    }

    auto getConnectivity()
        -> Neon::aGrid::Field<int32_t, 0>
    {
        auto connectivityField = mTopologyWithGhost.template newField<int32_t, 0>("GlobalMapping",
                                                                                  mStencil.nPoints(),
                                                                                  0,
                                                                                  Neon::DataUse::HOST_DEVICE);

        mTopologyWithGhost.getBackend().forEachDeviceSeq(
            [&](Neon::SetIdx const& setIdx) {
                auto& partition = connectivityField.getPartition(Neon::Execution::host, setIdx);
                using namespace partitioning;

                // Internal voxels will read only non ghost data
                for (auto byPartition : {ByPartition::internal}) {
                    const auto byDirection = ByDirection::up;
                    for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                        auto const& mapperVec = mSpanClassifier.getMapper1Dto3D(
                            setIdx,
                            byPartition,
                            byDirection,
                            byDomain);
                        auto const start = mSpanLayout.getBoundsInternal(setIdx, byDomain).first;
                        for (uint64_t blockIdx = 0; blockIdx < mapperVec.size(); blockIdx++) {
                            auto const& point3d = mapperVec[blockIdx];
                            for (int s = 0; s < mStencil.nPoints(); s++) {

                                auto const offset = mStencil.neighbours()[s];

                                auto findings = mSpanLayout.findNeighbourOfInternalPoint(
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
                            auto const& mapperVec = mSpanClassifier.getMapper1Dto3D(
                                setIdx,
                                byPartition,
                                byDirection,
                                byDomain);

                            auto const start = mSpanLayout.getBoundsBoundary(setIdx, byDirection, byDomain).first;
                            for (int64_t blockIdx = 0; blockIdx < int64_t(mapperVec.size()); blockIdx++) {
                                auto const& point3d = mapperVec[blockIdx];
                                for (int s = 0; s < mStencil.nPoints(); s++) {


                                    auto const offset = mStencil.neighbours()[s];

                                    auto findings = mSpanLayout.findNeighbourOfBoundaryPoint(
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
        return connectivityField;
    }

   private:
    int                   mBlockSize = 0;
    int                   mDiscreteVoxelSpacing = 0;
    Neon::domain::Stencil mStencil;

    partitioning::SpanDecomposition mSpanPartitioner;
    partitioning::SpanClassifier    mSpanClassifier;
    partitioning::SpanLayout        mSpanLayout;

    Neon::aGrid mTopologyWithGhost;
};

}  // namespace Neon::domain::tool