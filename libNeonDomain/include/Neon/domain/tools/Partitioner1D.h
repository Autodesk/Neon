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


        mTopology = aGrid(backend, mSpanPartitioner.getNumBlockPerPartition().template typedClone<size_t>(), {251, 1, 1});
    }

    auto getSpanClassifier()
        const -> partitioning::SpanClassifier const&;

    auto getSpanLayout()
        const -> partitioning::SpanLayout const&;

    template <class T = int32_t>
    auto getConectivity() -> Neon::aGrid::Field<T, 0>
    {
        auto result = mTopology.template newField<T, 0>("Connectivity",
                                                        mStencil.nPoints(),
                                                        T(0),
                                                        Neon::DataUse::HOST_DEVICE);

        NEON_DEV_UNDER_CONSTRUCTION("");
        return result;
    }

    auto getGlobalMapping() -> Neon::aGrid::Field<Neon::int32_3d, 0>
    {
        auto result = mTopology.template newField<Neon::int32_3d, 0>("GlobalMapping",
                                                                     1,
                                                                     Neon::int32_3d(0),
                                                                     Neon::DataUse::HOST_DEVICE);

        mTopology.getBackend().forEachDeviceSeq([&](Neon::SetIdx const& setIdx) {
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
    auto getConnectivity() -> Neon::aGrid::Field<int32_t, 0>
    {
        auto connectivityField = mTopology.template newField<int32_t, 0>("GlobalMapping",
                                                              mStencil.nPoints(),
                                                              0,
                                                              Neon::DataUse::HOST_DEVICE);

        mTopology.getBackend().forEachDeviceSeq(
            [&](Neon::SetIdx const& setIdx) {
                auto& partition = connectivityField.getPartition(Neon::Execution::host, setIdx);
                using namespace partitioning;

                for (auto byPartition : {ByPartition::internal}) {
                    const auto byDirection = ByDirection::up;
                    for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                        auto const& mapperVec = mSpanClassifier.getMapper1Dto3D(
                            setIdx,
                            byPartition,
                            byDirection,
                            byDomain);

                        auto const start = mSpanL(setIdx, byDomain).first;
                        for (uint64_t blockIdx = 0; blockIdx < mapperVec.size(); blockIdx++) {
                            auto const& point3d = mapperVec[blockIdx];
                            for (int s = 0; s < stencil.nPoints(); s++) {

                                auto const offset = stencil.neighbours()[s];

                                auto findings = findNeighbourOfInternalPoint(
                                    setIdx,
                                    point3d, offset);

                                uint32_t const noNeighbour = std::numeric_limits<uint32_t>::max();
                                uint32_t       targetNgh = noNeighbour;
                                if (findings.first) {
                                    targetNgh = findings.second;
                                }
                                partition(blockIdx, s) = targetNgh;
                            }
                        }
                    }
                }
                for (auto byPartition : {ByPartition::boundary}) {
                    for (auto byDirection : {ByDirection::up, ByDirection::down}) {

                        for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                            auto const& mapperVec = mSpanClassifierPtr->getMapper1Dto3D(
                                setIdx,
                                byPartition,
                                byDirection,
                                byDomain);

                            auto const start = this->getBoundsBoundary(setIdx, byDirection, byDomain).first;
                            for (int64_t blockIdx = 0; blockIdx < int64_t(mapperVec.size()); blockIdx++) {
                                auto const& point3d = mapperVec[blockIdx];
                                for (int s = 0; s < stencil.nPoints(); s++) {


                                    auto const offset = stencil.neighbours()[s];

                                    auto findings = findNeighbourOfBoundaryPoint(
                                        setIdx,
                                        point3d,
                                        offset.newType<int32_t>());

                                    uint32_t const noNeighbour = std::numeric_limits<uint32_t>::max();
                                    uint32_t       targetNgh = noNeighbour;
                                    if (findings.first) {
                                        targetNgh = findings.second;
                                    }
                                    partition(blockIdx, s) = targetNgh;
                                }
                            }
                        }
                    }
                }
            });
    }

   private:
    int                   mBlockSize = 0;
    int                   mDiscreteVoxelSpacing = 0;
    Neon::domain::Stencil mStencil;

    partitioning::SpanDecomposition mSpanPartitioner;
    partitioning::SpanClassifier    mSpanClassifier;
    partitioning::SpanLayout        mSpanLayout;

    Neon::aGrid mTopology;
};

}  // namespace Neon::domain::tool