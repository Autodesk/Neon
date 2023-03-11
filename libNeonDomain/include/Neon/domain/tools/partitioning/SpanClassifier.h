#pragma once
#include "Neon/core/core.h"


#include "Cassifications.h"
#include "Neon/domain/tools/PointHashTable.h"
#include "Neon/domain/tools/partitioning/SpanDecomposition.h"

namespace Neon::domain::tool::partitioning {

class SpanClassifier
{
   public:
    SpanClassifier() = default;

    template <typename ActiveCellLambda,
              typename BcLambda,
              typename Block3dIdxToBlockOrigin,
              typename GetVoxelAbsolute3DIdx>
    SpanClassifier(const Neon::Backend&                              backend,
                   const ActiveCellLambda&                           activeCellLambda,
                   const BcLambda&                                   bcLambda,
                   const Block3dIdxToBlockOrigin&                    block3dIdxToBlockOrigin,
                   const GetVoxelAbsolute3DIdx&                      getVoxelAbsolute3DIdx,
                   const Neon::int32_3d&                             block3DSpan,
                   const int&                                        dataBlockEdge,
                   const Neon::int32_3d&                             domainSize,
                   const Neon::domain::Stencil                       stencil,
                   const int&                                        discreteVoxelSpacing,
                   std::shared_ptr<partitioning::SpanDecomposition> sp);


    /**
     * For the partition setIdx, it returns a vector that maps local ids to 3d points.
     * The local ids are local in terms of partition, domain and direction classes.
     */
    [[nodiscard]] auto getMapper1Dto3D(Neon::SetIdx const& setIdx,
                                       ByPartition,
                                       ByDirection,
                                       ByDomain) const
        -> const std::vector<Neon::index_3d>&;

    /**
     * For the partition setIdx, it returns a hash object that maps 3d points to local ids
     * The local ids are local in terms of partition, domain and direction classes.
     */
    [[nodiscard]] auto getMapper3Dto1D(Neon::SetIdx const& setIdx,
                                       ByPartition,
                                       ByDirection,
                                       ByDomain) const
        -> const Neon::domain::tool::PointHashTable<int32_t, uint32_t>&;

    [[nodiscard]] auto countInternal(Neon::SetIdx setIdx,
                                     ByDomain     byDomain) const -> int;

    [[nodiscard]] auto countInternal(Neon::SetIdx setIdx) const -> int;

    [[nodiscard]] auto countBoundary(Neon::SetIdx setIdx,
                                     ByDirection  byDirection,
                                     ByDomain     byDomain) const -> int;


    [[nodiscard]] auto countBoundary(Neon::SetIdx setIdx) const -> int;

    auto getMapper1Dto3D(Neon::SetIdx const& setIdx,
                         ByPartition,
                         ByDirection,
                         ByDomain)
        -> std::vector<Neon::index_3d>&;

    auto getMapper3Dto1D(Neon::SetIdx const& setIdx,
                         ByPartition,
                         ByDirection,
                         ByDomain)
        -> Neon::domain::tool::PointHashTable<int32_t, uint32_t>&;

   private:
    auto addPoint(Neon::SetIdx const&   setIdx,
                  Neon::int32_3d const& int323D,
                  ByPartition           byPartition,
                  ByDirection           byDirection,
                  ByDomain              byDomain) -> void;


    struct Info
    {
        std::vector<Neon::index_3d>                           id1dTo3d;
        Neon::domain::tool::PointHashTable<int32_t, uint32_t> id3dTo1d;
    };

    using Leve0_Info = Info;
    using Leve1_ByDomain = std::array<Leve0_Info, 2>;
    using Leve2_ByDirection = std::array<Leve1_ByDomain, 2>;
    using Leve3_ByPartition = std::array<Leve2_ByDirection, 2>;
    using Data = Neon::set::DataSet<Leve3_ByPartition>;

    Data                               mData;
    std::shared_ptr<SpanDecomposition> mSpanDecomposition;
};


template <typename ActiveCellLambda,
          typename BcLambda,
          typename Block3dIdxToBlockOrigin,
          typename GetVoxelAbsolute3DIdx>
SpanClassifier::SpanClassifier(const Neon::Backend& backend,
                               const ActiveCellLambda&,
                               const BcLambda& bcLambda,
                               const Block3dIdxToBlockOrigin&,
                               const GetVoxelAbsolute3DIdx&,
                               const Neon::int32_3d& block3DSpan,
                               const int&            dataBlockEdge,
                               const Neon::int32_3d&,
                               const Neon::domain::Stencil stencil,
                               const int&,
                               std::shared_ptr<SpanDecomposition> spanDecompositionNoUse)
{
    mData = backend.devSet().newDataSet<Leve3_ByPartition>();
    mSpanDecomposition = spanDecompositionNoUse;

    mData.forEachSeq([&](SetIdx, auto& leve3ByPartition) {
        //        using Leve0_Info = Info;
        //        using Leve1_ByDomain = std::array<Leve0_Info, 2>;
        //        using Leve2_ByDirection = std::array<Leve1_ByDomain, 2>;
        //        using Leve3_ByPartition = std::array<Leve2_ByDirection, 2>;
        //        using Data = Neon::set::DataSet<Leve3_ByPartition>;
        for (auto& level2 : leve3ByPartition) {
            for (auto& level1 : level2) {
                for (auto& level0 : level1) {
                    level0.id3dTo1d = Neon::domain::tool::PointHashTable<int32_t, uint32_t>(block3DSpan);
                }
            }
        }
    });

    // Computing the stencil radius at block granularity
    // If the dataBlockEdge is equal to 1 (element sparse block) the radius is
    // the same as the stencil radius.
    auto const zRadius = [&stencil, dataBlockEdge]() -> int {
        auto maxRadius = stencil.getRadius();
        maxRadius = ((maxRadius - 1) / dataBlockEdge) + 1;
        return maxRadius;
    }();

    // For each Partition
    backend.devSet()
        .forEachSetIdxSeq(
            [&](const Neon::SetIdx& setIdx) {
                int beginZ = mSpanDecomposition->getFirstZSliceIdx()[setIdx];
                int lastZ = mSpanDecomposition->getLastZSliceIdx()[setIdx];

                std::vector<int> const boundaryDwSlices = [&] {
                    std::vector<int> result;
                    for (int i = 0; i < zRadius; i++) {
                        result.push_back(beginZ + i);
                    }
                    return result;
                }();

                std::vector<int> const boundaryUpSlices = [&] {
                    std::vector<int> result;
                    for (int i = zRadius - 1; i >= 0; i--) {
                        result.push_back(lastZ - i);
                    }
                    return result;
                }();

                // We are running in the inner partition blocks
                for (int z = beginZ + zRadius; z <= lastZ - zRadius; z++) {
                    for (int y = 0; y < block3DSpan.y; y++) {
                        for (int x = 0; x < block3DSpan.x; x++) {
                            Neon::int32_3d const point(x, y, z);
                            ByPartition const    byPartition = ByPartition::internal;
                            ByDomain const       byDomain = bcLambda(point) ? ByDomain::bc : ByDomain::bulk;
                            addPoint(setIdx, point, byPartition, ByDirection::up, byDomain);
                        }
                    }
                }
                // We are running in the inner partition blocks
                for (auto& z : boundaryDwSlices) {
                    for (int y = 0; y < block3DSpan.y; y++) {
                        for (int x = 0; x < block3DSpan.x; x++) {
                            Neon::int32_3d const point(x, y, z);
                            ByPartition const    byPartition = ByPartition::boundary;
                            ByDirection const    byDirection = ByDirection::down;
                            ByDomain const       byDomain = bcLambda(point) ? ByDomain::bc : ByDomain::bulk;
                            addPoint(setIdx, point, byPartition, byDirection, byDomain);
                        }
                    }
                }

                // We are running in the inner partition blocks
                for (auto& z : boundaryUpSlices) {
                    for (int y = 0; y < block3DSpan.y; y++) {
                        for (int x = 0; x < block3DSpan.x; x++) {
                            Neon::int32_3d const point(x, y, z);
                            ByPartition const    byPartition = ByPartition::boundary;
                            ByDirection const    byDirection = ByDirection::up;
                            ByDomain const       byDomain = bcLambda(point) ? ByDomain::bc : ByDomain::bulk;
                            addPoint(setIdx, point, byPartition, byDirection, byDomain);
                        }
                    }
                }
            });
}
}  // namespace Neon::domain::tool::partitioning
