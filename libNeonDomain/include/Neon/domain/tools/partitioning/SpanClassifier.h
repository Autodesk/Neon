#pragma once
#include "Neon/core/core.h"


#include "Cassifications.h"
#include "Neon/domain/tools/PointHashTable.h"
#include "Neon/domain/tools/partitioning/SpanDecomposition.h"

namespace Neon::domain::tools::partitioning {

class SpanClassifier
{
   public:
    SpanClassifier() = default;

    template <typename ActiveCellLambda,
              typename BcLambda,
              typename Block3dIdxToBlockOrigin,
              typename GetVoxelAbsolute3DIdx>
    SpanClassifier(const Neon::Backend&           backend,
                   const ActiveCellLambda&        activeCellLambda,
                   const BcLambda&                bcLambda,
                   const Block3dIdxToBlockOrigin& block3dIdxToBlockOrigin,
                   const GetVoxelAbsolute3DIdx&   getVoxelAbsolute3DIdx,
                   const Neon::int32_3d&          block3DSpan,
                   const int&                     blockSize,
                   const Neon::int32_3d&          domainSize,
                   const Neon::domain::Stencil    stencil,
                   const int&                     discreteVoxelSpacing,
                   const SpanDecomposition&);


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

    [[nodiscard]] auto countBoundary(Neon::SetIdx setIdx,
                                     ByDirection  byDirection,
                                     ByDomain     byDomain) const -> int;

   private:
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

    Data mData;
};

auto SpanClassifier::addPoint(const SetIdx&   setIdx,
                              const int32_3d& int323D,
                              ByPartition     byPartition,
                              ByDirection     byDirection,
                              ByDomain        byDomain)
    -> void
{
    auto& vec = getMapper1Dto3D(setIdx, byPartition, byDirection, byDomain);
    auto& hash = getMapper3Dto1D(setIdx, byPartition, byDirection, byDomain);

    vec.push_back(int323D);
    uint32_t key_value = vec.size();
    hash.addPoint(int323D, key_value);
}


auto SpanClassifier::getMapper1Dto3D(const SetIdx& setIdx,
                                     ByPartition   byPartition,
                                     ByDirection   byDirection,
                                     ByDomain      byDomain) const -> const std::vector<Neon::index_3d>&
{
    if (byDirection == ByDirection::down && byPartition == ByPartition::internal) {
        NEON_THROW_UNSUPPORTED_OPERATION("");
    }
    return mData[setIdx]
                [static_cast<int>(byPartition)]
                [static_cast<int>(byDirection)]
                [static_cast<int>(byDomain)]
                    .id1dTo3d;
}

auto SpanClassifier::getMapper3Dto1D(const SetIdx& setIdx,
                                     ByPartition   byPartition,
                                     ByDirection   byDirection,
                                     ByDomain      byDomain) const -> const Neon::domain::tool::PointHashTable<int32_t, uint32_t>&
{
    return mData[setIdx]
                [static_cast<int>(byPartition)]
                [static_cast<int>(byDirection)]
                [static_cast<int>(byDomain)]
                    .id3dTo1d;
}

auto SpanClassifier::getMapper1Dto3D(const SetIdx& setIdx,
                                     ByPartition   byPartition,
                                     ByDirection   byDirection,
                                     ByDomain      byDomain)
    -> std::vector<Neon::index_3d>&
{
    return mData[setIdx]
                [static_cast<int>(byPartition)]
                [static_cast<int>(byDirection)]
                [static_cast<int>(byDomain)]
                    .id1dTo3d;
}

auto SpanClassifier::getMapper3Dto1D(const SetIdx& setIdx,
                                     ByPartition   byPartition,
                                     ByDirection   byDirection,
                                     ByDomain      byDomain)
    -> Neon::domain::tool::PointHashTable<int32_t, uint32_t>&
{
    return mData[setIdx]
                [static_cast<int>(byPartition)]
                [static_cast<int>(byDirection)]
                [static_cast<int>(byDomain)]
                    .id3dTo1d;
}


auto SpanClassifier::countInternal(Neon::SetIdx setIdx,
                                   ByDomain     byDomain) const -> int
{
    auto count = getMapper1Dto3D(setIdx, ByPartition::internal, ByDirection::up, byDomain).size();
    return static_cast<int>(count);
}

auto SpanClassifier::countBoundary(Neon::SetIdx setIdx, ByDirection byDirection, ByDomain byDomain) const -> int
{
    auto count = getMapper1Dto3D(setIdx, ByPartition::boundary, byDirection, byDomain).size();
    return static_cast<int>(count);
}

template <typename ActiveCellLambda,
          typename BcLambda,
          typename Block3dIdxToBlockOrigin,
          typename GetVoxelAbsolute3DIdx>
SpanClassifier::SpanClassifier(const Neon::Backend&           backend,
                               const ActiveCellLambda&        activeCellLambda,
                               const BcLambda&                bcLambda,
                               const Block3dIdxToBlockOrigin& block3dIdxToBlockOrigin,
                               const GetVoxelAbsolute3DIdx&   getVoxelAbsolute3DIdx,
                               const Neon::int32_3d&          block3DSpan,
                               const int&                     blockSize,
                               const Neon::int32_3d&          domainSize,
                               const Neon::domain::Stencil    stencil,
                               const int&                     discreteVoxelSpacing,
                               const SpanDecomposition&       spanPartitioner)
{
    mData = backend.devSet().newDataSet<Leve3_ByPartition>();

    auto const zRadius = [&stencil]() -> int {
        auto maxRadius = 0;
        for (auto const& point : stencil.neighbours()) {
            auto newRadius = point.z >= 0 ? point.z : -1 * point.z;
            if (newRadius > maxRadius) {
                maxRadius = newRadius;
            }
        }
        return maxRadius;
    };

    // For each Partition
    backend.devSet()
        .forEachSetIdxSeq(
            [&](const Neon::SetIdx& setIdx) {
                int beginZ = spanPartitioner.getFirstZSliceIdx()[setIdx];
                int lastZ = spanPartitioner.getLastZSliceIdx()[setIdx];

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
                        result.push_back(lastZ - zRadius);
                    }
                    return result;
                }();

                // We are running in the inner partition blocks
                for (int z = beginZ + zRadius; z < lastZ - zRadius; z++) {
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
}  // namespace Neon::domain::tools::partitioning
