#pragma once
#include "Neon/core/core.h"


#include <numeric>
#include "Cassifications.h"
#include "Neon/domain/tools/PointHashTable.h"
#include "Neon/domain/tools/SpaceCurves.h"
#include "Neon/domain/tools/partitioning/SpanDecomposition.h"
namespace Neon::domain::tool::partitioning {

struct Hash
{
    std::vector<Neon::index_3d>                           id1dTo3d;
    Neon::domain::tool::PointHashTable<int32_t, uint64_t> id3dTo1d;

    auto reHash(Neon::domain::tool::spaceCurves::EncoderType encoderType) -> void
    {
        //        std::cout << "BEFORE Cartesian ";
        //        for (int i = 0; i < int(id1dTo3d.size()); i++) {
        //            std::cout << id1dTo3d[i] << " ";
        //        }
        //        std::cout << std::endl
        //                  << " ID ";
        //        for (int i = 0; i < int(id1dTo3d.size()); i++) {
        //            std::cout << *id3dTo1d.getMetadata(id1dTo3d[i]) << " ";
        //        }
        //        std::cout << std::endl
        //                  << " CODE ";
        //        for (int i = 0; i < int(id1dTo3d.size()); i++) {
        //            std::cout << Neon::domain::tool::spaceCurves::Encoder::encode(spaceCurve, id3dTo1d.getBBox(), id1dTo3d[i]) << " ";
        //        }
        //        std::cout << std::endl;
        //        std::cout << " BOX " << id3dTo1d.getBBox();
        //
        //        std::cout << std::endl;

        // Encoding all points w.r.t the encoder type
        std::vector<uint64_t> code;
        for (auto const& point : id1dTo3d) {
            code.push_back(Neon::domain::tool::spaceCurves::Encoder::encode(encoderType, id3dTo1d.getBBox(), point));
        }
        // Sort id1dTo3d w.r.t. the codes
        std::vector<std::size_t> permutation = getSortedPermutation(code, [](uint64_t a, uint64_t b) {
            return a < b;
        });
        id1dTo3d = applyPermutation(id1dTo3d, permutation);
        for (uint64_t i = 0; i < id1dTo3d.size(); i++) {
            *(id3dTo1d.getMetadata(id1dTo3d[i])) = i;
        }
        //
        //        std::cout << "AFTER Cartesian ";
        //        for (int i = 0; i < int(id1dTo3d.size()); i++) {
        //            std::cout << id1dTo3d[i] << " ";
        //        }
        //        std::cout << std::endl
        //                  << " ID ";
        //        for (int i = 0; i < int(id1dTo3d.size()); i++) {
        //            std::cout << *id3dTo1d.getMetadata(id1dTo3d[i]) << " ";
        //        }
        //        std::cout << std::endl
        //                  << " CODE ";
        //        for (int i = 0; i < int(id1dTo3d.size()); i++) {
        //            std::cout << Neon::domain::tool::spaceCurves::Encoder::encode(spaceCurve, id3dTo1d.getBBox(), id1dTo3d[i]) << " ";
        //        }
        //        std::cout << std::endl;
    }

   private:
    template <typename T, typename Compare>
    std::vector<std::size_t> getSortedPermutation(
        const std::vector<T>& vec,
        Compare const&        compare)
    {
        std::vector<std::size_t> p(vec.size());
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(),
                  [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
        return p;
    }

    template <typename T>
    std::vector<T> applyPermutation(
        const std::vector<T>&           vec,
        const std::vector<std::size_t>& p)
    {
        std::vector<T> sorted_vec(vec.size());
        std::transform(p.begin(), p.end(), sorted_vec.begin(),
                       [&](std::size_t i) { return vec[i]; });
        return sorted_vec;
    }
};

class SpanClassifier
{
   public:
    SpanClassifier() = default;

    template <typename ActiveCellLambda,
              typename BcLambda,
              typename Block3dIdxToBlockOrigin,
              typename GetVoxelAbsolute3DIdx>
    SpanClassifier(const Neon::Backend&                             backend,
                   const ActiveCellLambda&                          activeCellLambda,
                   const BcLambda&                                  bcLambda,
                   const Block3dIdxToBlockOrigin&                   block3dIdxToBlockOrigin,
                   const GetVoxelAbsolute3DIdx&                     getVoxelAbsolute3DIdx,
                   const Neon::int32_3d&                            block3DSpan,
                   const Neon::int32_3d&                            dataBlockSize3D,
                   const Neon::int32_3d&                            domainSize,
                   const Neon::domain::Stencil                      stencil,
                   const int&                                       discreteVoxelSpacing,
                   Neon::domain::tool::spaceCurves::EncoderType     encoderType,
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
        -> const Neon::domain::tool::PointHashTable<int32_t, uint64_t>&;

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
        -> Neon::domain::tool::PointHashTable<int32_t, uint64_t>&;

   private:
    auto addPoint(Neon::SetIdx const&   setIdx,
                  Neon::int32_3d const& int323D,
                  ByPartition           byPartition,
                  ByDirection           byDirection,
                  ByDomain              byDomain) -> void;


    using Leve0_Info = Hash;
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
SpanClassifier::SpanClassifier(const Neon::Backend&                         backend,
                               const ActiveCellLambda&                      activeCellLambda,
                               const BcLambda&                              bcLambda,
                               const Block3dIdxToBlockOrigin&               block3dIdxToBlockOrigin,
                               const GetVoxelAbsolute3DIdx&                 getVoxelAbsolute3DIdx,
                               const Neon::int32_3d&                        block3DSpan,
                               const Neon::int32_3d&                        dataBlockSize3D,
                               const Neon::int32_3d&                        domainSize,
                               const Neon::domain::Stencil                  stencil,
                               const int&                                   discreteVoxelSpacing,
                               Neon::domain::tool::spaceCurves::EncoderType spaceFillingType,
                               std::shared_ptr<SpanDecomposition>           spanDecompositionNoUse)
{
    mData = backend.devSet().newDataSet<Leve3_ByPartition>();
    mSpanDecomposition = spanDecompositionNoUse;

    ByDirection defaultForInternal = ByDirection::up;

    mData.forEachSeq([&](SetIdx, auto& leve3ByPartition) {
        //        using Leve0_Info = Info;
        //        using Leve1_ByDomain = std::array<Leve0_Info, 2>;
        //        using Leve2_ByDirection = std::array<Leve1_ByDomain, 2>;
        //        using Leve3_ByPartition = std::array<Leve2_ByDirection, 2>;
        //        using Data = Neon::set::DataSet<Leve3_ByPartition>;
        for (auto& level2 : leve3ByPartition) {
            for (auto& level1 : level2) {
                for (auto& level0 : level1) {
                    level0.id3dTo1d = Neon::domain::tool::PointHashTable<int32_t, uint64_t>(block3DSpan);
                }
            }
        }
    });

    // Computing the stencil radius at block granularity
    // If the dataBlockEdge is equal to 1 (element sparse block) the radius is
    // the same as the stencil radius.
    auto const zRadius = [&stencil, dataBlockSize3D]() -> int {
        auto maxRadius = stencil.getRadius();
        maxRadius = ((maxRadius - 1) / dataBlockSize3D.z) + 1;
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

                auto inspectBlock = [&](int bx, int by, int bz, ByPartition byPartition, ByDirection byDirection) {
                    Neon::int32_3d blockOrigin = block3dIdxToBlockOrigin({bx, by, bz});

                    bool     doBreak = false;
                    bool     isActiveBlock = false;
                    ByDomain byDomain = ByDomain::bulk;
                    for (int z = 0; (z < dataBlockSize3D.z && !doBreak); z++) {
                        for (int y = 0; (y < dataBlockSize3D.y && !doBreak); y++) {
                            for (int x = 0; (x < dataBlockSize3D.x && !doBreak); x++) {

                                const Neon::int32_3d globalId = getVoxelAbsolute3DIdx(blockOrigin, {x, y, z});
                                if (globalId < domainSize * discreteVoxelSpacing) {
                                    if (activeCellLambda(globalId)) {
                                        doBreak = true;
                                        isActiveBlock = true;
                                        Neon::int32_3d const pointInBlock(bx + x, by + y, bz + z);
                                        if constexpr (std::is_same_v<BcLambda, nullptr_t>) {
                                            byDomain = ByDomain::bulk;
                                            doBreak = true;
                                            break;
                                        } else if constexpr (std::is_same_v<typename std::invoke_result<BcLambda, Neon::index_3d>::type, bool>) {
                                            NEON_THROW_UNSUPPORTED_OPERATION("bool");
                                        } else if constexpr (std::is_same_v<typename std::invoke_result<BcLambda, Neon::index_3d>::type, ByDomain>) {
                                            if (bcLambda(pointInBlock) == ByDomain::bc) {
                                                byDomain = ByDomain::bc;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (isActiveBlock) {
                        Neon::int32_3d const point(bx, by, bz);
                        addPoint(setIdx, point, byPartition, byDirection, byDomain);
                    }
                };
                if (backend.deviceCount() > 1) {
                    // We are running in the inner partition blocks
                    if (beginZ + zRadius > lastZ - zRadius) {
                        std::cout << spanDecompositionNoUse->toString(backend);

                        NeonException exception("1D Partitioner");
                        exception << "Domain too small for the number of devices that was providded.\n";
                        exception << "Block Span " << block3DSpan << "\n";
                        exception << spanDecompositionNoUse->toString(backend);
                        NEON_THROW(exception);
                    }
                    for (int bz = beginZ + zRadius; bz <= lastZ - zRadius; bz++) {
                        for (int by = 0; by < block3DSpan.y; by++) {
                            for (int bx = 0; bx < block3DSpan.x; bx++) {
                                inspectBlock(bx, by, bz, ByPartition::internal, defaultForInternal);
                            }
                        }
                    }
                    // We are running in the inner partition blocks
                    for (auto& bz : boundaryDwSlices) {
                        for (int by = 0; by < block3DSpan.y; by++) {
                            for (int bx = 0; bx < block3DSpan.x; bx++) {
                                inspectBlock(bx, by, bz, ByPartition::boundary, ByDirection::down);
                            }
                        }
                    }

                    // We are running in the inner partition blocks
                    for (auto& bz : boundaryUpSlices) {
                        for (int by = 0; by < block3DSpan.y; by++) {
                            for (int bx = 0; bx < block3DSpan.x; bx++) {
                                inspectBlock(bx, by, bz, ByPartition::boundary, ByDirection::up);
                            }
                        }
                    }
                } else {
                    // We are running in the inner partition blocks
                    for (int bz = beginZ; bz <= lastZ; bz++) {
                        for (int by = 0; by < block3DSpan.y; by++) {
                            for (int bx = 0; bx < block3DSpan.x; bx++) {
                                inspectBlock(bx, by, bz, ByPartition::internal, defaultForInternal);
                            }
                        }
                    }
                }
            });

    mData.forEachSeq([&](SetIdx, auto& leve3ByPartition) {
        //        using Leve0_Info = Info;
        //        using Leve1_ByDomain = std::array<Leve0_Info, 2>;
        //        using Leve2_ByDirection = std::array<Leve1_ByDomain, 2>;
        //        using Leve3_ByPartition = std::array<Leve2_ByDirection, 2>;
        //        using Data = Neon::set::DataSet<Leve3_ByPartition>;
        for (auto& level2 : leve3ByPartition) {
            for (auto& level1 : level2) {
                for (auto& level0 : level1) {
                    level0.reHash(spaceFillingType);
                }
            }
        }
    });
}
}  // namespace Neon::domain::tool::partitioning
