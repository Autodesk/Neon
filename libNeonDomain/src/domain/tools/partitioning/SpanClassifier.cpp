#include "Neon/core/core.h"
#include "Neon/domain/tools/partitioning/SpanClassifier.h"
namespace Neon::domain::tool::partitioning {



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
    uint32_t key_value = static_cast<uint32_t>(vec.size());
    hash.addPoint(int323D, key_value-1);
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

auto SpanClassifier::countInternal(Neon::SetIdx setIdx) const -> int
{
    auto bulkCount = getMapper1Dto3D(setIdx, ByPartition::internal, ByDirection::up, ByDomain::bulk).size();
    auto bcCount = getMapper1Dto3D(setIdx, ByPartition::internal, ByDirection::up, ByDomain::bc).size();

    return static_cast<int>(bulkCount + bcCount);
}

auto SpanClassifier::countBoundary(Neon::SetIdx setIdx, ByDirection byDirection, ByDomain byDomain) const -> int
{
    auto count = getMapper1Dto3D(setIdx, ByPartition::boundary, byDirection, byDomain).size();
    return static_cast<int>(count);
}

auto SpanClassifier::countBoundary(Neon::SetIdx setIdx) const -> int
{

    int count = 0;
    for (auto const& byDirection : {ByDirection::up, ByDirection::down}) {
        for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
            count += static_cast<int>(getMapper1Dto3D(setIdx, ByPartition::boundary, byDirection, byDomain).size());
        }
    }
    return static_cast<int>(count);
}

}  // namespace Neon::domain::tool::partitioning
