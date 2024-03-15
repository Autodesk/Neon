#include "Neon/domain/tools/partitioning/SpanLayout.h"
#include "Neon/domain/tools/partitioning/SpanDecomposition.h"


namespace Neon::domain::tool::partitioning {

SpanLayout::SpanLayout(Neon::Backend const&               backend,
                       std::shared_ptr<SpanDecomposition> spanPartitionerPtr,
                       std::shared_ptr<SpanClassifier>    spanClassifierPtr)
{

    mSpanDecompositionPrt = spanPartitionerPtr;
    mSpanClassifierPtr = spanClassifierPtr;

    mCountXpu = backend.devSet().setCardinality();
    mDataByPartition = backend.devSet().newDataSet<InfoByPartition>();
    // Setting up internal and boudary indexes
    auto lastFreeIndex = backend.devSet().newDataSet<int>();

    mDataByPartition.forEachSeq([&](Neon::SetIdx const& setIdx,
                                    InfoByPartition&    data) {
        int counter = 0;
        for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
            int toAdd = mSpanClassifierPtr->countInternal(setIdx, byDomain);
            data.getInternal()(byDomain).first = counter;
            data.getInternal()(byDomain).count = toAdd;
            counter += toAdd;
        }

        for (auto const& byDirection : {ByDirection::up, ByDirection::down}) {
            for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
                int   toAdd = mSpanClassifierPtr->countBoundary(setIdx, byDirection, byDomain);
                auto& info = data.getBoundary(byDirection)(byDomain);
                info.first = counter;
                info.count = toAdd;
                counter += toAdd;
            }
        }
        lastFreeIndex[setIdx] = counter;
    });


    mDataByPartition.forEachSeq([&](Neon::SetIdx const& setIdx,
                                    InfoByPartition&    data) -> void {
        for (auto const& byDirection : {ByDirection::up, ByDirection::down}) {
            //            for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {

            auto const& neighbourMirror = getTargetGhost(setIdx, byDirection);
            data.getGhost(byDirection).getGhost() = neighbourMirror;


            auto& mirrorData = mDataByPartition[neighbourMirror.setIdx];
            auto& mirrorInfoBulk = mirrorData.getBoundary(neighbourMirror.byDirection)(ByDomain::bulk);
            auto& mirrorInfoBc = mirrorData.getBoundary(neighbourMirror.byDirection)(ByDomain::bc);

            auto& infoBulk = data.getGhost(byDirection)(ByDomain::bulk);
            auto& infoBc = data.getGhost(byDirection)(ByDomain::bc);

            infoBulk.count = mirrorInfoBulk.count;
            infoBc.count = mirrorInfoBc.count;

            infoBulk.first = lastFreeIndex[setIdx];
            infoBc.first = lastFreeIndex[setIdx] + infoBulk.count;

            lastFreeIndex[setIdx] += (infoBulk.count + infoBc.count);
        }
    });

    mStandardAndGhostCount = backend.newDataSet<int32_t>();
    mStandardCount = backend.newDataSet<int32_t>();

    mStandardAndGhostCount.forEachSeq([&](const Neon::SetIdx& setIdx,
                                          int32_t&            standardAndGhostCount) {
        standardAndGhostCount = 0;
        auto& standardCount = mStandardCount[setIdx];
        standardCount = 0;

        {
            const auto internalBounds = getBoundsInternal(setIdx);
            standardAndGhostCount += internalBounds.count;
            standardCount += internalBounds.count;
        }
        {
            const auto boundaryUp = getBoundsBoundary(setIdx, partitioning::ByDirection::up);
            const auto boundaryDw = getBoundsBoundary(setIdx, partitioning::ByDirection::down);
            standardAndGhostCount += boundaryUp.count + boundaryDw.count;
            standardCount += boundaryUp.count + boundaryDw.count;
        }
        {
            const auto ghostUp = getGhostBoundary(setIdx, partitioning::ByDirection::up);
            const auto ghostDw = getGhostBoundary(setIdx, partitioning::ByDirection::down);
            standardAndGhostCount += ghostUp.count + ghostDw.count;
        }
    });
}

auto SpanLayout::getBoundsInternal(
    SetIdx setIdx)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getInternal().operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getInternal().operator()(byDomain).count;
    }
    return result;
}

auto SpanLayout::getBoundsInternal(
    SetIdx   setIdx,
    ByDomain byDomain)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getInternal().operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getInternal().operator()(byDomain).count;
    return result;
}

auto SpanLayout::getBoundsBoundary(
    SetIdx      setIdx,
    ByDirection byDirection)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getBoundary(byDirection).operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).count;
    }
    return result;
}

auto SpanLayout::getBoundsBoundary(
    SetIdx      setIdx,
    ByDirection byDirection,
    ByDomain    byDomain)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getBoundary(byDirection).operator()(byDomain).count;
    return result;
}

auto SpanLayout::getGhostBoundary(SetIdx setIdx, ByDirection byDirection)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getGhost(byDirection).operator()(ByDomain::bulk).first;
    result.count = 0;
    for (auto const& byDomain : {ByDomain::bulk, ByDomain::bc}) {
        result.count += mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).count;
    }
    return result;
}

auto SpanLayout::getGhostBoundary(
    SetIdx      setIdx,
    ByDirection byDirection,
    ByDomain    byDomain)
    const -> SpanLayout::Bounds
{
    SpanLayout::Bounds result{};
    result.first = mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).first;
    result.count = mDataByPartition[setIdx].getGhost(byDirection).operator()(byDomain).count;
    return result;
}

auto SpanLayout::getGhostTarget(SetIdx setIdx, ByDirection byDirection)
    const -> SpanLayout::GhostTarget
{
    return mDataByPartition[setIdx].getGhost(byDirection).getGhost();
}

auto SpanLayout::getLocalPointOffset(
    SetIdx          setIdx,
    const int32_3d& point) const -> std::pair<bool, int32_t>
{
    auto findings = findPossiblyLocalPointOffset(setIdx, point);
    if (std::get<0>(findings)) {
        auto classificationOffset = getClassificationOffset(setIdx,
                                                            std::get<2>(findings),
                                                            std::get<3>(findings),
                                                            std::get<4>(findings));
        return {true, std::get<1>(findings) + classificationOffset};
    }
    return {false, -1};
}


auto SpanLayout::findPossiblyLocalPointOffset(
    SetIdx          setIdx,
    const int32_3d& point)
    const -> std::tuple<bool, int32_t, ByPartition, ByDirection, ByDomain>
{
    for (auto byPartition : {ByPartition::internal, ByPartition::boundary}) {
        for (auto byDirection : {ByDirection::up, ByDirection::down}) {
            if (byPartition == ByPartition::internal && byDirection == ByDirection::down) {
                continue;
            }
            for (auto byDomain : {ByDomain::bulk, ByDomain::bc}) {
                auto const& mapper = mSpanClassifierPtr->getMapper3Dto1D(setIdx,
                                                                         byPartition,
                                                                         byDirection,
                                                                         byDomain);
                auto const  infoPtr = mapper.getMetadata(point);
                if (infoPtr != nullptr) {
                    return {true, int32_t(*infoPtr), byPartition, byDirection, byDomain};
                }
            }
        }
    }
    return {false, -1, ByPartition::internal, ByDirection::up, ByDomain::bulk};
}

auto SpanLayout::getClassificationOffset(
    Neon::SetIdx setIdx,
    ByPartition  byPartition,
    ByDirection  byDirection,
    ByDomain     byDomain)
    const -> int32_t
{
    if (byPartition == ByPartition::internal) {
        return this->getBoundsInternal(setIdx, byDomain).first;
    }
    return this->getBoundsBoundary(setIdx, byDirection, byDomain).first;
}

auto SpanLayout::findNeighbourOfInternalPoint(
    SetIdx                setIdx,
    const Neon::int32_3d& point,
    const Neon::int32_3d& offset)
    const -> std::pair<bool, int32_t>
{
    // Neighbours of internal points can be internal or boundary
    auto nghPoint = point + offset;
    return getLocalPointOffset(setIdx, nghPoint);
}

auto SpanLayout::findNeighbourOfBoundaryPoint(
    SetIdx                setIdx,
    const Neon::int32_3d& point,
    const Neon::int32_3d& nghOffset)
    const -> std::pair<bool, int32_t>
{
    // Neighbours of internal points can be internal or boundary
    auto nghPoint = point + nghOffset;
    auto findings = findPossiblyLocalPointOffset(setIdx, nghPoint);
    if (std::get<0>(findings)) {
        auto classificationOffset = getClassificationOffset(setIdx,
                                                            std::get<2>(findings),
                                                            std::get<3>(findings),
                                                            std::get<4>(findings));
        return {true, std::get<1>(findings) + classificationOffset};
    }
    // We need to search on local partitions
    // We select the target partition based on the .z component of the offset
    int          partitionOffset = nghOffset.z > 0 ? +1 : -1;
    Neon::SetIdx nghSetIdx = (setIdx.idx() + mCountXpu + partitionOffset) % mCountXpu;

    findings = findPossiblyLocalPointOffset(nghSetIdx, nghPoint);
    if (std::get<0>(findings)) {
        // Ghost direction is the opposite w.r.t. the neighbour partition direction
        ByDirection ghostByDirection = nghOffset.z > 0
                                           ? ByDirection::up
                                           : ByDirection::down;
        ByDomain    ghostByDomain = std::get<4>(findings);

        Bounds ghostBounds = getGhostBoundary(setIdx,
                                              ghostByDirection,
                                              ghostByDomain);

        return {true, std::get<1>(findings) + ghostBounds.first};
    }
    return {false, -1};
}


auto SpanLayout::getStandardAndGhostCount() const -> const Neon::set::DataSet<int32_t>&
{
    return mStandardAndGhostCount;
}

auto SpanLayout::getStandardCount() const -> const Neon::set::DataSet<int32_t>&
{
    return mStandardCount;
}


}  // namespace Neon::domain::tool::partitioning
