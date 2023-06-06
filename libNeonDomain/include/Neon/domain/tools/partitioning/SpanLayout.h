#pragma once
#include "Neon/core/core.h"
#include "Neon/domain/tools/partitioning/SpanClassifier.h"

namespace Neon::domain::tool::partitioning {

class SpanLayout
{
   public:
    struct Bounds
    {
        int first;
        int count;
    };

    struct GhostTarget
    {
        Neon::SetIdx setIdx;
        ByDirection  byDirection;
    };
    // -------------------------------------
    // | Internal  | BoundaryUP| BoundaryDW | Ghost UP     |  Ghost Dw    |
    // | Bulk | Bc | Bulk | Bc | Bulk | Bc  | Bulk | Bc    | Bulk | Bc    |
    // |           |           |            | Ghost setIdx | Ghost setIdx |
    // -------------------------------------
    SpanLayout() = default;

    SpanLayout(
        Neon::Backend const&               backend,
        std::shared_ptr<SpanDecomposition> spanPartitionerPtr,
        std::shared_ptr<SpanClassifier>    spanClassifierPtr);

    auto getCount()
        -> Neon::set::DataSet<uint64_t>;

    auto getBoundsInternal(
        SetIdx)
        const -> Bounds;

    auto getBoundsInternal(
        SetIdx,
        ByDomain)
        const -> Bounds;

    auto getBoundsBoundary(
        SetIdx,
        ByDirection)
        const -> Bounds;

    auto getBoundsBoundary(
        SetIdx,
        ByDirection,
        ByDomain)
        const -> Bounds;

    auto getGhostBoundary(
        SetIdx,
        ByDirection)
        const -> Bounds;

    auto getGhostBoundary(
        SetIdx,
        ByDirection,
        ByDomain)
        const -> Bounds;

    auto getGhostTarget(
        SetIdx,
        ByDirection)
        const -> GhostTarget;

    auto getLocalPointOffset(
        SetIdx                setIdx,
        Neon::int32_3d const& point)
        const -> std::pair<bool, int32_t>;

    auto findPossiblyLocalPointOffset(
        SetIdx          setIdx,
        const int32_3d& point)
        const -> std::tuple<bool, int32_t, ByPartition, ByDirection, ByDomain>;

    auto findNeighbourOfInternalPoint(
        SetIdx                setIdx,
        const Neon::int32_3d& point,
        const Neon::int32_3d& offset)
        const -> std::pair<bool, int32_t>;

    auto findNeighbourOfBoundaryPoint(
        SetIdx                setIdx,
        const Neon::int32_3d& point,
        const Neon::int32_3d& nghOffset)
        const -> std::pair<bool, int32_t>;

    template <typename Field>
    auto computeBlockOrigins(
        NEON_OUT Field& field,
        int             stream)
        const -> void;

    auto allocateStencilRelativeIndexMap(
        const Backend&               backend,
        int                          stream,
        const Neon::domain::Stencil& stencil)
        const -> Neon::set::MemSet<int8_3d>;

    auto getStandardAndGhostCount() const
        -> const Neon::set::DataSet<int32_t>&;

    auto getStandardCount() const
        -> const Neon::set::DataSet<int32_t>&;

   private:
    /**
     * Returns the firs index of the selected partition of the partition logical span
     */
    auto getClassificationOffset(
        Neon::SetIdx,
        ByPartition,
        ByDirection,
        ByDomain)
        const -> int32_t;

    auto getTargetGhost(
        Neon::SetIdx setIdx,
        ByDirection  direction)
        -> GhostTarget
    {
        int         offset = direction == ByDirection::up ? 1 : -1;
        int         ngh = (setIdx.idx() + mCountXpu + offset) % mCountXpu;
        GhostTarget result;
        result.setIdx = ngh;
        result.byDirection = direction == ByDirection::up ? ByDirection::down : ByDirection::up;
        return result;
    }

    struct InfoByPartition
    {
        struct Info
        {


           private:
            std::array<Bounds, 2> mByDomain;
            GhostTarget           ghost;

           public:
            auto operator()(ByDomain byDomain)
                -> Bounds&
            {
                return mByDomain[static_cast<int>(byDomain)];
            }

            auto operator()(ByDomain byDomain) const
                -> Bounds const&
            {
                return mByDomain[static_cast<int>(byDomain)];
            }

            auto getGhost() const -> GhostTarget const&
            {
                return ghost;
            }

            auto getGhost() -> GhostTarget&
            {
                return ghost;
            }
        };

       public:
        auto getInternal()
            const -> Info const&
        {
            return internal;
        }

        auto getBoundary(ByDirection byDirection) const
            -> Info const&
        {
            return boundary[static_cast<int>(byDirection)];
        }

        auto getGhost(ByDirection byDirection)
            const -> Info const&
        {
            return ghost[static_cast<int>(byDirection)];
        }

        auto getInternal()
            -> Info&
        {
            return internal;
        }

        auto getBoundary(ByDirection byDirection)
            -> Info&
        {
            return boundary[static_cast<int>(byDirection)];
        }

        auto getGhost(ByDirection byDirection)
            -> Info&
        {
            return ghost[static_cast<int>(byDirection)];
        }

       private:
        Info internal;
        Info boundary[2];
        Info ghost[2];
    };

    Neon::set::DataSet<InfoByPartition> mDataByPartition;
    int                                 mCountXpu;
    std::shared_ptr<SpanClassifier>     mSpanClassifierPtr;
    std::shared_ptr<SpanDecomposition>  mSpanDecompositionPrt;
    Neon::MemoryOptions                 mMemOptionsAoS;
    Neon::set::DataSet<int32_t>         mStandardAndGhostCount;
    Neon::set::DataSet<int32_t>         mStandardCount;
};


}  // namespace Neon::domain::tool::partitioning
