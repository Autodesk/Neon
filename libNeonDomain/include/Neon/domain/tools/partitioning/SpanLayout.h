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
        Neon::Backend const&     backend,
        SpanDecomposition const& spanPartitioner,
        SpanClassifier const&    spanClassifier);

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
            Bounds      mByDomain[2];
            GhostTarget ghost;

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
    SpanClassifier const*               mSpanClassifierPtr;
    SpanDecomposition const*            mSpanPartitioner;
    Neon::MemoryOptions                 mMemOptionsAoS;
    Neon::set::DataSet<int32_t>        mStandardAndGhostCount;
};


auto SpanLayout::allocateStencilRelativeIndexMap(
    const Backend&               backend,
    int                          stream,
    const Neon::domain::Stencil& stencil) const -> Neon::set::MemSet<int8_3d>
{

    auto stencilNghSize = backend.devSet().template newDataSet<uint64_t>(
        stencil.neighbours().size());

    Neon::set::MemSet<int8_3d> stencilNghIndex = backend.devSet().template newMemSet<int8_3d>(
        Neon::DataUse::HOST_DEVICE,
        1,
        mMemOptionsAoS,
        stencilNghSize);

    for (int32_t c = 0; c < stencilNghIndex.cardinality(); ++c) {
        SetIdx devID(c);
        for (int64_t s = 0; s < int64_t(stencil.neighbours().size()); ++s) {
            stencilNghIndex.eRef(c, s).x = static_cast<int8_3d::Integer>(stencil.neighbours()[s].x);
            stencilNghIndex.eRef(c, s).y = static_cast<int8_3d::Integer>(stencil.neighbours()[s].y);
            stencilNghIndex.eRef(c, s).z = static_cast<int8_3d::Integer>(stencil.neighbours()[s].z);
        }
    }

    stencilNghIndex.updateDeviceData(backend, stream);
    return stencilNghIndex;
}



}  // namespace Neon::domain::tool::partitioning
