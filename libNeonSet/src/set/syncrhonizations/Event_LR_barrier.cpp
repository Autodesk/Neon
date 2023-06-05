#include "Neon/set/syncrhonizations/event_LR_barrier.h"

namespace Neon {
namespace set {

auto Event_LR_barrier::h_begin(const Neon::set::DevSet& devSet,
                               Neon::set::StreamSet&    s,
                               Neon::set::GpuEventSet&  e) -> void
{
    const int card = devSet.setCardinality();
    if (card == 1) {
        // Nothing to do when cardinality is 1
        return;
    }
    s.enqueueEvent(e);
}

auto Event_LR_barrier::h_end(const Neon::set::DevSet& devSet,
                             Neon::set::StreamSet&    s,
                             Neon::set::GpuEventSet&  e) -> void
{
    const int card = devSet.setCardinality();
    if (card == 1) {
        // Nothing to do when cardinality is 1
        return;
        ;
    }
    if (card == 2) {
        devSet.forEachSetIdxPar([&](const Neon::SetIdx& selfIdx) {
            Neon::SetIdx  nextIdx((selfIdx.idx() + card + 1) % card);
            const auto&   targetStream = s.get(selfIdx.idx());
            const auto&   targetEvent = e.event(nextIdx.idx());
            targetStream.waitForEvent(targetEvent);
        });
        return;
    }
    devSet.forEachSetIdxPar([&](const Neon::SetIdx& selfIdx) {
        Neon::SetIdx  leftIdx((selfIdx.idx() + (card - 1)) % card);
        Neon::SetIdx  rightIdx((selfIdx.idx() + (card + 1)) % card);
        const auto&   targetStream = s.get(selfIdx.idx());
        const auto&   targetLeftEvent = e.event(leftIdx.idx());
        const auto&   targetRightEvent = e.event(rightIdx.idx());
        targetStream.waitForEvent(targetLeftEvent);
        targetStream.waitForEvent(targetRightEvent);
    });
    return;
}

auto Event_LR_barrier::sync(const Neon::set::DevSet& devSet,
                            Neon::set::StreamSet&    s,
                            Neon::set::GpuEventSet&  e) -> void
{
    if (devSet.type() == Neon::DeviceType::CPU) {
        return;
    }
    h_begin(devSet, s, e);
    h_end(devSet, s, e);
}

auto Event_LR_barrier::sync(Neon::Backend&   bk,
                            Neon::StreamIdx& s,
                            Neon::EventIdx&  e) -> void
{
    Neon::set::StreamSet&   streamSet = bk.streamSet(s);
    Neon::set::GpuEventSet& eventSet = bk.eventSet(e);
    Event_LR_barrier::sync(bk.devSet(), streamSet, eventSet);
}
}  // namespace set
}  // namespace Neon