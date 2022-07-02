#include "Neon/set/DevSet.h"
#pragma once

namespace Neon {
namespace set {

/**
 * Providing a Left-Right barrier over events
 * The barrier is compose by two phases: begin and end
 * In the begin phase events are enqueue in the target stream set
 * In the end phase streams wait for the enqueue event
 */
struct Event_LR_barrier
{
   private:
    /**
     * Phase 1 of the barrier, enqueuing events into streams
     * @param devSet
     * @param s
     * @param e
     */
    static auto h_begin(const Neon::set::DevSet& devSet,
                        Neon::set::StreamSet&    s,
                        Neon::set::GpuEventSet&  e) -> void;

    /**
     * Phase 2 of the barrier, waiting for events
     * @param devSet
     * @param s
     * @param e
     */
    static auto h_end(const Neon::set::DevSet& devSet,
                      Neon::set::StreamSet&    s,
                      Neon::set::GpuEventSet&  e) -> void;

   public:
    /**
     * Execute a left and right sync over streams and events
     * @param devSet
     * @param s
     * @param e
     */
    static auto sync(const Neon::set::DevSet& devSet,
                     Neon::set::StreamSet&    s,
                     Neon::set::GpuEventSet&  e) -> void;

    static auto sync(Neon::Backend&   bk,
                     Neon::StreamIdx& s,
                     Neon::EventIdx&  e) -> void;
};

}  // namespace set
}  // namespace Neon