#pragma once

#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/GpuEvent.h"

#include <atomic>
#include <vector>

namespace Neon {
namespace set {

/**
 * GpuEventSet_t stores an ordered set of GpuEvent_t objects.
 * In other words GpuEventSet_t creates a vectorized interface for GpuEvent_t objects.
 * The vectorized interface extend the capabilities of working with multi GPUs
 */
class GpuEventSet
{
   private:
    std::vector<Neon::sys::GpuEvent> eventVec;

   public:
    enum timingOption_e : bool
    {
        disableTiming = true,
        enableTiming = false
    };

    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------

    /**
     * Empty constructor
     */
    GpuEventSet() = default;

    /**
     * Empty constructor
     */
    GpuEventSet(int setCardinality);

    //--------------------------------------------------------------------------
    // INSPECTION
    //--------------------------------------------------------------------------

    /**
     * Method used to access in read only mode a specific GpuEvent_t object stored in the ordered set defined by this object.
     * @tparam accessType
     * @param id
     * @return
     */
    template <Neon::Access accessType = Neon::Access::read>
    inline typename std::enable_if<metaProgramming::isReadOnly_type_t<accessType>::value, const Neon::sys::GpuEvent&>::type event(SetIdx id)
    {
        if (id.idx() > int32_t(eventVec.size()) || eventVec.empty()) {
            if (eventVec.empty()) {
                Neon::NeonException exp("GpuEventSet_t");
                exp << "Incompatible stream id " << id << ". This stream set was not initialized";
                NEON_THROW(exp);
            } else {
                Neon::NeonException exp("GpuEventSet_t");
                exp << "Incompatible stream id " << id << ". Id range is: 0," << eventVec.size();
                NEON_THROW(exp);
            }
        }
        return eventVec[id.idx()];
    }

    /**
     * Method used to access in read and write mode a specific GpuEvent_t object stored in the ordered set defined by this object.
     * @tparam accessType
     * @param id
     * @return
     */
    template <Neon::Access accessType = Neon::Access::read>
    inline typename std::enable_if<metaProgramming::isreadAndWrite_type_t<accessType>::value, Neon::sys::GpuEvent&>::type event(SetIdx id)
    {
        if (id.idx() > int32_t(eventVec.size()) || eventVec.empty()) {
            if (eventVec.empty()) {
                Neon::NeonException exp("GpuEventSet_t");
                exp << "Incompatible stream id " << id << ". This stream set was not initialized";
                NEON_THROW(exp);
            } else {
                Neon::NeonException exp("GpuEventSet_t");
                exp << "Incompatible stream id " << id << ". Id range is: 0," << eventVec.size();
                NEON_THROW(exp);
            }
        }
        return eventVec[id.idx()];
    }

    /**
     * Returns the cudaEvent_t object associated with the i-th GpuEvent_t stored by this object.
     * @param idx
     * @return
     */
    cudaEvent_t& cudaEvent(int idx)
    {
        return eventVec[idx].event();
    }

    /**
     * Returns the cudaEvent_t object associated with the i-th GpuEvent_t stored by this object.
     * @param idx
     * @return
     */
    const cudaEvent_t& cudaEvent(int idx) const
    {
        return eventVec[idx].event();
    }

    //--------------------------------------------------------------------------
    // RESOURCE MANAGEMENT
    //--------------------------------------------------------------------------

    /**
     * It releases all the resources acquired by any of the GpuEvent_t objects stored by this object.
     */
    void release();

    //--------------------------------------------------------------------------
    // SYNCRHONIZATION
    //--------------------------------------------------------------------------

    /**
     * The method blocks the calling CPU thread until all the GpuEvent_t are associated with a "occurred" status/
     */
    void sync() const
    {
        for (auto&& cudaEvent : eventVec) {
            cudaEvent.sync();
        }
    }

    /**
     * The method blocks the calling CPU thread until all the i-th GpuEvent_t is associated with a "occurred" status/
     */
    void sync(int idx) const
    {
        return eventVec[idx].sync();
    }
};

}  // namespace set
}  // End of namespace Neon
