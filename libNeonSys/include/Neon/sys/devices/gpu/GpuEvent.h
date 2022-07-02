#pragma once

#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/ComputeID.h"

#include <atomic>
#include <vector>

//#include "cuda.h"
#include "cuda_runtime.h"

namespace Neon::sys {


/**
 * Forward declaration. Class declaration is at the end of the file
 */
struct GpuEventSet_t;

/**
 * Wrapper around cudaEvent_t.
 * GpuEvent_t provides also automatic garbage collation of all allocated resources.
 * A release method is also given to force the release at specific time before the end of the object life.
 */
struct GpuEvent
{
    friend class GpuDevice;

   private:
    cudaEvent_t m_cudaEvent{nullptr};
    ComputeID   m_gpuId;

    std::atomic_size_t* m_referenceCounter{nullptr}; /**< Reference counter */

   private:
    //--------------------------------------------------------------------------
    // PRIVATE INITIALIZATION
    //--------------------------------------------------------------------------

    GpuEvent(ComputeID gpuIdx, cudaEvent_t cudaEvent);

    void helper_ResetLocal();
    void helper_ResetGlobal();

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
    GpuEvent();

    /**
     * Copy constructor
     */
    GpuEvent(const GpuEvent& other);

    /**
     * Move constructor
     */
    GpuEvent(GpuEvent&& other);

    /**
     * Copy operator
     * @param other
     * @return
     */
    GpuEvent& operator=(const GpuEvent& other) noexcept;

    /**
     * Move operator
     * @param other
     * @return
     */
    GpuEvent& operator=(GpuEvent&& other) noexcept;

    /**
     * Destructor
     */
    virtual ~GpuEvent();

    //--------------------------------------------------------------------------
    // INSPECTION
    //--------------------------------------------------------------------------
    /**
     * Returns a reference to the managed cudaEven_t object.
     * @return
     */
    cudaEvent_t& event();

    /**
     * Returns a read only reference to the managed cudaEven_t object.
     * @return
     */
    /*[[nodiscard]]*/ const cudaEvent_t& event() const;

    /**
     * Returns the ID of the GPU that owns the event.
     * @return
     */
    /*[[nodiscard]]*/ const ComputeID& gpuId() const;

    //--------------------------------------------------------------------------
    // RESOURCE MANAGEMENT
    //--------------------------------------------------------------------------

    /**
     * Release all the resources associated with the object and
     * in particular the managed cudaEvent_t event object.
     */
    void release();

    //--------------------------------------------------------------------------
    // SYNCRHONIZATION
    //--------------------------------------------------------------------------

    /**
     * Blocks the calling CPU thread until the status of the event is set to "occurred"
     */
    void sync() const;
};


}  // End of namespace sys
