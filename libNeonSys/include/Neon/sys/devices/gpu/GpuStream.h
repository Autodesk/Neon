#pragma once

#include <atomic>
#include <vector>
#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/GpuEvent.h"
#include "Neon/sys/devices/gpu/ComputeID.h"

namespace Neon {
namespace sys {

struct GpuStream
{
    friend class GpuDevice;

   private:
    cudaStream_t m_cudaStream{nullptr};
    ComputeID    m_gpuId;

    /**< Reference counter */
    std::atomic_size_t* m_referenceCounter{nullptr};

   private:
    //--------------------------------------------------------------------------
    // PRIVATE INITIALIZATION
    //--------------------------------------------------------------------------

    /**
     * Private constructor called by DevSet
     */
    GpuStream(ComputeID gpuIdx, cudaStream_t cudaStream);

    /**
     * Private helper function to clean out resources
     */
    void helper_ResetLocal();
    /**
     * Private helper function to clean out resources
     */
    void helper_ResetGlobal();

   public:
    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------

    /**
     * Empty constructor
     */
    GpuStream() = default;

    /**
     * Copy constructor
     * @param other
     */
    GpuStream(const GpuStream& other);

    /**
     * Move constructor
     * @param other
     */
    GpuStream(GpuStream&& other);

    /**
     * Copy operator
     * @param other
     */
    GpuStream& operator=(const GpuStream& other) noexcept;

    /**
     * Copy operator
     * @param other
     */
    GpuStream& operator=(GpuStream&& other) noexcept;

    /**
     * Destructor
     */
    virtual ~GpuStream();

    //--------------------------------------------------------------------------
    // INSPECTION
    //--------------------------------------------------------------------------

    /**
     * Returns a reference to the managed CUDA stream.
     * @return
     */
    cudaStream_t& stream();

    /**
     * Returns a const reference to the managed CUDA stream.
     * @return
     */
    const cudaStream_t& stream() const;

    /**
     * Returns the ID of the GPU that owns the stream.
     * @return
     */
    const ComputeID& gpuId() const;

    //--------------------------------------------------------------------------
    // RESOURCE MANAGEMENT
    //--------------------------------------------------------------------------

    /**
     * Release all resources allocated by the object.
     * It basically destroy the cudaStream_t object that has been initialized.
     */
    void release();

    //--------------------------------------------------------------------------
    // SYNCRHONIZATION
    //--------------------------------------------------------------------------
    /**
     * Enqueue an event into the targeted GpuStream_t object
     */
    void enqueueEvent(GpuEvent& event) const;

    /**
     * Blocks the current thread until the provided event has been computed.
     * The function has the same semantic as cudaStreamWaitEvent.
     * @param event
     */
    void waitForEvent(const GpuEvent& event) const;

    /**
     * It blocks the current cpu thread until all the even in the stream have been completed.
     * This method has the same semantic as cudaStreamSynchronize
     * @tparam runMode
     */
    template <run_et::et runMode = run_et::et::sync>
    void sync() const
    {
        if constexpr (run_et::et::sync == runMode) {
            cudaError_t retCode = cudaStreamSynchronize(m_cudaStream);
            if (cudaSuccess != retCode) {
                NeonException exc("GpuStream_t");
                exc << "CUDA error completing cudaStreamSynchronize operation.\n";
                exc << "   target device: " << m_gpuId;
                exc << "\n   Error: " << cudaGetErrorString(retCode);
                NEON_THROW(exc);
            }
            return;
        } else {
            return;
        }
    }
};


}  // End of namespace sys
}  // End of namespace Neon
