#include "Neon/sys/devices/gpu/GpuEvent.h"
#include "Neon/sys/devices/gpu/GpuStream.h"

#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/ComputeID.h"
#include "Neon/sys/global/GpuSysGlobal.h"

#include <vector>

namespace Neon {
namespace sys {


GpuEvent::GpuEvent()
{
    this->helper_ResetLocal();
}

GpuEvent::GpuEvent(ComputeID gpuIdx, cudaEvent_t cudaEvent)
{
    m_gpuId = gpuIdx;
    m_cudaEvent = cudaEvent;
    m_referenceCounter = new std::atomic_size_t(1);
}

GpuEvent::GpuEvent(const GpuEvent& other)
{
    m_gpuId = other.m_gpuId;
    m_cudaEvent = other.m_cudaEvent;
    m_referenceCounter = other.m_referenceCounter;
    if (other.m_referenceCounter != nullptr) {
        m_referenceCounter->fetch_add(1);
    }
}

GpuEvent::GpuEvent(GpuEvent&& other)
{
    /**
     * With the move constructor we don't have to update the reference counter.
     * The new object will take the place of the old.
     */
    m_gpuId = other.m_gpuId;
    m_cudaEvent = other.m_cudaEvent;
    m_referenceCounter = other.m_referenceCounter;
    //    if(other.m_referenceCounter != nullptr) {
    //        m_referenceCounter->fetch_add(1);
    //    }
    other.helper_ResetLocal();
}

GpuEvent& GpuEvent::operator=(const GpuEvent& other) noexcept
{
    if (this != &other) {
        this->release();

        m_gpuId = other.m_gpuId;
        m_cudaEvent = other.m_cudaEvent;
        m_referenceCounter = other.m_referenceCounter;
        if (other.m_referenceCounter != nullptr) {
            m_referenceCounter->fetch_add(1);
        }
    }
    return *this;
}

GpuEvent& GpuEvent::operator=(GpuEvent&& other) noexcept
{
    if (this != &other) {
        release();

        m_gpuId = other.m_gpuId;
        m_cudaEvent = other.m_cudaEvent;
        m_referenceCounter = other.m_referenceCounter;

        other.helper_ResetLocal();
    }
    other.helper_ResetLocal();
    return *this;
}

GpuEvent::~GpuEvent()
{
    this->release();
}

void GpuEvent::helper_ResetLocal()
{
    m_gpuId = Neon::sys::DeviceID();
    m_gpuId.setInvalid();
    m_cudaEvent = nullptr;
    m_referenceCounter = nullptr;
}

void GpuEvent::helper_ResetGlobal()
{
    delete m_referenceCounter;
    Neon::sys::globalSpace::gpuSysObj().dev(m_gpuId).tools.eventDestroy(*this);
    helper_ResetLocal();
}


cudaEvent_t& GpuEvent::event()
{
    return m_cudaEvent;
}

const cudaEvent_t& GpuEvent::event() const
{
    return m_cudaEvent;
}

void GpuEvent::release()
{
    /*
     * 1. check if the reference counter is a valid pointer.
     * 2. remove one form the counter
     * 3. if counter is 1 (counter is the value before the sub operation) do a global reset
     * 4. do a local reset
     */
    if (this->m_referenceCounter != nullptr) {
        size_t count = m_referenceCounter->fetch_sub(1);
        if (1 == count) {
            helper_ResetGlobal();
        }
    }
    helper_ResetLocal();
}

const ComputeID& GpuEvent::gpuId() const
{
    return m_gpuId;
}

void GpuEvent::sync() const
{
    auto res = cudaEventSynchronize(m_cudaEvent);
    if (res != cudaSuccess) {
        NeonException exp("GpuEvent_t");
        exp << "Cuda error detected during or before cudaEventSynchronize.";
        NEON_THROW(exp);
    }
}

}  // End of namespace sys
}  // End of namespace Neon
