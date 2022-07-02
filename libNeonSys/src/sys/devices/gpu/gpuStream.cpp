#include "Neon/sys/devices/gpu/GpuStream.h"
#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/ComputeID.h"
#include "Neon/sys/global/GpuSysGlobal.h"

#include <vector>

namespace Neon {
namespace sys {


GpuStream::GpuStream(ComputeID gpuIdx, cudaStream_t cudaStream)
{
    m_gpuId = gpuIdx;
    m_cudaStream = cudaStream;
    m_referenceCounter = new std::atomic_size_t(1);
}

//GpuStream_t::GpuStream_t(gpu_id gpuIdx, cudaStream_t cudaStream, GpuStreamSet* streamSet){
//    m_gpuId = gpuIdx;
//    m_cudaStream = cudaStream;
//    m_referenceCounter = nullptr;
//
//}

GpuStream::GpuStream(const GpuStream& other)
{
    m_gpuId = other.m_gpuId;
    m_cudaStream = other.m_cudaStream;
    m_referenceCounter = other.m_referenceCounter;
    if (other.m_referenceCounter != nullptr) {
        m_referenceCounter->fetch_add(1);
    }
}

GpuStream::GpuStream(GpuStream&& other)
{
    /**
     * With the move constructor we don't have to update the reference counter.
     * The new object will take the place of the old.
     */
    m_gpuId = other.m_gpuId;
    m_cudaStream = other.m_cudaStream;
    m_referenceCounter = other.m_referenceCounter;

    other.helper_ResetLocal();
}

GpuStream& GpuStream::operator=(const GpuStream& other) noexcept
{
    if (this != &other) {
        this->release();
        m_gpuId = other.m_gpuId;
        m_cudaStream = other.m_cudaStream;
        m_referenceCounter = other.m_referenceCounter;
        if (other.m_referenceCounter != nullptr) {
            m_referenceCounter->fetch_add(1);
        }
    }
    return *this;
}

GpuStream& GpuStream::operator=(GpuStream&& other) noexcept
{
    if (this != &other) {
        release();
        /**
         * With the move constructor we don't have to update the reference counter.
         * The new object will take the place of the old.
         */
        m_gpuId = other.m_gpuId;
        m_cudaStream = other.m_cudaStream;
        m_referenceCounter = other.m_referenceCounter;
    }
    other.helper_ResetLocal();
    return *this;
}

GpuStream::~GpuStream()
{
    this->release();
}

void GpuStream::helper_ResetLocal()
{
    m_gpuId = Neon::sys::DeviceID();
    m_gpuId.setInvalid();
    m_cudaStream = nullptr;
    m_referenceCounter = nullptr;
}

void GpuStream::helper_ResetGlobal()
{
    delete m_referenceCounter;
    m_referenceCounter = nullptr;
    Neon::sys::globalSpace::gpuSysObj().dev(m_gpuId).tools.streamDestroy(*this);
    helper_ResetLocal();
}


cudaStream_t& GpuStream::stream()
{
    return m_cudaStream;
}

const cudaStream_t& GpuStream::stream() const
{
    return m_cudaStream;
}

void GpuStream::release()
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

void GpuStream::enqueueEvent(GpuEvent& event) const
{
    Neon::sys::globalSpace::gpuSysObj().dev(m_gpuId).tools.setActiveDevContext();
    cudaEvent_t cudaEvent = event.event();
    cudaError_t retCode = cudaEventRecord(cudaEvent, m_cudaStream);
    if (cudaSuccess != retCode) {
        NeonException exc("GpuStream_t");
        exc << "CUDA error completing cudaEventRecord operation.\n";
        exc << "   target device: " << m_gpuId;
        exc << "\n   Error: " << cudaGetErrorString(retCode);
        NEON_THROW(exc);
    }

    return;
}

void GpuStream::waitForEvent(const GpuEvent& event) const
{
    Neon::sys::globalSpace::gpuSysObj().dev(m_gpuId).tools.setActiveDevContext();
    cudaEvent_t cudaEvent = event.event();
    cudaError_t retCode = cudaStreamWaitEvent(m_cudaStream, cudaEvent, 0);
    if (cudaSuccess != retCode) {
        NeonException exc("GpuStream_t");
        exc << "CUDA error completing cudaStreamWaitEvent operation.\n";
        exc << "   target device: " << m_gpuId;
        exc << "\n   Error: " << cudaGetErrorString(retCode);
        NEON_THROW(exc);
    }
    return;
}

const ComputeID& GpuStream::gpuId() const
{
    return m_gpuId;
}


}  // End of namespace sys
}  // End of namespace Neon
