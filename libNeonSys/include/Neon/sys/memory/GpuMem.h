#pragma once

#include <atomic>
#include "Neon/core/types/Allocator.h"
#include "Neon/sys/devices/gpu/GpuSys.h"

namespace Neon {
namespace sys {


template <typename T_ta>
struct MemDevice;

class GpuMem
{

   private:
    template <typename T_ta>
    friend struct Neon::sys::MemDevice;

    const GpuDevice* m_gpuDev{nullptr};

    std::atomic_size_t m_allocatedMem{0};
    std::atomic_size_t m_maxUsedMem{0};

    void* allocateMem(const Neon::Allocator& allocType, size_t size);
    void  releaseMem(const Neon::Allocator& allocType, size_t size, void* mem);

    void updateMaxUse(size_t usedNow);

   public:
    explicit GpuMem(GpuDevice& gpuDev);
    GpuMem(const GpuMem& other);

    GpuMem() = default;

    // GpuMem_t(GpuMem_t &&) = delete;

    /**
     * We don't want a copy operator. One a GpuDevicehas been created it can only be moved.
     * We should delete this operator but if we do it, VS complains (clang, gcc, intel are ok).
     * So we leave the copy but we fire an exception if called.
     */
    GpuMem& operator=(const GpuMem&);

    /**
     * Returns how much memory is in use at the moment of the call
     * @return
     */
    size_t inUse() const;

    /**
     * Returns the max amount of memory that has been used until now
     * @return
     */
    size_t maxUse() const;


    size_t expL1DataCacheLine() const;
    size_t expL2CacheLine() const;
    size_t nByteL1DataCacheLine() const;
    size_t nByteL2CacheLine() const;

    size_t expL1_TLB_PageSize() const;
    size_t expL2_TLB_PageSize() const;
    size_t nByte_L1_TLB_PageSize() const;
    size_t nByte_L2_TLB_PageSize() const;
};


}  // namespace sys
}  // End of namespace Neon
