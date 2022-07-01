#pragma once

#include <atomic>
#include "Neon/sys/devices/cpu/CpuSys.h"

namespace Neon {
namespace sys {

template <typename T_ta>
struct MemDevice;

class CpuMem
{
    template <typename T_ta>
    friend struct Neon::sys::MemDevice;

   private:
    CpuDev* m_cpuDev{nullptr};

    std::atomic_size_t m_allocatedMemPinned{0};
    std::atomic_size_t m_allocatedMemPageable{0};

    std::atomic_size_t m_maxUsedMemPinned{0};
    std::atomic_size_t m_maxUsedMemPagable{0};

    void* allocateMem(const Neon::Allocator& allocType, size_t size);
    void  releaseMem(const Neon::Allocator& allocType, size_t size, void* mem);

    void updateMaxUsePinned(size_t usedNow);
    void updateMaxUsePageable(size_t usedNow);

   public:
    CpuMem() = default;

    explicit CpuMem(CpuDev& cpuDev);

    CpuMem(const CpuMem& other);
    CpuMem(CpuMem&&) noexcept;

    CpuMem& operator=(const CpuMem& other);
    CpuMem& operator=(CpuMem&& other) noexcept;

    /*[[nodiscard]]*/ size_t inUsedMemPinned() const;
    /*[[nodiscard]]*/ size_t inUsedMemPageable() const;
};


}  // namespace sys
}  // End of namespace Neon
