#include "Neon/sys/memory/CpuMem.h"

namespace Neon {
namespace sys {

CpuMem::CpuMem(CpuDev& cpuDev)
    : m_cpuDev(&cpuDev),
      m_allocatedMemPinned(0),
      m_allocatedMemPageable(0),
      m_maxUsedMemPinned(0),
      m_maxUsedMemPagable(0){};

CpuMem::CpuMem(const CpuMem& other)
    : m_cpuDev(other.m_cpuDev),
      m_allocatedMemPinned(other.m_maxUsedMemPinned.load()),
      m_allocatedMemPageable(other.m_allocatedMemPageable.load()),
      m_maxUsedMemPinned(other.m_maxUsedMemPinned.load()),
      m_maxUsedMemPagable(other.m_maxUsedMemPagable.load()){};

CpuMem::CpuMem(CpuMem&& other) noexcept
    : m_cpuDev(other.m_cpuDev),
      m_allocatedMemPinned(other.m_maxUsedMemPinned.load()),
      m_allocatedMemPageable(other.m_allocatedMemPageable.load()),
      m_maxUsedMemPinned(other.m_maxUsedMemPinned.load()),
      m_maxUsedMemPagable(other.m_maxUsedMemPagable.load())
{
    other.m_cpuDev = nullptr;
}

CpuMem& CpuMem::operator=(const CpuMem& other)
{
    m_cpuDev = other.m_cpuDev;
    m_allocatedMemPinned = other.m_maxUsedMemPinned.load();
    m_allocatedMemPageable = other.m_allocatedMemPageable.load();
    m_maxUsedMemPinned = other.m_maxUsedMemPinned.load();
    m_maxUsedMemPagable = other.m_maxUsedMemPagable.load();
    return *this;
}
CpuMem& CpuMem::operator=(CpuMem&& other) noexcept
{
    m_cpuDev = other.m_cpuDev;
    m_allocatedMemPinned = other.m_maxUsedMemPinned.load();
    m_allocatedMemPageable = other.m_allocatedMemPageable.load();
    m_maxUsedMemPinned = other.m_maxUsedMemPinned.load();
    m_maxUsedMemPagable = other.m_maxUsedMemPagable.load();

    other.m_cpuDev = nullptr;
    return *this;
}


void CpuMem::updateMaxUsePinned(size_t usedNow)
{
    bool done = false;

    while (!done) {
        size_t usedMax = m_maxUsedMemPinned.load();
        if (usedMax >= usedNow) {
            return;
        } else {
            done = m_maxUsedMemPinned.compare_exchange_strong(usedMax, usedNow);
        }
    }
}

void CpuMem::updateMaxUsePageable(size_t usedNow)
{
    bool done = false;

    while (!done) {
        size_t usedMax = m_maxUsedMemPagable.load();
        if (usedMax >= usedNow) {
            return;
        } else {
            done = m_maxUsedMemPagable.compare_exchange_strong(usedMax, usedNow);
        }
    }
}

void* CpuMem::allocateMem(const Neon::Allocator& allocType, size_t size)
{
    switch (allocType) {
        case Neon::Allocator::MALLOC: {

            void* buffer = nullptr;
            buffer = CpuDev::memory_t::mallocByte(size);

            size_t usedNow = m_allocatedMemPageable.fetch_add(size) + size;
            this->updateMaxUsePageable(usedNow);

            return buffer;
        }
        case Neon::Allocator::CUDA_MEM_HOST: {

            void* buffer = nullptr;
            buffer = CpuDev::memory_t::mallocCudaHostByte(size);

            size_t usedNow = m_allocatedMemPinned.fetch_add(size) + size;
            this->updateMaxUsePinned(usedNow);

            return buffer;
        }
        case Neon::Allocator::CUDA_MEM_UNIFIED: {

            void* buffer = nullptr;
            buffer = CpuDev::memory_t::mallocCudaManagedByte(size);

            size_t usedNow = m_allocatedMemPinned.fetch_add(size) + size;
            this->updateMaxUsePinned(usedNow);

            return buffer;
        }
        default: {
            Neon::NeonException exp("CpuMem_t");
            exp << "Unsupported->memory.allocation mType " << allocType << " for device " << m_cpuDev->getIdx();
            NEON_THROW(exp);
        }
    }
}


void CpuMem::releaseMem(const Neon::Allocator& allocType, size_t size, void* mem)
{
    switch (allocType) {
        case Neon::Allocator::MALLOC: {

            CpuDev::memory_t::free(mem);
            m_allocatedMemPageable.fetch_sub(size);

            return;
        }
        case Neon::Allocator::CUDA_MEM_HOST: {

            CpuDev::memory_t::freeCudaHostByte(mem);
            m_allocatedMemPinned.fetch_sub(size);

            return;
        }
        case Neon::Allocator::CUDA_MEM_UNIFIED: {

            CpuDev::memory_t::freeCudaManagedByte(mem);
            m_allocatedMemPinned.fetch_sub(size);

            return;
        }
        default: {
            Neon::NeonException exp("CpuMem_t");
            exp << "Unsupported memory de-allocation mType " << allocType << " for device " << m_cpuDev->getIdx();
            NEON_THROW(exp);
        }
    }
}


size_t CpuMem::inUsedMemPinned() const
{
    return m_allocatedMemPinned.load();
}
size_t CpuMem::inUsedMemPageable() const
{
    return m_allocatedMemPageable.load();
}

}  // namespace sys
}  // End of namespace Neon
