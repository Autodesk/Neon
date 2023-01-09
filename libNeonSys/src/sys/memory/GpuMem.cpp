#include "Neon/sys/memory/GpuMem.h"


namespace Neon {
namespace sys {

GpuMem::GpuMem(GpuDevice& gpuDev)
    : m_gpuDev(&gpuDev), m_allocatedMem(0), m_maxUsedMem(0){};

GpuMem& GpuMem::operator=(const GpuMem& other)
{
    m_gpuDev = other.m_gpuDev;
    m_maxUsedMem = other.m_maxUsedMem.load();
    m_allocatedMem = other.m_allocatedMem.load();
    return *this;
}

GpuMem::GpuMem(const GpuMem& other)
    : m_gpuDev(other.m_gpuDev), m_allocatedMem(0), m_maxUsedMem(0)
{
    m_maxUsedMem = other.m_maxUsedMem.load();
    m_allocatedMem = other.m_allocatedMem.load();
};

void GpuMem::updateMaxUse(size_t usedNow)
{
    bool done = false;

    while (!done) {
        size_t usedMax = m_maxUsedMem.load();
        if (usedMax >= usedNow) {
            return;
        } else {
            done = m_maxUsedMem.compare_exchange_strong(usedMax, usedNow);
        }
    }
}

void* GpuMem::allocateMem(const Neon::Allocator& allocType, size_t size)
{
    switch (allocType) {
        case Neon::Allocator::CUDA_MEM_DEVICE: {

            void* buffer = nullptr;
            buffer = m_gpuDev->memory.mallocByte(size);

            size_t usedNow = m_allocatedMem.fetch_add(size) + size;
            this->updateMaxUse(usedNow);

            return buffer;
        }

        default: {
            Neon::NeonException exp("GpuMem_t");
            exp << "Unsupported memory allocation mType " << AllocatorUtils::toString(allocType) << " for device " << m_gpuDev->getIdx();
            NEON_THROW(exp);
        }
    }
}


void GpuMem::releaseMem(const Neon::Allocator& allocType, size_t size, void* mem)
{
    switch (allocType) {
        case Neon::Allocator::CUDA_MEM_DEVICE:

            m_gpuDev->memory.free(mem);
            m_allocatedMem.fetch_sub(size);

            return;

        default: {
            Neon::NeonException exp("GpuMem_t");
            exp << "Unsupported memory de-allocation mType " << allocType << " for device " << m_gpuDev->getIdx();
            NEON_THROW(exp);
        }
    }
}

size_t GpuMem::inUse() const
{
    return m_allocatedMem.load();
}

size_t GpuMem::maxUse() const
{
    return m_maxUsedMem.load();
}

size_t GpuMem::expL1DataCacheLine() const
{
    /**
     * Reference : https://arxiv.org/pdf/1804.06826.pdf
     * Dissecting theNVIDIA VoltaGPU Architecturevia Microbenchmarking
     */
    auto arch = m_gpuDev->tools.arch();
    int  res = 0;
    switch (arch) {
        case decltype(arch)::OldGen: {
            res = 0;
            break;
        }
        case decltype(arch)::Kepler: {
            res = 7;  // 128 byte
            break;
        }
        case decltype(arch)::NextGen:
        case decltype(arch)::Turing:
        case decltype(arch)::Volta:
        case decltype(arch)::Pascal:
        case decltype(arch)::Ampere:
        case decltype(arch)::Maxwell: {
            res = 5;  // 32 byte
            break;
        }
    }
    return res;
}

size_t GpuMem::expL2CacheLine() const
{
    /**
     * Reference : https://arxiv.org/pdf/1804.06826.pdf
     * Dissecting theNVIDIA VoltaGPU Architecturevia Microbenchmarking
     */

    auto arch = m_gpuDev->tools.arch();
    int  res = 0;
    switch (arch) {
        case decltype(arch)::OldGen: {
            res = 0;
            break;
        }
        case decltype(arch)::Pascal:
        case decltype(arch)::Maxwell:
        case decltype(arch)::Kepler: {
            res = 5;  // 32 byte
            break;
        }
        case decltype(arch)::NextGen:
        case decltype(arch)::Turing:
        case decltype(arch)::Ampere:
        case decltype(arch)::Volta: {
            res = 6;  // 64 byte
            break;
        }
    }
    return res;
}

size_t GpuMem::nByteL1DataCacheLine() const
{
    size_t lineExp = expL1DataCacheLine();
    size_t lineByte = (static_cast<size_t>(1) << lineExp);
    return lineByte;
}

size_t GpuMem::nByteL2CacheLine() const
{
    size_t lineExp = expL2CacheLine();
    size_t lineByte = (static_cast<size_t>(1) << lineExp);
    return lineByte;
}

size_t GpuMem::expL1_TLB_PageSize() const
{
    /**
     * Reference : https://arxiv.org/pdf/1804.06826.pdf
     * Dissecting theNVIDIA VoltaGPU Architecturevia Microbenchmarking
     */
    auto arch = m_gpuDev->tools.arch();
    int  res = 0;
    switch (arch) {
        case decltype(arch)::OldGen: {
            res = 0;
            break;
        }
        case decltype(arch)::Maxwell:
        case decltype(arch)::Kepler: {
            res = 17;  // 128 KiB
            break;
        }
        case decltype(arch)::NextGen:
        case decltype(arch)::Turing:
        case decltype(arch)::Volta:
        case decltype(arch)::Ampere:
        case decltype(arch)::Pascal: {
            res = 22;  // 2 MiB
            break;
        }
    }
    return res;
}


size_t GpuMem::expL2_TLB_PageSize() const
{
    /**
     * Reference : https://arxiv.org/pdf/1804.06826.pdf
     * Dissecting theNVIDIA VoltaGPU Architecturevia Microbenchmarking
     */
    auto arch = m_gpuDev->tools.arch();
    int  res = 0;
    switch (arch) {
        case decltype(arch)::OldGen: {
            res = 0;
            break;
        }
        case decltype(arch)::Maxwell:
        case decltype(arch)::Kepler: {
            res = 22;  // 2 MiB
            break;
        }
        case decltype(arch)::NextGen:
        case decltype(arch)::Turing:
        case decltype(arch)::Volta:
        case decltype(arch)::Ampere:
        case decltype(arch)::Pascal: {
            res = 25;  // 32 MiB
            break;
        }
    }
    return res;
}
size_t GpuMem::nByte_L1_TLB_PageSize() const
{
    size_t pageExp = expL1_TLB_PageSize();
    size_t pageByte = (static_cast<size_t>(1) << pageExp);
    return pageByte;
}

size_t GpuMem::nByte_L2_TLB_PageSize() const
{
    size_t pageExp = expL2_TLB_PageSize();
    size_t pageByte = (static_cast<size_t>(1) << pageExp);
    return pageByte;
}
}  // namespace sys
}  // End of namespace Neon
