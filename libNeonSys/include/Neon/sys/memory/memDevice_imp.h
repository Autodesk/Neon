#pragma once

#include "Neon/core/core.h"
#include "Neon/sys/devices/gpu/GpuSys.h"
#include "Neon/sys/global/CpuSysGlobal.h"
#include "Neon/sys/global/GpuSysGlobal.h"
#include "Neon/sys/memory/GpuMem.h"

namespace Neon {
namespace sys {

/*
 * Empty constructor,m_counter set to null -> user managed.
 */
template <typename T_ta>
MemDevice<T_ta>::MemDevice()
    : m_devType(), m_devIdx(), m_alignment()
{
}

/*
 * Managed constructor,m_counter set to null -> user managed.
 */
template <typename T_ta>
MemDevice<T_ta>::MemDevice(DeviceType devType,
                           DeviceID   devIdx,
                           uint64_t   nElements,
                           T_ta*      buffer)
    : m_devType(devType),
      m_devIdx(devIdx),
      m_allocType(Neon::Allocator::MANAGED),
      m_nElements(nElements),
      m_allocatedBytes(nElements * sizeof(T_ta)),
      m_requiredBytes(nElements * sizeof(T_ta)),
      m_compute(buffer, {1, 0})
{
}

/*
 * Constructor where memory is allocated by the system, m_counter is not null -> sys managed.
 */
template <typename T_ta>
MemDevice<T_ta>::MemDevice(DeviceType      devType,
                           DeviceID        devIdx,
                           Neon::Allocator allocType,
                           uint64_t        nElements,
                           MemAlignment    alignment)
    : m_devType(devType),
      m_devIdx(devIdx),
      m_allocType(allocType),
      m_nElements(nElements),
      m_alignment(alignment)
{
    if (m_allocType == Neon::Allocator::NULL_MEM) {
        m_compute = Partition(nullptr, {0, 0});
        m_notAlignedBuffer = nullptr;
        // Setting m_counter to nullptr defines the Mem_t object as managed.
        m_refCounter = nullptr;
        return;
    }
    m_refCounter = new std::atomic_uint64_t(0);
    helperAllocMem();
}

template <typename T_ta>
MemDevice<T_ta>::MemDevice(DeviceType      devType,
                           DeviceID        devId,
                           Neon::Allocator allocType,
                           uint64_t        nElements,
                           memAlignment_et alignment)
    : MemDevice(devType, devId, allocType, nElements, MemAlignment(alignment))
{
    /**
     * Nothing to do
     * */
}

/**
 * Constructor (Sys managed)
 */
template <typename T_ta>
MemDevice<T_ta>::MemDevice(int                           cardinality,
                           Neon::memLayout_et::order_e   order,
                           Neon::memLayout_et::padding_e padding,
                           DeviceType                    devType,
                           DeviceID                      devId,
                           Neon::Allocator               allocType,
                           uint64_t                      nElements,
                           MemAlignment                  alignment)
    : m_devType(devType),
      m_devIdx(devId),
      m_allocType(allocType),
      m_nElements(nElements),
      m_cardinality(cardinality),
      m_order(order),
      m_padding(padding),
      m_alignment(alignment)
{
    if (m_allocType == Neon::Allocator::NULL_MEM) {
        m_compute = Partition(nullptr, {0, 0});
        m_notAlignedBuffer = nullptr;
        // Setting m_counter to nullptr defines the Mem_t object as managed.
        m_refCounter = nullptr;
        return;
    }
    m_refCounter = new std::atomic_uint64_t(0);
    helperAllocMem();
}

template <typename T_ta>
MemDevice<T_ta>::MemDevice(int                  cardinality,
                           Neon::sys::memConf_t memConf,
                           DeviceID             devId,
                           uint64_t             nElements)
    : m_devType(memConf.devEt()),
      m_devIdx(devId),
      m_allocType(memConf.allocEt()),
      m_nElements(nElements),
      m_cardinality(cardinality),
      m_order(memConf.order()),
      m_padding(memConf.padding()),
      m_alignment(memConf.alignment())
{
    if (m_allocType == Neon::Allocator::NULL_MEM) {
        m_compute = Partition(nullptr, {0, 0});
        m_notAlignedBuffer = nullptr;
        // Setting m_counter to nullptr defines the Mem_t object as managed.
        m_refCounter = nullptr;
        return;
    }
    m_refCounter = new std::atomic_uint64_t(0);
    helperAllocMem();
}


/*
 * Copy Constructor, what to do depends on the managedMode.
 */
template <typename T_ta>
MemDevice<T_ta>::MemDevice(const MemDevice& other)
    : m_devType(other.m_devType),
      m_devIdx(other.m_devIdx),
      m_allocType(other.m_allocType),
      m_nElements(other.m_nElements),
      m_cardinality(other.m_cardinality),
      m_order(other.m_order),
      m_padding(other.m_padding),
      m_alignment(other.m_alignment),
      m_allocatedBytes(other.m_allocatedBytes),
      m_requiredBytes(other.m_requiredBytes),
      m_notAlignedBuffer(other.m_notAlignedBuffer),
      m_refCounter(other.m_refCounter),
      m_compute(other.m_compute)
{
    /**
     * Everything is copy over... now it mater if it is system or used manged.
     */
    if (other.managedMode() == managedMode_t::system) {
        m_refCounter->fetch_add(1);
    }
}

/*
 * Move Constructor, what to do depends on the managedMode.
 */
template <typename T_ta>
MemDevice<T_ta>::MemDevice(MemDevice&& other)
    : m_devType(other.m_devType),
      m_devIdx(other.m_devIdx),
      m_allocType(other.m_allocType),
      m_nElements(other.m_nElements),
      m_cardinality(other.m_cardinality),
      m_order(other.m_order),
      m_padding(other.m_padding),
      m_alignment(other.m_alignment),
      m_allocatedBytes(other.m_allocatedBytes),
      m_requiredBytes(other.m_requiredBytes),
      m_notAlignedBuffer(other.m_notAlignedBuffer),
      m_refCounter(other.m_refCounter),
      m_compute(other.m_compute)
{
    /*
     * Everything is copy over... Same behaviour for system and user mode.
     */
    other.m_refCounter = nullptr;
    other.m_compute.m_buffer = nullptr;
    other.m_notAlignedBuffer = nullptr;
}


/*
 * Assignment operator, what to do depends on the managedMode but it's masked by release() and the copy constructor;
 */
template <typename T_ta>
MemDevice<T_ta>&
MemDevice<T_ta>::operator=(const MemDevice& other) noexcept
{
    if (this != &other) {
        release();
        new (this) MemDevice(other);
    }
    return *this;
}
/*
 * Move assignment operator, what to do depends on the managedMode but it's masked by release() and the move constructor;
 */
template <typename T_ta>
MemDevice<T_ta>&
MemDevice<T_ta>::operator=(MemDevice&& other) noexcept
{
    if (this != &other) {
        release();
        new (this) MemDevice(other);
    }
    return *this;
}


template <typename T_ta>
MemDevice<T_ta>::~MemDevice()
{
    this->release();
}

template <typename T_ta>
void MemDevice<T_ta>::helperResetLocal()
{
    m_compute = Partition(nullptr, {0, 0});
    m_refCounter = nullptr;
}

template <typename T_ta>
void MemDevice<T_ta>::helperResetGlobal()
{
    delete m_refCounter;
    helperFreeMem();
    helperResetLocal();
}

template <typename T_ta>
void MemDevice<T_ta>::release()
{
    if (m_compute.m_buffer == nullptr || this->managedMode() == managedMode_t::user) {
        // if m_buffer == nullptr -> local info has already been reset.
        helperResetLocal();
        return;
    }
    uint64_t count = m_refCounter->fetch_sub(1);
    if (1 == count) {
        helperResetGlobal();
    }
    helperResetLocal();
}

template <typename T_ta>
MemAlignment MemDevice<T_ta>::alignment() const
{
    return m_alignment;
}

template <typename T_ta>
Neon::memLayout_et::order_e MemDevice<T_ta>::order() const
{
    return m_order;
}
template <typename T_ta>
Neon::memLayout_et::padding_e MemDevice<T_ta>::padding() const
{
    return m_padding;
}

template <typename T_ta>
int MemDevice<T_ta>::cardinality() const
{
    return m_cardinality;
}

template <typename T_ta>
const index64_2d& MemDevice<T_ta>::pitch() const
{
    return m_compute.m_pitch;
}


template <typename T_ta>
void MemDevice<T_ta>::helperAllocMem()
{
    auto getAlignedAddress = [](void* addr, size_t alignment, size_t availableMem, size_t requiredSize) -> T_ta* {
        T_ta* alignedAddr = (T_ta*)std::align(alignment,
                                              requiredSize,
                                              addr,
                                              availableMem);


        if (alignedAddr == nullptr) {
            NeonException exp;
            exp << "Unable to satisfy required alignment"
                << "\n alignment = " << alignment
                << "\n requiredSize= " << requiredSize
                << "\n addr= " << addr
                << "\n availableMem= " << availableMem;
            NEON_THROW(exp);
        }
        return alignedAddr;
    };

    const uint32_t align_exp = m_alignment.expAlign(m_devType, m_devIdx);
    const uint32_t align_byte = m_alignment.exp2byte(align_exp);
    const uint32_t align_byEl = align_byte / sizeof(T_ta);
    // const uint32_t align_reminder = align_byte % sizeof(T_ta);
    const uint64_t offset_byte = align_byte - 1;

    auto getElementPadding = [&]() -> size_t {
        if (cardinality() == 1) {
            return 0;
        } else {
            switch (m_order) {
                case Neon::memLayout_et::order_e::structOfArrays: {
                    size_t tmpElPadding = 0;
                    if (align_byEl == 0) {
                        return 0;
                    }
                    tmpElPadding = (m_nElements) % align_byEl;
                    tmpElPadding = tmpElPadding == 0 ? tmpElPadding : align_byEl - tmpElPadding;
                    return tmpElPadding;
                }
                case Neon::memLayout_et::order_e::arrayOfStructs: {
                    return 0;
                }
                default: {
                    Neon::NeonException exp("getElementPadding");
                    NEON_THROW(exp);
                }
            }
        }
    };
    auto getAllocatedMemorySizeTotal = [&](size_t elPadding) {
        if (cardinality() == 1) {
            size_t byteSize = m_nElements * sizeof(T_ta) + offset_byte;
            return byteSize;
        } else {
            size_t allocatedElementsForOneCardinality = m_nElements + elPadding;
            size_t byteSize = m_cardinality *
                                  allocatedElementsForOneCardinality *
                                  sizeof(T_ta) +
                              offset_byte;
            return byteSize;
        }
    };
    const size_t elPadding = getElementPadding();
    const size_t allocatedMemorySizeTotal = getAllocatedMemorySizeTotal(elPadding);
    const size_t requiredMemorySizeTotal = allocatedMemorySizeTotal - offset_byte;
    m_allocatedBytes = allocatedMemorySizeTotal;
    m_requiredBytes = requiredMemorySizeTotal;

    T_ta* computeBuffer = nullptr;
    switch (m_devType) {
        case Neon::DeviceType::CUDA: {
            Neon::sys::GpuMem& mem = Neon::sys::globalSpace::gpuSysObj().allocator(m_devIdx.idx());
            m_notAlignedBuffer = mem.allocateMem(m_allocType, allocatedMemorySizeTotal);
            computeBuffer = getAlignedAddress((void*)m_notAlignedBuffer, align_byte, allocatedMemorySizeTotal, requiredMemorySizeTotal);
            m_refCounter->fetch_add(1);
            break;
        }

        case Neon::DeviceType::CPU: {
            Neon::sys::CpuMem& mem = Neon::sys::globalSpace::cpuSysObj().allocator();
            m_notAlignedBuffer = mem.allocateMem(m_allocType, allocatedMemorySizeTotal);
            computeBuffer = getAlignedAddress((void*)m_notAlignedBuffer, align_byte, allocatedMemorySizeTotal, requiredMemorySizeTotal);
            m_refCounter->fetch_add(1);
            break;
        }

        default: {
            Neon::NeonException exp("Mem_t");
            exp << "Unsupported memory allocation for device " << m_devType;
            NEON_THROW(exp);
        }
    }
    Neon::index64_2d computePitch;
    switch (m_order) {
        case Neon::memLayout_et::order_e::structOfArrays: {
            computePitch.pMain = 1;
            computePitch.pCardinality = m_nElements + elPadding;

            break;
        }
        case Neon::memLayout_et::order_e::arrayOfStructs: {
            computePitch.pMain = cardinality();
            computePitch.pCardinality = 1;

            break;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION();
        }
    }

    m_compute = Partition(computeBuffer, computePitch);

    return;
}

template <typename T_ta>
void MemDevice<T_ta>::helperFreeMem()
{
    switch (m_devType) {
        case Neon::DeviceType::CUDA: {

            Neon::sys::GpuMem& mem = Neon::sys::globalSpace::gpuSysObj().allocator(m_devIdx.idx());
            mem.releaseMem(m_allocType, m_allocatedBytes, m_notAlignedBuffer);
            break;
        }
        case Neon::DeviceType::CPU: {

            Neon::sys::CpuMem& mem = Neon::sys::globalSpace::cpuSysObj().allocator();
            mem.releaseMem(m_allocType, m_allocatedBytes, m_notAlignedBuffer);
            break;
        }
        default: {
            Neon::NeonException exp("Mem_t");
            exp << "Unsupported memory de-allocation for device " << m_devType;
            NEON_THROW(exp);
        }
    }
    m_compute = Partition(nullptr, {0, 0});
}

template <typename T_ta>
managedMode_t
MemDevice<T_ta>::managedMode() const
{
    if (m_refCounter == nullptr) {
        return managedMode_t(managedMode_t::user);
    }
    return managedMode_t(managedMode_t::system);
}

template <typename T_ta>
const DeviceType&
MemDevice<T_ta>::devType() const
{
    return m_devType;
}

template <typename T_ta>
const DeviceID&
MemDevice<T_ta>::devIdx() const
{
    return m_devIdx;
}

template <typename T_ta>
const Neon::Allocator&
MemDevice<T_ta>::allocType() const
{
    return m_allocType;
}

template <typename T_ta>
uint64_t
MemDevice<T_ta>::nBytes() const
{
    return m_nElements * sizeof(T_ta);
}

template <typename T_ta>
void MemDevice<T_ta>::copyFrom(const MemDevice<T_ta>& mem)
{
    if (m_nElements != mem.m_nElements || m_requiredBytes != mem.m_requiredBytes) {
        NeonException exception("copyFrom");
        exception << "Two Mem_t must have the same size.";
        NEON_THROW(exception);
    }
    if (m_devType == mem.m_devType) {
        if (m_devType == Neon::DeviceType::CPU) {
            std::memcpy(m_compute.m_buffer, mem.m_compute.m_buffer, m_requiredBytes);
            return;
        }
        if (m_devType == Neon::DeviceType::CUDA) {
            if (m_devIdx == mem.devIdx()) {
                const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
                auto        gpuStream = gpuDev.tools.stream();
                const T_ta* otherBuffer = mem.m_compute.m_buffer;
                gpuDev.memory.template transfer<char, mem_et::gpu,
                                                mem_et::gpu,
                                                run_et::sync>(gpuStream, (char*)m_compute.m_buffer, (const char*)otherBuffer, m_requiredBytes);
                gpuStream.sync();
                return;
            } else {
                const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
                auto        gpuStream = gpuDev.tools.stream();
                gpuDev.memory.peerTransfer(gpuStream,
                                           m_devIdx,
                                           (char*)m_compute.m_buffer,
                                           mem.m_devIdx,
                                           (char*)mem.m_compute.m_buffer,
                                           m_requiredBytes);
                gpuStream.sync();
                return;
            }
        }
    }

    if (m_devType == Neon::DeviceType::CUDA && mem.m_devType == Neon::DeviceType::CPU) {
        const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
        auto        gpuStream = gpuDev.tools.stream();
        gpuDev.memory.template transfer<char, mem_et::gpu,
                                        mem_et::cpu,
                                        run_et::sync>(gpuStream, (char*)m_compute.m_buffer, (char*)mem.m_compute.m_buffer, m_requiredBytes);
        gpuStream.sync();
        return;
    }

    if (m_devType == Neon::DeviceType::CPU && mem.m_devType == Neon::DeviceType::CUDA) {
        const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(mem.m_devIdx);
        auto        gpuStream = gpuDev.tools.stream();
        gpuDev.memory.template transfer<char,
                                        mem_et::cpu,
                                        mem_et::gpu,
                                        run_et::sync>(gpuStream, (char*)m_compute.m_buffer, (char*)mem.m_compute.m_buffer, m_requiredBytes);
        gpuStream.sync();
        return;
    }

    NeonException exception("copyFrom");
    exception << "Configuration not supported";
    NEON_THROW(exception);
}


template <typename T_ta>
void MemDevice<T_ta>::copyTo(MemDevice& mem, Neon::sys::GpuStream& stream)
{
    if (m_nElements != mem.m_nElements || m_requiredBytes != mem.m_requiredBytes) {
        NeonException exception("copyTo");
        exception << "Two Mem_t must have the same size.";
        NEON_THROW(exception);
    }
    if (m_devType == mem.m_devType) {
        if (m_devType == Neon::DeviceType::CPU) {
            std::memcpy(mem.m_compute.m_buffer, m_compute.m_buffer, m_requiredBytes);
            return;
        }
        if (m_devType == Neon::DeviceType::CUDA) {
            if (m_devIdx == mem.devIdx()) {
                const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
                T_ta*       dest = mem.m_compute.m_buffer;
                gpuDev.memory.template transfer<char, mem_et::gpu,
                                                mem_et::gpu,
                                                run_et::async>(stream,
                                                               (char*)dest,
                                                               (char*)m_compute.m_buffer,
                                                               m_requiredBytes);
                return;
            } else {
                const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
                gpuDev.memory.peerTransfer(stream,
                                           mem.m_devIdx,
                                           (char*)mem.m_compute.m_buffer,
                                           m_devIdx,
                                           (char*)m_compute.m_buffer,
                                           m_requiredBytes);
                return;
            }
        }
    }

    if (m_devType == Neon::DeviceType::CPU && mem.m_devType == Neon::DeviceType::CUDA) {
        const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
        gpuDev.memory.template transfer<char,
                                        mem_et::gpu,
                                        mem_et::cpu,
                                        run_et::async>(stream,
                                                       (char*)mem.m_compute.m_buffer,
                                                       (char*)m_compute.m_buffer,
                                                       m_requiredBytes);
        return;
    }

    if (m_devType == Neon::DeviceType::CUDA && mem.m_devType == Neon::DeviceType::CPU) {
        const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(mem.m_devIdx);
        gpuDev.memory.template transfer<char,
                                        mem_et::cpu,
                                        mem_et::gpu,
                                        run_et::async>(stream,
                                                       (char*)mem.m_compute.m_buffer,
                                                       (char*)m_compute.m_buffer,
                                                       m_requiredBytes);
        return;
    }

    NeonException exception("copyTo");
    exception << "Configuration not supported";
    NEON_THROW(exception);
}


template <typename T_ta>
template <Neon::run_et::et run_eta>
void MemDevice<T_ta>::updateFrom(const Neon::sys::GpuStream& gpuStream, const MemDevice<T_ta>& mem)
{
    if (m_nElements != mem.m_nElements) {
        NeonException exception("updateFrom");
        exception << "Two Mem_t must have the same size.";
        NEON_THROW(exception);
    }
    if (m_devType == mem.m_devType) {
        if (m_devType == Neon::DeviceType::CPU) {
            std::memcpy((void*)m_compute.m_buffer, (void*)mem.m_compute.m_buffer, m_requiredBytes);
            return;
        }
        if (m_devType == Neon::DeviceType::CUDA) {
            if (m_devIdx == mem.devIdx()) {
                const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
                gpuDev.memory.template transfer<T_ta, mem_et::gpu,
                                                mem_et::gpu,
                                                run_eta>(gpuStream, m_compute.m_buffer, mem.m_compute.m_buffer, this->nElements());
                return;
            } else {
                const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
                gpuDev.memory.peerTransfer(gpuStream,
                                           m_devIdx,
                                           (char*)m_compute.m_buffer,
                                           mem.m_devIdx,
                                           (char*)mem.m_compute.m_buffer,
                                           m_requiredBytes);
                gpuStream.sync<run_eta>();
                return;
            }
        }
    }

    if (m_devType == Neon::DeviceType::CUDA && mem.m_devType == Neon::DeviceType::CPU) {
        const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
        gpuDev.memory.template transfer<char, mem_et::gpu,
                                        mem_et::cpu,
                                        run_eta>(gpuStream, (char*)m_compute.m_buffer, (char*)mem.m_compute.m_buffer, m_requiredBytes);
        return;
    }

    if (m_devType == Neon::DeviceType::CPU && mem.m_devType == Neon::DeviceType::CUDA) {
        const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(mem.m_devIdx);
        gpuDev.memory.template transfer<char,
                                        mem_et::cpu,
                                        mem_et::gpu,
                                        run_eta>(gpuStream, (char*)m_compute.m_buffer, (char*)mem.m_compute.m_buffer, m_requiredBytes);
        return;
    }

    NeonException exception("copyFrom");
    exception << "Configuration not supported";
    NEON_THROW(exception);
}

template <typename T_ta>
template <Neon::run_et::et runMode_ta>
void MemDevice<T_ta>::updateWindowFrom(const Neon::sys::GpuStream& gpuStream,
                                       const MemDevice<T_ta>&      mem,
                                       eIdx_t                      start,
                                       int                         cardinality,
                                       int64_t                     nEl)
{
    if (m_nElements != mem.m_nElements) {
        NeonException exception("updateFrom");
        exception << "Two Mem_t must have the same size.";
        NEON_THROW(exception);
    }

    element_t*       dest = &m_compute.eRef(start, cardinality);
    const element_t* src = &mem.m_compute.eRef(start, cardinality);
    const size_t     nBytes = nEl * sizeof(element_t);

    if (m_devType == mem.m_devType) {
        if (m_devType == Neon::DeviceType::CPU) {
            std::memcpy(dest, src, nBytes);
            return;
        }
        if (m_devType == Neon::DeviceType::CUDA) {
            if (m_devIdx == mem.devIdx()) {
                const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
                gpuDev.memory.template transfer<T_ta, mem_et::gpu,
                                                mem_et::gpu,
                                                runMode_ta>(gpuStream, dest, src, nEl);
                return;
            } else {
                const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
                gpuDev.memory.peerTransfer(gpuStream,
                                           m_devIdx,
                                           (char*)dest,
                                           mem.m_devIdx,
                                           (const char*)src,
                                           nBytes);
                gpuStream.sync<runMode_ta>();
                return;
            }
        }
    }

    if (m_devType == Neon::DeviceType::CUDA && mem.m_devType == Neon::DeviceType::CPU) {
        const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(m_devIdx);
        gpuDev.memory.template transfer<char, mem_et::gpu,
                                        mem_et::cpu,
                                        runMode_ta>(gpuStream, (char*)dest, (const char*)src, nBytes);
        return;
    }

    if (m_devType == Neon::DeviceType::CPU && mem.m_devType == Neon::DeviceType::CUDA) {
        const auto& gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(mem.m_devIdx);
        gpuDev.memory.template transfer<char,
                                        mem_et::cpu,
                                        mem_et::gpu,
                                        runMode_ta>(gpuStream, (char*)dest, (const char*)src, nBytes);
        return;
    }

    NeonException exception("updateWindowFrom");
    exception << "Configuration not supported";
    NEON_THROW(exception);
}

template <typename T_ta>
MemDevice<T_ta>
MemDevice<T_ta>::clone(DeviceType devType, DeviceID devIdx, Neon::Allocator allocType) const
{
    MemDevice clone(devType, devIdx, allocType, m_requiredBytes);
    clone.copyFrom(*this);
    return clone;
}

}  // namespace sys
}  // End of namespace Neon
