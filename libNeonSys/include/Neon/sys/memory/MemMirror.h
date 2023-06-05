#pragma once

#pragma once

#include <atomic>

#include "Neon/core/core.h"
#include "Neon/core/types/Execution.h"

#include "Neon/core/tools/io/exportVTI.h"
#include "Neon/core/types/Allocator.h"
#include "Neon/core/types/memOptions.h"
#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/devices/cpu/CpuDevice.h"
#include "Neon/sys/devices/gpu/GpuDevice.h"
#include "Neon/sys/memory/MemDevice.h"
namespace Neon::sys {

template <typename T_ta>
struct MemMirror
{
   private:
    MemDevice<T_ta> m_cpu;
    MemDevice<T_ta> m_gpu;

    bool m_noThrowForAZeroSize = {false};
    bool cpu_configured = {false};
    bool gpu_configured = {false};
    bool isACPUdevice = {false};

   public:
    using Self = MemMirror<T_ta>;
    using Device = MemDevice<T_ta>;
    using Partition = typename Device::Partition;
    using Element = typename Partition::element_t;
    using eIdx_t = typename Partition::eIdx_t;
    /**
     * Default empty constructor.
     */
    MemMirror() = default;

    /**
     * Constructor that will also directly allocate the memory for both
     * cpu and gpu Mem_t objects
     *
     * @param cardinality: cardinality of the type
     * @param order: memory layout order: structure of array or arrays of structures
     * @param cpuDevId: Id of the CPU for allocation
     * @param cpuAllocType: type of cpu allocation
     * @param gpuDevId: ID of the GPU for allocation
     * @param gpuAllocType: type of GPU allocation
     * @param nElements: number of element to be allocated.
     */
    MemMirror(int                cardinality,
              Neon::MemoryLayout order,
              DeviceID           cpuDevId,
              Neon::Allocator    cpuAllocType,
              DeviceID           gpuDevId,
              Neon::Allocator    gpuAllocType,
              int64_t            nElements,
              bool               noThrowForAZeroSize = false)
    {
        m_noThrowForAZeroSize = noThrowForAZeroSize;
        if (m_noThrowForAZeroSize && nElements == 0) {
            return;
        }
        m_cpu = MemDevice<T_ta>(cardinality, order,  Neon::DeviceType::CPU, cpuDevId, cpuAllocType, nElements);
        m_gpu = MemDevice<T_ta>(cardinality, order,  Neon::DeviceType::CUDA, gpuDevId, gpuAllocType, nElements);
        if (gpuAllocType == Neon::Allocator::NULL_MEM) {
            isACPUdevice = true;
        }
    }

    /**
     * Function that check if two Mem_t object are compatible and can be used to create a mirror object.
     * @param cpu
     * @param gpu
     * @return
     */
    static auto compatibility(const MemDevice<T_ta>& cpu, const MemDevice<T_ta>& gpu) -> bool
    {
        bool compatible = true;
        compatible = cpu.cardinality() == gpu.cardinality() ? compatible : false;
        compatible = cpu.nElements() == gpu.nElements() ? compatible : false;
        if (cpu.allocType() == Neon::Allocator::NULL_MEM || gpu.allocType() == Neon::Allocator::NULL_MEM) {
            return compatible;
        }
        compatible = cpu.pitch() == gpu.pitch() ? compatible : false;
        return compatible;
    }

    /**
     * Attach a Mem_t to the mirror.
     * The mirror acquires an ownership token of the Mem_t.
     * If the input Mem_t is destructed, the mirror will present the resources associated
     * to the Mem_t object to be released.
     * @param mem
     */
    void link(MemDevice<T_ta>& mem)
    {
        if (Neon::DeviceType::CPU == mem.devType()) {
            if (gpu_configured) {
                if (!compatibility(mem, m_gpu)) {
                    NeonException exc("");
                    exc << "Linking error: incompatible layout between the two Mem_t";
                    NEON_THROW(exc);
                }
            }
            cpu_configured = true;
            m_cpu = mem;

            if (m_gpu.allocType() == Neon::Allocator::NULL_MEM) {
                isACPUdevice = true;
            }
            return;
        }
        if (Neon::DeviceType::CUDA == mem.devType()) {
            if (cpu_configured) {
                if (!compatibility(mem, m_cpu)) {
                    NeonException exc("");
                    exc << "Linking error: incompatible layout between the two Mem_t";
                    NEON_THROW(exc);
                }
            }
            m_gpu = mem;
            gpu_configured = true;

            if (m_gpu.allocType() == Neon::Allocator::NULL_MEM) {
                isACPUdevice = true;
            }
            return;
        }
        NeonException exc("");
        exc << "Unsupported device type";
        NEON_THROW(exc);
    }

    /**
     * Detach the cpu or gpu Mem_t object from the mirror.
     * @param devEt
     * @return
     */
    auto detach(Neon::DeviceType devEt) -> MemDevice<T_ta>
    {
        if (Neon::DeviceType::CPU == devEt) {
            cpu_configured = false;
            MemDevice<T_ta> ret = m_cpu;
            m_cpu = MemDevice<T_ta>();
            return ret;
        }
        if (Neon::DeviceType::CUDA == devEt) {
            gpu_configured = false;
            MemDevice<T_ta> ret = m_gpu;
            m_gpu = MemDevice<T_ta>();
            return ret;
        }
        NeonException exc("");
        exc << "Unsupported device type";
        NEON_THROW(exc);
    }

    /**
     * Reset gpu and cpu Mem_t object.
     * The resources of those object are released only
     * if their reference counter has gone to zero.
     */
    auto reset() -> void
    {
        m_noThrowForAZeroSize = false;
        cpu_configured = false;
        m_cpu = MemDevice<T_ta>();
        gpu_configured = false;
        m_gpu = MemDevice<T_ta>();
        return;
    }

    /**
     * Copy the content form one Mem_t to the other: i.e. form gpu to cpu
     * or form cpu to gpu.
     * A stream is needed for the operation.
     * @tparam run_ta
     * @param devEt
     * @param gpuStream
     */
    template <Neon::run_et::et run_ta = Neon::run_et::et::async>
    void update(Neon::DeviceType devEt, const Neon::sys::GpuStream& gpuStream)
    {
        if (m_noThrowForAZeroSize) {
            return;
        }
        MemDevice<T_ta>* src;
        MemDevice<T_ta>* dst;
        if (Neon::DeviceType::CPU == devEt) {
            dst = &m_cpu;
            src = &m_gpu;
        } else {
            dst = &m_gpu;
            src = &m_cpu;
        }
        dst->template updateFrom<run_ta>(gpuStream, *src);
    }

    template <Neon::run_et::et run_ta = Neon::run_et::et::async>
    void updateWindow(Neon::DeviceType            devEt,
                      const Neon::sys::GpuStream& gpuStream,
                      eIdx_t                      start,
                      int64_t                     nEl,
                      int                         cardinality = 0)
    {
        if (m_noThrowForAZeroSize) {
            return;
        }
        MemDevice<T_ta>* src;
        MemDevice<T_ta>* dst;
        if (Neon::DeviceType::CPU == devEt) {
            dst = &m_cpu;
            src = &m_gpu;
        } else {
            dst = &m_gpu;
            src = &m_cpu;
        }
        dst->template updateWindowFrom<run_ta>(gpuStream, *src, start, cardinality, nEl);
    }

    template <Neon::run_et::et run_ta = Neon::run_et::et::async>
    void updateWindow(Neon::DeviceType devEt, const Neon::sys::GpuStream& gpuStream)
    {
        if (m_noThrowForAZeroSize) {
            return;
        }
        MemDevice<T_ta>* src;
        MemDevice<T_ta>* dst;
        if (Neon::DeviceType::CPU == devEt) {
            dst = &m_cpu;
            src = &m_gpu;
        } else {
            dst = &m_gpu;
            src = &m_cpu;
        }
        dst->template updateFrom<run_ta>(gpuStream, *src);
    }
    /**
     * Extract raw memory from one of the two Mem_t object
     * that are stored by this class.
     * @param devEt
     * @return
     */
    T_ta* rawMem(Neon::DeviceType devEt)
    {
        MemDevice<T_ta>* target;
        if (Neon::DeviceType::CPU == devEt) {
            target = &m_cpu;
        } else {
            target = &m_gpu;
        }
        return target->mem();
    }

    auto rawMem(Neon::Execution execution) -> T_ta*
    {
        MemDevice<T_ta>* target;
        if (Neon::Execution::host == execution || isACPUdevice) {
            target = &m_cpu;
        } else {
            target = &m_gpu;
        }
        return target->mem();
    }

    auto rawMem(Neon::Execution execution) const -> T_ta const*
    {
        MemDevice<T_ta> const* target;
        if (Neon::Execution::host == execution) {
            target = &m_cpu;
        } else {
            target = &m_gpu;
        }
        return target->mem();
    }

    auto compute(Neon::DeviceType devEt)
        -> Partition&
    {
        switch (devEt) {
            case Neon::DeviceType::CPU: {
                return m_cpu.compute();
            }
            case Neon::DeviceType::CUDA: {
                return m_gpu.compute();
            }
            default: {
                NEON_THROW_UNSUPPORTED_OPTION();
            }
        }
    }

    auto compute(Neon::DeviceType devEt)
        const
        -> const Partition&
    {
        switch (devEt) {
            case Neon::DeviceType::CPU: {
                return cself().m_cpu.compute();
            }
            case Neon::DeviceType::CUDA: {
                return cself().m_gpu.compute();
            }
            default: {
                NEON_THROW_UNSUPPORTED_OPTION();
            }
        }
    }

    inline auto cself() const -> const MemMirror&
    {
        return *this;
    }
    inline auto self() -> MemMirror&
    {
        return *this;
    }

    auto eRef(Neon::SetIdx /*setIdx*/,
              const Self::eIdx_t& /*eIdx*/,
              int /*cardinality*/) const -> const Element&
    {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

    auto eRef(Neon::SetIdx /*setIdx*/,
              const Self::eIdx_t& /*eIdx*/,
              int /*cardinalityy*/) -> Element&
    {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }

    auto ceRef(Neon::SetIdx /*setIdx*/,
               const Self::eIdx_t& /*eIdx*/,
               int /*cardinality*/) const -> const Element&
    {
        NEON_DEV_UNDER_CONSTRUCTION("");
    }
};

}  // namespace Neon
