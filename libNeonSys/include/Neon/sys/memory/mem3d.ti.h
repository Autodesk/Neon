#pragma once

#include "Neon/core/core.h"
#include "Neon/core/tools/io/exportVTI.h"
#include "Neon/sys/devices/cpu/CpuDevice.h"
#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/devices/gpu/GpuDevice.h"
#include "Neon/sys/memory/MemDevice.h"

#include <atomic>


namespace Neon {
namespace sys {

template <typename T_ta>
void Mem3d_t<T_ta>::h_init(int                     cardinality,
                           DeviceType              devType,
                           DeviceID                devId,
                           Neon::Allocator         allocType,
                           const Neon::index_3d&   dim,
                           const Neon::index_3d&   halo,
                           memLayout_et::order_e   layoutOrder,
                           MemAlignment            alignment,
                           memLayout_et::padding_e padding)
{
    m_cardinality = cardinality;
    m_dim = dim;
    m_halo = halo;
    m_haloedDim = dim + halo * 2;
    m_layoutOrder = layoutOrder;
    m_alignment = alignment;
    m_padding = padding;

    switch (padding) {
        case memLayout_et::padding_e::OFF: {
            const auto haloedNumElementsPerComponent = m_haloedDim.rMulTyped<int64_t>();
            m_nElementsIncludingHaloPaddingCardinality =
                haloedNumElementsPerComponent * m_cardinality;

            m_mem = MemDevice<T_ta>(devType, devId,
                                   allocType,
                                   m_nElementsIncludingHaloPaddingCardinality,
                                   alignment);

            if (m_cardinality == 1) {
                m_elPitch.x = 1;
                m_elPitch.y = size_t(m_haloedDim.x);
                m_elPitch.z = m_elPitch.y * size_t(m_haloedDim.y);
                m_elPitch.w = 0;
                m_userPointer = m_mem.mem();
                return;
            } else {
                switch (m_layoutOrder) {
                    case memLayout_et::structOfArrays: {
                        m_elPitch.x = 1;
                        m_elPitch.y = m_haloedDim.x;
                        m_elPitch.z = m_elPitch.y * size_t(m_haloedDim.y);
                        m_elPitch.w = m_elPitch.z * size_t(m_haloedDim.z);
                        m_userPointer = m_mem.mem();
                        return;
                    }
                    case memLayout_et::arrayOfStructs: {
                        m_elPitch.x = m_cardinality;
                        m_elPitch.y = size_t(m_haloedDim.x) * m_elPitch.x;
                        m_elPitch.z = m_elPitch.y * size_t(m_haloedDim.y);
                        m_elPitch.w = 1;
                        m_userPointer = m_mem.mem();
                        return;
                    }
                }
            }
            return;
        }
        case memLayout_et::ON: {
            const uint32_t align_exp = m_alignment.expAlign(devType, devId);
            const uint32_t align_byte = m_alignment.exp2byte(align_exp);
            const uint32_t align_byEl = align_byte / sizeof(T_ta);
            const uint32_t align_reminder = align_byte % sizeof(T_ta);

            if (align_reminder != 0) {
                NeonException exp("Mem3d_t");
                exp << "Padding violation: alignment is not multiple of user type.";
                NEON_THROW(exp);
            }

            int elPadding = 0;

            switch (m_layoutOrder) {
                case memLayout_et::structOfArrays: {
                    {
                        elPadding = (dim.x + 2 * halo.x) % align_byEl;
                        elPadding = elPadding == 0 ? elPadding : align_byEl - elPadding;

                        Neon::index_3d haloedDim = m_haloedDim;
                        haloedDim.x += elPadding;
                        const auto haloedNumElementsPerComponent = haloedDim.rMulTyped<int64_t>();
                        m_nElementsIncludingHaloPaddingCardinality =
                            haloedNumElementsPerComponent * m_cardinality;
                    }
                    const auto allocatedElements = m_nElementsIncludingHaloPaddingCardinality + align_byEl;
                    m_mem = MemDevice<T_ta>(devType, devId,
                                           allocType, allocatedElements,
                                           memAlignment_et::system);

                    void*  rawUnAlignedPointer = m_mem.mem();
                    size_t totalSize = allocatedElements * sizeof(T_ta);
                    m_userPointer = (T_ta*)std::align(align_byEl * sizeof(T_ta),
                                                      m_nElementsIncludingHaloPaddingCardinality * sizeof(T_ta),
                                                      rawUnAlignedPointer,
                                                      totalSize);

                    if (m_userPointer == nullptr) {
                        NeonException exp;
                        exp << "Unable to satisfy required alignment";
                        NEON_THROW(exp);
                    }

                    m_elPitch.x = 1;
                    m_elPitch.y = (m_haloedDim.x + elPadding);
                    m_elPitch.z = m_elPitch.y * size_t(m_haloedDim.y);
                    m_elPitch.w = m_elPitch.z * size_t(m_haloedDim.z);

                    return;
                }
                case memLayout_et::arrayOfStructs: {
                    {
                        elPadding = ((dim.x + 2 * halo.x) * m_cardinality) % align_byEl;
                        elPadding = elPadding == 0 ? elPadding : align_byEl - elPadding;

                        Neon::index_3d haloedPaddedDim;
                        haloedPaddedDim.x = ((dim.x + 2 * halo.x) * m_cardinality) + elPadding;
                        haloedPaddedDim.y = (dim.y + 2 * halo.y);
                        haloedPaddedDim.z = (dim.z + 2 * halo.z);
                        m_nElementsIncludingHaloPaddingCardinality = haloedPaddedDim.rMulTyped<int64_t>();
                    }
                    const auto allocatedElements = m_nElementsIncludingHaloPaddingCardinality + align_byEl;

                    m_mem = MemDevice<T_ta>(devType, devId,
                                           allocType, allocatedElements,
                                           memAlignment_et::system);

                    void*  rawUnAlignedPointer = m_mem.mem();
                    size_t totalSize = allocatedElements * sizeof(T_ta);

                    m_userPointer = (T_ta*)std::align(align_byEl * sizeof(T_ta),
                                                      m_nElementsIncludingHaloPaddingCardinality * sizeof(T_ta),
                                                      rawUnAlignedPointer,
                                                      totalSize);

                    if (m_userPointer == nullptr) {
                        NeonException exp("Mem3d_t");
                        exp << "Unable to satisfy required alignment";
                        NEON_THROW(exp);
                    }

                    m_elPitch.x = m_cardinality;
                    m_elPitch.y = (size_t(m_haloedDim.x) * m_cardinality) + elPadding;
                    m_elPitch.z = m_elPitch.y * size_t(m_haloedDim.y);
                    m_elPitch.w = 1;
                    return;
                }
            }
        }
    }
}

template <typename T_ta>
template <Neon::run_et::et runMode_ta>
void Mem3d_t<T_ta>::fastCopyFrom(const Neon::sys::GpuStream& stream,
                                 const Mem3d_t<T_ta>&          other)
{
    int exp = this->alignment().expAlign(this->devType(), this->devIdx());
    int expOther = other.alignment().expAlign(other.devType(), other.devIdx());
    if ((exp != expOther) && (m_padding == memLayout_et::padding_e::ON)) {
        NeonException exception("Mem3d_t::fastCopyFrom");
        exception << "fastCopyFrom can not be used. Different padding was detected";
        NEON_THROW(exception);
    }

    //we need to use  m_userPointer and not m_mem.
    MemDevice<T_ta> thisMem(m_mem.devType(), m_mem.devIdx(), m_nElementsIncludingHaloPaddingCardinality, m_userPointer);
    MemDevice<T_ta> otherMem(other.devType(), other.devIdx(), m_nElementsIncludingHaloPaddingCardinality, other.m_userPointer);

    thisMem.template updateFrom<runMode_ta>(stream, otherMem);
}


template <typename T_ta>
void Mem3d_t<T_ta>::copyFrom(const Mem3d_t<T_ta>& other)
{
    if (m_dim != other.dim()) {
        NeonException exception("copyFrom");
        exception << "Two Mem3d_t must have the same dim and halo.";
        NEON_THROW(exception);
    }

    if (m_mem.devType() == other.devType()) {
        /**
         * CPU <- CPU
         */
        if (m_mem.devType() == Neon::DeviceType::CPU) {
            if (m_halo == other.halo()) {
                index_3d fullDim = m_dim + m_halo * 2;
#if _MSC_VER <= 1916
#pragma omp parallel for simd collapse(2)
#endif
                for (int z = 0; z < fullDim.z; z++) {
                    for (int y = 0; y < fullDim.y; y++) {
                        for (int x = 0; x < fullDim.x; x++) {
                            for (int v = 0; v < m_cardinality; v++) {

                                const size_t elPitch = this->idxInHaloedSpace(x, y, z, v);
                                const size_t otherElPitch = other.idxInHaloedSpace(x, y, z, v);

                                this->mem()[elPitch] = other.mem()[otherElPitch];
                            }
                        }
                    }
                }
            } else {
                index_3d fullDim = m_dim;
#if _MSC_VER <= 1916
#pragma omp parallel for simd collapse(2)
#endif
                for (int z = 0; z < fullDim.z; z++) {
                    for (int y = 0; y < fullDim.y; y++) {
                        for (int x = 0; x < fullDim.x; x++) {
                            for (int v = 0; v < m_cardinality; v++) {
                                const size_t elPitch = this->idx(x, y, z, v);
                                const size_t otherElPitch = other.idx(x, y, z, v);

                                this->mem()[elPitch] = other.mem()[otherElPitch];
                            }
                        }
                    }
                }
            }
            return;
        }
        if (m_mem.devType() == Neon::DeviceType::CUDA) {
            /**
             * CUDA <- CUDA
             */

            if (m_mem.devIdx() == other.devIdx()) {
                Mem3d_t<T_ta> thisCpuCopy_mem3d(
                    cloning_e::LAYOUT, *this, this->halo(),
                    Neon::DeviceType::CPU, Neon::sys::DeviceID(0), Neon::Allocator::MALLOC,
                    this->layout(), this->alignment(), this->padding());

                Mem3d_t<T_ta> cpuCopyOther_mem3d(
                    cloning_e::LAYOUT, other, other.halo(),
                    Neon::DeviceType::CPU, Neon::sys::DeviceID(0), Neon::Allocator::MALLOC,
                    other.layout(), other.alignment(), other.padding());

                thisCpuCopy_mem3d.copyFrom(cpuCopyOther_mem3d);
                m_mem.copyFrom(thisCpuCopy_mem3d.m_mem);

                return;
            }
        }
    }

    /**
     * CUDA <- CPU
     */
    if (m_mem.devType() == Neon::DeviceType::CUDA && other.devType() == Neon::DeviceType::CPU) {

        // Create a copy of this on CPU
        Mem3d_t<T_ta> thisCpuCopy_mem3d(
            cloning_e::LAYOUT, *this, this->halo(), Neon::DeviceType::CPU,
            Neon::sys::DeviceID(0), Neon::Allocator::MALLOC, this->layout(),
            this->alignment(), this->padding());

        // copy this CPU from other CPU
        thisCpuCopy_mem3d.copyFrom(other);

        // copy this CPU to GPU
        m_mem.copyFrom(thisCpuCopy_mem3d.m_mem);
        return;
    }
    /**
     * CPU <- CUDA
     */
    if (m_mem.devType() == Neon::DeviceType::CPU && other.devType() == Neon::DeviceType::CUDA) {
        // Create other CPU
        Mem3d_t<T_ta> cpuCopyOther_mem3d(
            cloning_e::LAYOUT, other, other.halo(), Neon::DeviceType::CPU,
            Neon::sys::DeviceID(0), Neon::Allocator::MALLOC, other.layout(),
            other.alignment(), other.padding());

        // copy this from other CPU
        this->copyFrom(cpuCopyOther_mem3d);

        return;
    }

    NeonException exception("copyFrom");
    exception << "Configuration not supported";
    NEON_THROW(exception);
}  // namespace sys


template <typename T_ta>
template <Neon::vti_e::e vti_ta,
          typename vtiWriteType_ta,
          typename vtiGridLocationType_ta>
void Mem3d_t<T_ta>::exportVti(bool                                 includeHalo,
                              std::string                          filename,
                              const Vec_3d<vtiGridLocationType_ta> spacingData,
                              const Vec_3d<vtiGridLocationType_ta> origin,
                              std::string                          dataName)
{
    bool directExport = true;
    directExport = directExport && (m_padding == memLayout_et::OFF);
    directExport = directExport && (m_layoutOrder == memLayout_et::arrayOfStructs);
    directExport = directExport && (this->devType() == Neon::DeviceType::CPU);
    directExport = directExport && ((this->halo() == 0) || (includeHalo == true));

    if (directExport) {
        std::string newFilename = filename + std::string(".vti");

        index_3d targetDim = (!includeHalo) ? m_dim : m_dim + m_halo * 2;
        Neon::exportVti<vti_ta, T_ta, vtiWriteType_ta>(this->mem(), m_cardinality,
                                                       newFilename, targetDim, spacingData, origin, dataName);
        return;
    }

    auto     newAlignment = Neon::sys::MemAlignment(Neon::sys::memAlignment_et::system);
    index_3d newHalo = includeHalo ? this->halo() : index_3d(0);

    Mem3d_t<T_ta> newLayout(cloning_e::DATA,
                            *this,
                            newHalo,
                            Neon::DeviceType::CPU,
                            0,
                            Neon::Allocator::MALLOC,
                            memLayout_et::arrayOfStructs,
                            newAlignment,
                            memLayout_et::OFF);
    newLayout.template exportVti<vti_ta, vtiWriteType_ta>(includeHalo,
                                                          filename, spacingData, origin, dataName);

    return;
}


template <typename T_ta>
void Mem3d_t<T_ta>::memset(uint8_t val) const
{
    const size_t byteCount = m_nElementsIncludingHaloPaddingCardinality * sizeof(T_ta);
    if (m_mem.devType() == Neon::DeviceType::CPU) {
        std::memset((void*)(m_userPointer), val, byteCount);
    }
    if (m_mem.devType() == Neon::DeviceType::CUDA) {
        Neon::sys::DeviceID devId = this->devIdx();
        auto&             gpuDev = Neon::sys::globalSpace::gpuSysObj().dev(devId);
        gpuDev.memory.memSet((void*)(m_userPointer),
                             val,
                             byteCount);
    }

    return;
}

}  // namespace sys
}  // namespace Neon
