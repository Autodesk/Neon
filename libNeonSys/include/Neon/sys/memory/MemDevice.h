#pragma once

#include <atomic>

#include "Neon/core/core.h"
#include "Neon/core/tools/io/exportVTI.h"
#include "Neon/core/types/Allocator.h"
#include "Neon/core/types/memOptions.h"
#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/devices/cpu/CpuDevice.h"
#include "Neon/sys/devices/gpu/GpuDevice.h"
#include "Neon/sys/memory/memConf.h"
namespace Neon::sys {

/**
 * Mem_t is used to abstract memory buffers. A Mem_t can be user of system initialized.
 *
 * System initialized means that memory is allocated and tracked by the Neon system.
 * In such case Mem_t behaves as a smart pointer, cleaning up memory then are no more reference to the object.
 *
 * User initialized means that the memory is provided by the user. In this case, Mem_t stores only some information
 * such as size and associated device, but memory is not tracked. This user initialized provides the same buffer
 * interface for memory that for certain reasons have already been allocated by the user.
 */
template <typename T_ta = char>
struct MemDevice
{

   public:
    // ALIAS
    struct Partition
    {
       private:
        T_ta*      m_buffer{nullptr}; /**< Pointer to the actual memory */
        index64_2d m_pitch{-1, -1};

       public:
        friend MemDevice<T_ta>;

        using element_t = T_ta;
        using eIdx_t = index64_t;

        Partition() = default;
        ~Partition() = default;
        Partition(T_ta* buffer, const index64_2d& pitch)
        {
            m_buffer = buffer;
            m_pitch = pitch;
        }

        NEON_CUDA_HOST_DEVICE auto addr() -> T_ta*
        {
            return m_buffer;
        }

        NEON_CUDA_HOST_DEVICE auto addr() const -> const T_ta*
        {
            return m_buffer;
        }


        NEON_CUDA_HOST_DEVICE auto ref() -> T_ta&
        {
            return *m_buffer;
        }

        NEON_CUDA_HOST_DEVICE auto ref() const -> const T_ta&
        {
            return *m_buffer;
        }

        NEON_CUDA_HOST_DEVICE auto cref() const -> const T_ta&
        {
            return *m_buffer;
        }

        NEON_CUDA_HOST_DEVICE auto eRef(const index64_t& idx, int cardIdx = 0) -> T_ta&
        {
            const size_t pitch = m_pitch.pMain * size_t(idx) + m_pitch.pCardinality * size_t(cardIdx);
            return m_buffer[pitch];
        }

        NEON_CUDA_HOST_DEVICE auto eRef(const index64_t& idx, int cardIdx = 0) const -> const T_ta&
        {
            const size_t pitch = m_pitch.pMain * size_t(idx) + m_pitch.pCardinality * size_t(cardIdx);
            return m_buffer[pitch];
        }

        NEON_CUDA_HOST_DEVICE auto eVal(const index64_t& idx, int cardIdx = 0) const -> T_ta
        {
            const size_t pitch = m_pitch.pMain * size_t(idx) + m_pitch.pCardinality * size_t(cardIdx);
            return m_buffer[pitch];
        }

        NEON_CUDA_HOST_DEVICE auto cardPitch() const -> size_t
        {
            return m_pitch.pCardinality;
        }

        NEON_CUDA_HOST_DEVICE auto mainPitch() const -> size_t
        {
            return m_pitch.pMain;
        }
    };

    using local_t = Partition;
    using element_t = typename Partition::element_t;
    using eIdx_t = typename Partition::eIdx_t;


   private:
    const DeviceType      m_devType;   /**< Type of the device holding the memory buffer */
    const DeviceID        m_devIdx;    /**< ID of the device holding the memory buffer */
    Neon::Allocator       m_allocType; /**< Type of allocator used */
    const uint64_t        m_nElements = {0};
    int                   m_cardinality = {1};
    Neon::MemoryLayout    m_order = {Neon::MemoryLayout::structOfArrays};
    size_t                m_allocatedBytes = {0};
    size_t                m_requiredBytes = {0};          /** required memory to support padding */
    void*                 m_notAlignedBuffer = {nullptr}; /**< Not Aligned buffer */
    std::atomic_uint64_t* m_refCounter = {nullptr};       /**< Reference counter used for garbage collection */
    Partition             m_compute;

   public:
    /**
     * Destructor with a garbage collection policy
     */
    virtual ~MemDevice();

    /**
     * Empty constructor
     */
    MemDevice();

    /**
     * Constructor (User Managed)
     */
    MemDevice(DeviceType devType,
              DeviceID   devIdx,
              uint64_t   nElements,
              T_ta*      buffer);

    /**
     * Constructor (Sys managed)
     */
    MemDevice(DeviceType      devType,
              DeviceID        devId,
              Neon::Allocator allocType,
              uint64_t        nElements);

    /**
     * Constructor (Sys managed)
     */
    MemDevice(int                cardinality,
              Neon::MemoryLayout order,
              DeviceType         devType,
              DeviceID           devId,
              Neon::Allocator    allocType,
              uint64_t           nElements);


    /**
     * Copy constructor
     * @param other
     */
    MemDevice(const MemDevice& other);

    /**
     * Move constructor
     * @param other
     */
    MemDevice(MemDevice&& other);

    /**
     * Copy assignment operator
     * @param other
     */
    MemDevice& operator=(const MemDevice& other) noexcept;

    /**
     * Move assignment operator
     * @param other
     */
    MemDevice& operator=(MemDevice&& other) noexcept;

    /**
     * Returns the raw address of the allocated memory, for read and write operations
     * @tparam T_ta
     * @return
     */
    T_ta* mem()
    {
        return (T_ta*)m_compute.m_buffer;
    }

    /**
     * Returns the raw address of the allocated memory, for read only operations
     * @tparam T_ta
     * @return
     */
    const T_ta* mem() const
    {
        return (const T_ta*)m_compute.m_buffer;
    }

    /**
     * Release all the memory that has been allocated.
     */
    void release();

    /**
     * Returns what type of management policy apply to the memory buffers:
     * 1. user: no de-allocation is done by this object
     * 2. system: this object is in charge of release the allocated memory
     * @return
     */
    managedMode_t managedMode() const;

    /**
     * Returns the type of the device that owns the allocated memory
     * @return
     */
    const DeviceType& devType() const;

    /**
     * Returns the ID of the device that owns the allocated memory
     * @return
     */
    const DeviceID& devIdx() const;

    /**
     * Returns the type of memory that was allocated
     * @return
     */
    const Neon::Allocator& allocType() const;

    /**
     * Returns size of the managed memory buffer.
     * @return
     */
    uint64_t nBytes() const;

    /**
     *
     * @param mem
     */
    void copyFrom(const MemDevice& mem);

    /**
     * Copy the content of the this MemDev to mem. While copyFrom can be used to do the same operation,
     * copyTo launch all memory transfer async and let the user sync when they actually need.
     */
    void copyTo(MemDevice& mem, Neon::sys::GpuStream& stream);

    template <Neon::run_et::et runMode_ta>
    void updateFrom(const Neon::sys::GpuStream& gpuStream, const MemDevice<T_ta>& mem);

    template <Neon::run_et::et runMode_ta>
    void updateWindowFrom(const Neon::sys::GpuStream& gpuStream,
                          const MemDevice<T_ta>&      mem,
                          eIdx_t                      start,
                          int                         cardinality,
                          int64_t                     nEl);

    /**
     *
     * @param devType
     * @param devIdx
     * @param allocType
     * @return
     */
    MemDevice clone(DeviceType devType, DeviceID devIdx, Neon::Allocator allocType) const;

    /**
     *
     * @tparam Type_ta
     * @param nComponents
     * @param filename
     * @param mat_space
     * @param spacingData
     * @param origin
     * @param dataName
     */
    template <Neon::vti_e::e vti_ta, typename userReadType_ta = T_ta, typename vtiWriteType_ta = userReadType_ta>
    void exporVti(const int32_t                 nComponents,
                  std::string                   filename,
                  const index_3d                mat_space,
                  const Vec_3d<vtiWriteType_ta> spacingData = Vec_3d<vtiWriteType_ta>(1, 1, 1),
                  const Vec_3d<vtiWriteType_ta> origin = Vec_3d<vtiWriteType_ta>(0, 0, 0),
                  std::string                   dataName = std::string("Data"))
    {
        if (m_devType == Neon::DeviceType::CUDA) {
            MemDevice tmp = this->clone(DeviceType::CPU, 0, Neon::Allocator::MALLOC);
            tmp.template exporVti<vti_ta, userReadType_ta, vtiWriteType_ta>(nComponents, filename, mat_space, spacingData, origin, dataName);
            return;
        }
        //        template <typename userReadType_ta, typename vtiWriteType_ta = userReadType_ta>
        //        void writeNodesToVTI(const userReadType_ta*    memGrid,
        //                             const int32_t         nComponents,
        //                             std::string           filename,
        //                             const index_3d        mat_space,
        //                             const Vec_3d<vtiWriteType_ta> spacingData,
        //                             const Vec_3d<vtiWriteType_ta> origin,
        //                             std::string           dataName = std::string("Data"))

        Neon::exportVti<vti_ta, userReadType_ta, vtiWriteType_ta>((userReadType_ta*)m_compute.m_buffer,
                                                                  nComponents,
                                                                  filename,
                                                                  mat_space,
                                                                  spacingData, origin, dataName);
        return;
    }

    /**
     * Returns the number of object of type T_ta as if would have
     * cardinality equal to one
     * @return
     */
    uint64_t nElements() const
    {
        return m_nElements;
    }

    int               cardinality() const;
    const index64_2d& pitch() const;
    auto              order() const -> MemoryLayout;

    /**
     * Returns a read only reference to i-th element stored in the Mem_t buffer
     * @tparam access_ta
     * @param idx
     * @return
     */
    template <Neon::Access access_ta = Neon::Access::read>
    std::enable_if_t<access_ta == Neon::Access::read, const T_ta&> elRef(const index64_t& idx) const
    {
        return this->mem()[idx];
    }

    /**
     * Returns a mutable reference to i-th element stored in the Mem_t buffer
     * @tparam access_ta
     * @param idx
     * @return
     */
    template <Neon::Access access_ta = Neon::Access::read>
    std::enable_if_t<access_ta == Neon::Access::readWrite, T_ta&> elRef(const index64_t& idx)
    {
        return this->mem()[idx];
    }

    /**
     * Returns a read only reference to i-th element stored in the Mem_t buffer
     * @tparam access_ta
     * @param idx
     * @return
     */
    template <Neon::Access access_ta = Neon::Access::read>
    std::enable_if_t<access_ta == Neon::Access::read, const T_ta&> elRef(const index64_t& idx, int cardIdx) const
    {
        // static_assert(m_pitch.pMain>0);
        size_t pitch = m_compute.m_pitch.pMain * size_t(idx) + m_compute.m_pitch.pCardinality * size_t(cardIdx);
        return this->mem()[pitch];
    }

    /**
     * Returns a mutable reference to i-th element stored in the Mem_t buffer
     * @tparam access_ta
     * @param idx
     * @return
     */
    template <Neon::Access access_ta = Neon::Access::read>
    std::enable_if_t<access_ta == Neon::Access::readWrite, T_ta&> elRef(const index64_t& idx, int cardIdx)
    {
        // static_assert(m_pitch.pMain>0);
        size_t pitch = m_compute.m_pitch.pMain * size_t(idx) + m_compute.m_pitch.pCardinality * size_t(cardIdx);
        return this->mem()[pitch];
    }

    /**
     *
     * @return
     */
    auto compute()
        const
        -> const Partition&
    {
        return m_compute;
    }

    /**
     *
     * @return
     */
    auto compute(Neon::DeviceType)
        const
        -> const Partition&
    {
        return m_compute;
    }

    /**
     *
     * @return
     */
    auto compute()
        -> Partition&
    {
        return m_compute;
    }

    /**
     *
     * @return
     */
    auto compute(Neon::DeviceType)
        -> Partition&
    {
        return m_compute;
    }

   private:
    /**
     * Helper function to free the memory
     */
    void helperFreeMem();
    /**
     * Helper function to allocate GPU memory
     */
    void helperAllocMem();

    /**
     * Helper function to reset local information
     */
    void helperResetLocal();

    /**
     * Helper function to reset global information
     */
    void helperResetGlobal();
};

template <typename T_ta>
using Memlocal_t = typename MemDevice<T_ta>::Partition;

}  // namespace Neon::sys

#include "Neon/sys/memory/memDevice_imp.h"
