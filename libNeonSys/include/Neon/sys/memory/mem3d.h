#pragma once

#include "Neon/core/core.h"
#include "Neon/core/tools/io/exportVTI.h"
#include "Neon/core/types/Allocator.h"
#include "Neon/core/types/memOptions.h"
#include "Neon/sys/devices/cpu/CpuDevice.h"
#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/devices/gpu/GpuDevice.h"
#include "Neon/sys/memory/MemDevice.h"

#include <atomic>


namespace Neon {
namespace sys {

struct cloning_e
{
    enum e : int32_t
    {
        DATA,
        LAYOUT,
    };

   private:
    e m_cloning{e::DATA};

   public:
    cloning_e() = default;

    cloning_e(e cloning)
    {
        m_cloning = cloning;
    }


    e config() const
    {
        return m_cloning;
    }
};


/**
 * Mem3d_t is used to abstract memory buffers. A Mem3d_t can be user of system initialized.
 *
 * System initialized means that memory is allocated and tracked by the Neon system.
 * In such case Mem3d_t behaves as a smart pointer, cleaning up memory then are no more reference to the object.
 *
 * User initialized means that the memory is provided by the user. In this case, Mem3d_t stores only some information
 * such as size and associated device, but memory is not tracked. This user initialized provides the same buffer
 * interface for memory that for certain reasons have already been allocated by the user.
 */
template <typename T_ta>
struct Mem3d_t
{
   private:
    MemDevice<T_ta>          m_mem;
    Neon::index_3d          m_dim;                  /** 3D dimension of the discrete domain */
    Neon::index_3d          m_haloedDim;            /** 3D size of the haloed domain */
    Neon::index_3d          m_halo;                 /** Radius of the halo in x, y, and z directions */
    Neon::size_4d           m_elPitch;              /** Element pitch for the 3D layout. The m_elPitch is used to map 3D indexes to 1D raw memory indexing space */
    T_ta*                   m_userPointer{nullptr}; /** User pointer to the padded and aligned memory **/
    size_t                  m_nElementsIncludingHaloPaddingCardinality{0};
    MemAlignment             m_alignment;
    int                     m_cardinality;
    memLayout_et::padding_e m_padding;
    memLayout_et::order_e   m_layoutOrder;

    void h_init(int                     cardinality,
                DeviceType              devType,
                DeviceID                devId,
                Neon::Allocator         allocType,
                const Neon::index_3d&   dims,
                const Neon::index_3d&   halo,
                memLayout_et::order_e   layoutOrder = memLayout_et::structOfArrays,
                MemAlignment            alignment = MemAlignment(),
                memLayout_et::padding_e padding = memLayout_et::OFF);

   public:
    virtual ~Mem3d_t() = default;

    Mem3d_t() = default;

    Mem3d_t(int                     cardinality,
            DeviceType              devType,
            DeviceID                devId,
            Neon::Allocator         allocType,
            const Neon::index_3d&   dim,
            const Neon::index_3d&   halo,
            memLayout_et::order_e   layoutOrder = memLayout_et::structOfArrays,
            MemAlignment            alignment = MemAlignment(),
            memLayout_et::padding_e padding = memLayout_et::OFF)
    {
        this->h_init(cardinality, devType, devId, allocType, dim, halo, layoutOrder,
                     alignment, padding);
    }


    Mem3d_t(const Mem3d_t& other) = default;


    Mem3d_t(cloning_e               cloning,
            const Mem3d_t<T_ta>&    other,
            index_3d                newHalo,
            DeviceType              devType,
            DeviceID                devId,
            Neon::Allocator         allocType,
            memLayout_et::order_e   layoutOrder,
            MemAlignment            alignment,
            memLayout_et::padding_e padding)
    {
        this->h_init(other.m_cardinality, devType, devId, allocType, other.dim(),
                     newHalo, layoutOrder, alignment, padding);
        if (cloning.config() == cloning_e::DATA) {
            this->copyFrom(other);
        }
    }


    /*Mem3d_t<T_ta> clone(cloning_e cloning,
                        dev_et    devType,
                        dev_id    devId,
                        Neon::Allocator  allocType)
    {
        Mem3d_t<T_ta> newMem(cloning, *this, devType, devId, allocType);
        return newMem;
    }*/


    /*Mem3d_t<T_ta> clone(cloning_e      cloning,
                        dev_et         devType,
                        dev_id         devId,
                        Neon::Allocator       allocType,
                        memAlignment_t alignment) const
    {
        Mem3d_t<T_ta> newMem(cloning, devType, devIdx, allocType, alignment);
        return newMem;
    }*/

    /**
     * Move constructor
     * @param other
     */
    Mem3d_t(Mem3d_t&& other) = default;

    /**
     * Copy assignment operator
     * @param other
     */
    Mem3d_t& operator=(const Mem3d_t& other) = default;

    /**
     * Move assignment operator
     * @param other
     */
    Mem3d_t& operator=(Mem3d_t&& other) = default;

    /**
     * Returns the raw address of the allocated memory, for read and write operations
     * @tparam T_ta
     * @return
     */
    T_ta* mem()
    {
        return m_userPointer;
    }

    /**
     * Returns the raw address of the allocated memory, for read only operations
     * @tparam T_ta
     * @return
     */
    const T_ta* mem() const
    {
        return m_userPointer;
    }

    const Neon::size_4d& pitch() const
    {
        return m_elPitch;
    }

    /**
     * Release all the memory that has been allocated.
     */
    void release()
    {
        m_mem.release();
    }

    /**
     * Returns what type of management policy apply to the memory buffers:
     * 1. user: no de-allocation is done by this object
     * 2. system: this object is in charge of release the allocated memory
     * @return
     */
    managedMode_t managedMode() const
    {
        return m_mem.managedMode();
    }

    /**
     * Returns the type of the device that owns the allocated memory
     * @return
     */
    const DeviceType& devType() const
    {
        return m_mem.devType();
    }

    /**
     * Returns the ID of the device that owns the allocated memory
     * @return
     */
    const DeviceID& devIdx() const
    {
        return m_mem.devIdx();
    }

    /**
     * Returns the type of memory that was allocated
     * @return
     */
    const Neon::Allocator& allocType() const
    {
        return m_mem.allocType();
    }

    /**
     *
     * @param other
     */
    void copyFrom(const Mem3d_t<T_ta>& other);


    template <Neon::run_et::et runMode_ta>
    void fastCopyFrom(const Neon::sys::GpuStream& stream,
                      const Mem3d_t<T_ta>&          other);

    /**
     *
     * @tparam Type_ta
	 * @param[in] includeHalo boolean flag
     * @param filename
     * @param spacingData
     * @param origin
     * @param dataName
     */
    template <Neon::vti_e::e vti_ta, typename vtiWriteType_ta = T_ta, typename vtiGridLocationType_ta = double>
    void exportVti(bool                                 includeHalo,
                   std::string                          filename,
                   const Vec_3d<vtiGridLocationType_ta> spacingData = Vec_3d<vtiGridLocationType_ta>(1, 1, 1),
                   const Vec_3d<vtiGridLocationType_ta> origin = Vec_3d<vtiGridLocationType_ta>(0, 0, 0),
                   std::string                          dataName = std::string("Data"));

    void memset(uint8_t val) const;

    size_t idx(const Neon::index_3d& idx3d) const
    {
        size_t elPitch = size_t(idx3d.x + m_halo.x) +
                         size_t(idx3d.y + m_halo.y) * m_elPitch.y +
                         size_t(idx3d.z + m_halo.z) * m_elPitch.z;
        return elPitch;
    }

    size_t idxInHaloedSpace(const Neon::index_3d& idx3d) const
    {
        size_t elPitch = size_t(idx3d.x) +
                         size_t(idx3d.y) * m_elPitch.y +
                         size_t(idx3d.z) * m_elPitch.z;
        return elPitch;
    }

    size_t idx(const Neon::index_3d& idx3d, int component) const
    {
        size_t elPitch = 0;
        elPitch += size_t(idx3d.x + m_halo.x) * m_elPitch.y;
        elPitch += size_t(idx3d.y + m_halo.y) * m_elPitch.y;
        elPitch += size_t(idx3d.z + m_halo.z) * m_elPitch.z;
        elPitch += size_t(component) * m_elPitch.w;

        return elPitch;
    }

    size_t idxInHaloedSpace(const Neon::index_3d& idx3d, int component) const
    {
        size_t elPitch = 0;
        elPitch += size_t(idx3d.x) * m_elPitch.y;
        elPitch += size_t(idx3d.y) * m_elPitch.y;
        elPitch += size_t(idx3d.z) * m_elPitch.z;
        elPitch += size_t(component) * m_elPitch.w;

        return elPitch;
    }

    size_t idx(int x, int y, int z, int component) const
    {
        size_t elPitch = 0;
        elPitch += size_t(x + m_halo.x) * m_elPitch.x;
        elPitch += size_t(y + m_halo.y) * m_elPitch.y;
        elPitch += size_t(z + m_halo.z) * m_elPitch.z;
        elPitch += size_t(component) * m_elPitch.w;

        return elPitch;
    }

    size_t idxInHaloedSpace(int x, int y, int z, int component) const
    {
        size_t elPitch = 0;
        elPitch += size_t(x) * m_elPitch.x;
        elPitch += size_t(y) * m_elPitch.y;
        elPitch += size_t(z) * m_elPitch.z;
        elPitch += size_t(component) * m_elPitch.w;

        return elPitch;
    }

    const index_3d& dim() const
    {
        return m_dim;
    }

    const index_3d& haloedDim() const
    {
        return m_haloedDim;
    }

    const index_3d& halo() const
    {
        return m_halo;
    }

    const MemAlignment& alignment() const
    {
        return m_alignment;
    }

    memLayout_et::padding_e padding() const
    {
        return m_padding;
    }

    memLayout_et::order_e layout() const
    {
        return m_layoutOrder;
    }


    /**
    * NOTE: Indexing_ta defines the indexing schema associated with idx.
    * If idx comes from setThrIdx it is already in HALO_EXTENDED indexing mode
    *
    * NOTE: we return an uint64 rather than size_t because we can support negative value for idx.
    *       This is useful when working with halo but still using the STANDARD indexing.
    * @tparam fieldComponentId_ta: number of component of the field. If scalar than is set to 1
    * @tparam Indexing_ta : indexing schema associated to idx
    * @param idx
    * @return
    */
    template <Neon::DataView Indexing_ta = Neon::DataView::STANDARD>
    inline int64_t elPitch(const index_3d& idx,
                           int             fieldComponentId_ta = 0) const
    {
        switch (Indexing_ta) {
            case Neon::DataView::STANDARD: {
                size_t ret = (idx.x + m_halo.x) * int64_t(m_elPitch.x) +
                             (idx.y + m_halo.y) * int64_t(m_elPitch.y) +
                             (idx.z + m_halo.z) * int64_t(m_elPitch.z) +
                             fieldComponentId_ta * int64_t(m_elPitch.w);
                return ret;
            }            
            case Neon::DataView::INTERNAL: {
                size_t ret = (idx.x + 2 * m_halo.x) * int64_t(m_elPitch.x) +
                             (idx.y + 2 * m_halo.y) * int64_t(m_elPitch.y) +
                             (idx.z + 2 * m_halo.z) * int64_t(m_elPitch.z) +
                             fieldComponentId_ta * int64_t(m_elPitch.w);
                return ret;
            }
        }
        NeonException exp("mem3d");
        exp << "Configuration not supported";
        NEON_THROW(exp);
    }

    /**
     * Returns a reference to the element in location x,y,z provided by the idx parameter    
     */
    template <Neon::DataView Indexing_ta = Neon::DataView::STANDARD>
    inline T_ta& elRef(const index_3d& idx, int fieldComponentId_ta = 0)
    {
        int64_t ret = elPitch<Indexing_ta>(idx, fieldComponentId_ta);
        return m_mem.mem()[ret];
    }

    /**
     * Returns a const reference to the element in location x,y,z provided by the idx parameter     
     */
    template <Neon::DataView Indexing_ta = Neon::DataView::STANDARD>
    inline const T_ta& elRef(const index_3d& idx, int fieldComponentId_ta = 0) const
    {
        int64_t ret = elPitch<Indexing_ta>(idx, fieldComponentId_ta);
        return m_mem.mem()[ret];
    }

    /**
     * Returns a pointer to the element in location x,y,z provided by the idx parameter     
     */
    template <Neon::DataView Indexing_ta = Neon::DataView::STANDARD>
    inline T_ta* elPtr(const index_3d& idx, int fieldComponentId_ta = 0)
    {
        int64_t ret = elPitch<Indexing_ta>(idx, fieldComponentId_ta);
        return m_mem.mem() + ret;
    }

    /**
     * Returns a const pointer to the element in location x,y,z provided by the idx parameter     
     */
    template <Neon::DataView Indexing_ta = Neon::DataView::STANDARD>
    inline const T_ta* elPtr(const index_3d& idx, int fieldComponentId_ta = 0) const
    {
        int64_t ret = elPitch<Indexing_ta>(idx, fieldComponentId_ta);
        return m_mem.mem() + ret;
    }
};

}  // namespace sys
}  // End of namespace Neon

#include "Neon/sys/memory/mem3d.ti.h"
