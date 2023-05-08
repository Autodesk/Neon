#pragma once

#include <memory>
#include <vector>

#include "Neon/set/DataSet.h"
#include "Neon/set/GpuStreamSet.h"
#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/devices/gpu/GpuSys.h"
#include "Neon/sys/memory/MemDevice.h"

namespace Neon {
namespace set {

template <typename T_ta = char>
class MemDevSet
{
    friend class DevSet;

    // ALIAS
   public:
    using self_t = MemDevSet<T_ta>;
    using eType_t = T_ta;
    using mem_t = Neon::sys::MemDevice<T_ta>;
    using local_t = typename mem_t::Partition;


   private:
    std::shared_ptr<Neon::set::DataSet<Neon::sys::MemDevice<T_ta>>> m_storage;

    inline auto vecRef() -> Neon::set::DataSet<Neon::sys::MemDevice<T_ta>>&
    {
        return *(m_storage.get());
    }

    inline auto vecRef() const -> const Neon::set::DataSet<Neon::sys::MemDevice<T_ta>>&
    {
        return *(m_storage.get());
    }

   private:
    /**
     * This private constructor should be used by a DevSet
     * */
    MemDevSet(Neon::DeviceType                               devType,
              const Neon::set::DataSet<Neon::sys::DeviceID>& devId,
              const Neon::Allocator&&                        allocType,
              uint64_t                                       nElements);
    /**
     * This private constructor should be used by a DevSet
     * */
    MemDevSet(Neon::DeviceType                               devType,
              const Neon::set::DataSet<Neon::sys::DeviceID>& devId,
              const Neon::Allocator&                         allocType,
              const Neon::set::DataSet<uint64_t>&            nElementVec);

    MemDevSet(int                                            cardinality,
              Neon::MemoryLayout                             order,
              Neon::DeviceType                               devType,
              const Neon::set::DataSet<Neon::sys::DeviceID>& devId,
              Neon::Allocator                                allocType,
              const Neon::set::DataSet<uint64_t>&            nElementVec);

   public:
    /**
     * An empty constructor. No resource allocation is done
     * */
    MemDevSet() = default;

    /**
     * Initialize the object to hold numEntries object of type Neon::sys::Mem_t. No resource allocation is done
     * */
    MemDevSet(int numEntries);

    /**
     * A constructor to initialize a MemDevSet object with user provided memory buffers
     * */
    MemDevSet(Neon::DeviceType devType, Neon::sys::DeviceID devId, Neon::set::DataSet<size_t> sizes, Neon::set::DataSet<char*> addrs);

    /**
     * A constructor to initialize a MemDevSet object with user provided memory buffers
     * */
    MemDevSet(Neon::DeviceType devType, Neon::set::DataSet<size_t> sizes, Neon::set::DataSet<char*> addrs);

    /**
     * A constructor to initialize a MemDevSet object with user Neon::sys::Mem_t buffers
     * */
    MemDevSet(Neon::set::DataSet<Neon::sys::MemDevice<T_ta>> memVec);

    /**
     * Set the i-th buffers with a Neon::sys::Mem_t buffers provided by the user
     * If a previous buffer was initialized for the i-th entry, it is destroyed
     * and if no reference are left it is automatically garbaged collected.
     * */
    void set(SetIdx id, Neon::sys::MemDevice<T_ta>& mem);

    /**
     * Returns a reference to the i-th Neon::sys::Mem_t object
     * */
    template <Neon::Access e = Neon::Access::read>
    std::enable_if_t<e == Neon::Access::read, const Neon::sys::MemDevice<T_ta>&>
    get(SetIdx id) const;

    /**
     * Returns a const reference to the i-th Neon::sys::Mem_t object
     * */
    template <Neon::Access e = Neon::Access::read>
    std::enable_if_t<e == Neon::Access::readWrite, Neon::sys::MemDevice<T_ta>&>
    get(SetIdx id);

    /**
     * Returning a data set with the pointers to managed buffers.
     * @tparam T_ta
     * @return
     */
    Neon::set::DataSet<T_ta*> memSet() const
    {
        Neon::set::DataSet<T_ta*> dataSet(vecRef().size());
        for (size_t idx = 0; idx < vecRef().size(); idx++) {
            dataSet[idx] = this->mem(idx);
        }
        return dataSet;
    }

    Neon::set::DataSet<T_ta*> memSet()
    {
        Neon::set::DataSet<T_ta*> dataSet(static_cast<int>(vecRef().size()));
        for (int idx = 0; idx < vecRef().size(); idx++) {
            dataSet[idx] = this->mem(static_cast<int32_t>(idx));
        }
        return dataSet;
    }

    /**
     * Return a reference to the i-th MemDev
     */
    auto getMemDev(int64_t idx) -> Neon::sys::MemDevice<T_ta>&
    {
        return vecRef()[idx];
    }

    /**
     * Return a const reference to the i-th MemDev
     */
    auto getMemDev(int64_t idx) const -> const Neon::sys::MemDevice<T_ta>&
    {
        return vecRef()[idx];
    }

    /**
     * Returns a pointer to the i-th buffers managed by this object.
     * @tparam T_ta: requested pointer type
     * @param id: id of the i-th buffer
     * @return: raw memory pointer
     */
    T_ta* mem(SetIdx id)
    {
        // return (T_ta*)(m_memVec[id.idx()].template mem<T_ta>());
        return vecRef()[id.idx()].mem();
    }

    /**
     * Returns a const pointer to the i-th buffers managed by this object.
     * @tparam T_ta: requested pointer type
     * @param id: id of the i-th buffer
     * @return: raw memory pointer
     */
    const T_ta* mem(SetIdx id) const
    {
        return vecRef()[id.idx()].mem();
    }

    /**
     * Release all the memory allocated for the set.
     * The buffers that were provided by the users are not deallocated.
     */
    void release();

    /**
     * Returns the number of Neon::sys::Mem_t object stored in the set
     */
    int32_t setCardinality() const;


    /**
     * Returns what type of management policy apply to the memory buffers:
     * 1. user: no de-allocation is done by this object
     * 2. system: this object is in charge of release the allocated memory
     * @return
     */
    managedMode_t managedMode(SetIdx idx) const;

    /**
     * Returns the type of the device that owns the allocated memory
     * @return
     */
    const Neon::DeviceType& devType(SetIdx idx) const;

    /**
     * Returns the ID of the device that owns the allocated memory
     * @return
     */
    const Neon::sys::DeviceID& devId(SetIdx idx) const;

    /**
     * Returns an aggregated type of memory of the allocated memory buffers.
     * If different buffers features different memory types than a MIXED_MEM type
     * is returned.
     * @return
     */
    auto allocType() const -> Neon::Allocator;


    /**
     * Returns size of the managed memory buffer.
     * @return
     */
    auto size(SetIdx idx) const -> size_t;

    /**
     * Returns the number of object of type T_ta that can be allocated
     * in the memory buffer
     * @return
     */
    size_t count(SetIdx idx) const
    {
        return vecRef()[idx.idx()].nElements();
    }

    /**
     *
     * @param memSet
     */
    void copyFrom(const MemDevSet& memSet)
    {
        if (memSet.setCardinality() != this->setCardinality()) {
            Neon::NeonException exp("MemDevSet");
            exp << " Can not copy a MemDevSet of setCardinality "
                << this->setCardinality()
                << " from an object of setCardinality "
                << memSet.setCardinality();
            NEON_THROW(exp);
        }
#pragma omp parallel for num_threads(static_cast<int>(this->setCardinality()))
        for (int i = 0; i < this->setCardinality(); ++i) {
            vecRef()[i].copyFrom(memSet.get(i));
        }
    }

    /**
     *
     * @tparam runMode_ta
     * @param gpuStream
     * @param memSet
     */
    template <Neon::run_et::et runMode_ta>
    void updateFrom(const Neon::set::StreamSet& gpuStream,
                    const MemDevSet<T_ta>&      memSet)
    {
        if (memSet.setCardinality() != this->setCardinality()) {
            Neon::NeonException exp("MemDevSet");
            exp << " Can not update a MemDevSet of setCardinality "
                << this->setCardinality()
                << " from an object of setCardinality "
                << memSet.setCardinality();
            NEON_THROW(exp);
        }
        const int setCardinality = static_cast<int>(this->setCardinality());
#pragma omp parallel for num_threads(setCardinality)
        for (int i = 0; i < setCardinality; ++i) {
            vecRef()[i].template updateFrom<runMode_ta>(gpuStream[i],
                                                        memSet.get(i));
        }
    }

    auto celRef(SetIdx setIdx, index64_t eIdx, int cardinality) const -> const T_ta&
    {
        return this->template get<Neon::Access::read>(setIdx).template elRef<Neon::Access::read>(eIdx, cardinality);
    }

    auto elRef(SetIdx setIdx, index64_t eIdx, int cardinality) const -> const T_ta&
    {
        return this->template get<Neon::Access::read>(setIdx).template elRef<Neon::Access::read>(eIdx, cardinality);
    }

    auto elRef(SetIdx setIdx, index64_t eIdx, int cardinality) -> T_ta&
    {
        return this->template get<Neon::Access::readWrite>(setIdx).template elRef<Neon::Access::readWrite>(eIdx, cardinality);
    }
};


}  // namespace set
}  // End of namespace Neon

#include "Neon/set/memory/memDevSet.ti.h"
