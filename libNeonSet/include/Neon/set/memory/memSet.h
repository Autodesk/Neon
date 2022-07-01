#pragma once

#include <Neon/set/Backend.h>
#include <vector>

#include "Neon/set/Backend.h"
#include "Neon/set/DataSet.h"
#include "Neon/set/GpuStreamSet.h"
#include "Neon/set/memory/memDevSet.h"
#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/devices/gpu/GpuStream.h"
#include "Neon/sys/memory/MemDevice.h"
#include "Neon/sys/memory/MemMirror.h"

namespace Neon {
namespace set {

/**
 * Class to store a set of MemMirror objects.
 *
 * THREAD SAFETY?
 * In general, any operation on this structure is not thread safe
 * Exception are operations working on specifics elements of the set.
 * As long as threads are operating on different sections, i.e. using different
 * set ids, the threads can work in parallel.
 *
 * @tparam T_ta
 */
template <typename T_ta>
class MemSet_t
{
    /**
     * Design Note:
     *
     * This class contains a vector of Neon:sys::MemMirror objects.
     * The vector is stored through a shared pointer so independently
     * of the copies created of this object only one vector exist.
     * This ensures us that any operation applied to one of this copies
     * is coherent.
     *
     */
   public:
    friend class DevSet;

    // using MemComputeSet_t = Neon::set::DataSet<Neon::sys::Memlocal_t<T_ta>>;
    using self_t = MemSet_t<T_ta>;
    using local_t = Neon::sys::Memlocal_t<T_ta>;
    using element_t = T_ta;

   private:
    // ALIAS
    using Mem_ta = Neon::sys::MemMirror<T_ta>;
    using Mem_vec = std::vector<Mem_ta>;
    using Mem_shp = std::shared_ptr<Mem_vec>;

   private:
    // MEMBERS
    Mem_shp m_memSet_shp; /**< this is a shared pinter to a vector of sys::MemMirror objects */

    /**
     * Private utility function to retrieve a reference from the shared pointer
     * @return
     */
    inline auto entryRef(SetIdx idx)
        -> Mem_ta&
    {
        checkId(idx);
        return m_memSet_shp.get()->at(idx.idx());
    }

    inline auto entryCRef(SetIdx idx)
        const
        -> const Mem_ta&
    {
        checkId(idx);
        return m_memSet_shp.get()->at(idx.idx());
    }

    /**
     * Private utility function to retrieve a reference from the shared pointer
     * @return
     */
    inline auto entryRef(SetIdx idx)
        const
        -> const Mem_ta&
    {
        checkId(idx);
        return m_memSet_shp.get()->at(idx.idx());
    }

    inline auto checkId(SetIdx id) const
    {
        if (size_t(id.idx()) >= size_t(this->cardinality())) {
            Neon::NeonException exception("MemSet_t");
            exception << "Incompatible set index was detected. Requested " << id.idx() << " but max is " << this->cardinality();
            NEON_THROW(exception);
        }
        return;
    }

   public:
    /**
     * Empty constructor. No resource allocation is done
     * */
    MemSet_t()
    {
        m_memSet_shp = std::make_shared<Mem_vec>();
    }

    /**
     * Initialize the object to hold numEntries object of type Neon::sys::Mem_t. No resource allocation is done
     * */
    MemSet_t(int numEntries)
    {
        m_memSet_shp = std::make_shared<Mem_vec>(numEntries);
    }

    auto uid() const -> Neon::set::MultiDeviceObjectUid
    {
        void*                           addr = static_cast<void*>(m_memSet_shp.get());
        Neon::set::MultiDeviceObjectUid uidRes = (size_t)addr;
        return uidRes;
    }

    /**
     * Link the provided Mem_t object with i-th mirror
     * @param id
     * @param mem
     */
    void link(SetIdx id, Neon::sys::MemDevice<T_ta>& mem)
    {
        entryRef(id).link(mem);
    }

    /**
     * Link the providded Mem_t object with i-th mirror
     * @param id
     * @param mem
     */
    void link(Neon::set::MemDevSet<T_ta>& mem)
    {
        int card = mem.setCardinality();
        for (int i = 0; i < card; i++) {
            this->link(i, mem.template get<Neon::Access::readWrite>(i));
        }
    }

    /**
     * Detach the Mem_t object of type devEt from the i-th mirror
     * @param id
     * @param devEt
     * @return
     */
    auto detach(SetIdx id, Neon::DeviceType devEt)
        -> Neon::sys::MemDevice<T_ta>
    {
        return entryRef(id).detach(devEt);
    }

    /**
     * Detach the i-th MemMirror_t object
     * @param id
     * @return
     */
    auto detach(SetIdx id)
        -> Neon::sys::MemMirror<T_ta>
    {
        Neon::sys::MemMirror<T_ta> tmp = entryRef(id);
        entryRef(id).reset();
        return tmp;
    }

    /**
     * Reset the i-th MemMirror_t object
     * @param id
     * @return
     */
    auto reset(SetIdx id)
        -> void
    {
        return entryRef(id).reset();
    }

    /**
     * Reset all the mirrors in the set
     */
    auto reset()
        -> void
    {
        for (int i = 0; i < cardinality(); i++) {
            reset(SetIdx(i));
        }
    }

    /**
     * Return the raw pointer of the i-th mirror for the devEt side
     * @param id
     * @return
     */
    auto rawMem(SetIdx id, Neon::DeviceType devEt) -> T_ta*
    {
        return entryRef(id).rawMem(devEt);
    }

    auto rawMem(SetIdx id, Neon::DeviceType devEt) const -> const T_ta*
    {
        return entryRef(id).rawMem(devEt);
    }

    auto rawMem(Neon::Execution execution, SetIdx id) -> T_ta*
    {
        return entryRef(id).rawMem(execution);
    }

    /**
     * Returns a dataSet with the raw pointers, one entry for each mirror.
     */
    auto rawMem(Neon::DeviceType devEt)
        -> Neon::set::DataSet<T_ta*>
    {
        Neon::set::DataSet<T_ta*> res(cardinality());
        for (int i = 0; i < cardinality(); i++) {
            res[i] = entryRef(i).rawMem(devEt);
        }
        return res;
    }

    /**
     * Returns reference to the i-th mirror
     * @param id
     * @return auto
     */
    auto get(SetIdx id)
        const
        -> const Neon::sys::MemMirror<T_ta>&
    {
        return entryRef(id);
    }

    /**
     * Returns reference to the i-th mirror
     * @param id
     * @return auto
     */
    auto get(SetIdx id)
        -> Neon::sys::MemMirror<T_ta>&
    {
        return entryRef(id);
    }

    /**
     * Returns the number of Neon::sys::Mem_t object stored in the set
     */
    auto cardinality()
        const
        -> int32_t
    {
        return int(m_memSet_shp->size());
    }

    /**
     * For any mirror in the set it updates one of the two side from the value of the other.
     * For example it copies all the content on the CPU mirrors on the GPU side
     * @tparam runMode_ta
     * @param devEt
     * @param gpuStreamSet
     */
    template <Neon::run_et::et runMode_ta>
    void update(const Neon::set::StreamSet& gpuStreamSet,
                Neon::DeviceType            devEt)
    {
        int32_t nDev = cardinality();
#pragma omp parallel for num_threads(nDev) default(none) shared(nDev, devEt, gpuStreamSet)
        for (int setIdx = 0; setIdx < nDev; setIdx++) {
            entryRef(setIdx).template update<Neon::run_et::async>(devEt, gpuStreamSet[setIdx]);
        }
        gpuStreamSet.template sync<runMode_ta>();
    }


    void updateCompute(Neon::Backend   bk,
                       Neon::StreamIdx streamId)
    {
        int32_t                     nDev = cardinality();
        const Neon::set::StreamSet& gpuStreamSet = bk.streamSet(streamId);
        auto                        devEt = Neon::DeviceType::CUDA;
#pragma omp parallel for num_threads(nDev) default(none) shared(nDev, devEt, gpuStreamSet)
        for (int setIdx = 0; setIdx < nDev; setIdx++) {
            entryRef(setIdx).template update<Neon::run_et::async>(devEt, gpuStreamSet[setIdx]);
        }
    }

    void updateIO(Neon::Backend   bk,
                  Neon::StreamIdx streamId)
    {
        int32_t                     nDev = cardinality();
        const Neon::set::StreamSet& gpuStreamSet = bk.streamSet(streamId);
        auto                        devEt = Neon::DeviceType::CPU;
#pragma omp parallel for num_threads(nDev) default(none) shared(nDev, devEt, gpuStreamSet)
        for (int setIdx = 0; setIdx < nDev; setIdx++) {
            entryRef(setIdx).template update<Neon::run_et::async>(devEt, gpuStreamSet[setIdx]);
        }
    }
    /**
     * For any mirror in the set it updates one of the two side from the value of the other.
     * For example it copies all the content on the CPU mirrors on the GPU side
     * @tparam runMode_ta
     * @param devEtc_vect
     * @param gpuStreamSet
     */
    template <Neon::run_et::et runMode_ta>
    void update(Neon::set::StreamSet& gpuStreamSet,
                Neon::DeviceType      devEt,
                SetIdx                id)
    {
        entryRef(id).template update<Neon::run_et::async>(devEt, gpuStreamSet[id]);
        gpuStreamSet.template sync<runMode_ta>();
    }

    auto local(Neon::DeviceType devEt, Neon::SetIdx setIdx, const Neon::DataView& unused = Neon::DataView::STANDARD)
        const
        -> const typename Neon::sys::MemDevice<T_ta>::Partition&
    {
        (void)unused;
        return entryCRef(setIdx).compute(devEt);
    }

    auto local(Neon::DeviceType devEt, Neon::SetIdx setIdx, const Neon::DataView& unused = Neon::DataView::STANDARD)
        -> typename Neon::sys::MemDevice<T_ta>::Partition&
    {
        (void)unused;
        return entryRef(setIdx).compute(devEt);
    }

    auto eRef(Neon::SetIdx setIdx, int64_t idx, int cardinality = 0) -> T_ta&
    {
        self_t::local_t c = local(Neon::DeviceType::CPU, setIdx);
        return c.eRef(idx, cardinality);
    }

    auto eRef(Neon::SetIdx setIdx, int64_t idx, int cardinality = 0) const -> const T_ta&
    {
        self_t::local_t c = local(Neon::DeviceType::CPU, setIdx);
        return c.eRef(idx, cardinality);
    }

    auto eVal(Neon::SetIdx setIdx, int64_t idx, int cardinality = 0) const -> T_ta
    {
        self_t::local_t c = local(Neon::DeviceType::CPU, setIdx);
        return c.eRef(idx, cardinality);
    }
};

}  // namespace set
}  // namespace Neon
