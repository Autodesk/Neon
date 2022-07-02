#pragma once


#define NEON_MEM_SET_CHECK_ID(ID)                                                                   \
    size_t vlen = vecRef().size();                                                                  \
    if (int64_t(ID) < 0 || uint64_t(ID) >= vlen) {                                                  \
        Neon::NeonException exp("MemDevSet");                                                       \
        exp << "an incorrect id has been detected (" << (ID) << " in range [0," << vlen - 1 << "]"; \
        NEON_THROW(exp);                                                                            \
    }

namespace Neon {
namespace set {

template <typename T_ta>
MemDevSet<T_ta>::MemDevSet(Neon::DeviceType                               devType,
                           const Neon::set::DataSet<Neon::sys::DeviceID>& devIdx,
                           const Neon::Allocator&&                        allocType,
                           uint64_t                                       nElements,
                           Neon::sys::MemAlignment                        alignment)
{
    m_storage = std::make_shared<Neon::set::DataSet<MemDevSet<T_ta>>>(devIdx.size());

    // size_t vlen = m_memVec.size();
    if (devIdx.empty()) {
        Neon::NeonException exp("MemDevSet");
        exp << "an incorrect vector size has been detected. A empty vector has been received";
        NEON_THROW(exp);
    }


    // we add elements to a temporary vector:
    // If any issue happens on the main loop, tmpMemVec is cleaned up
    // and all memory released.
    int setIdx = 0;
    for (auto&& id : devIdx) {
        vecRef()[setIdx] = Neon::sys::MemDevice<T_ta>(devType, id, allocType, nElements, alignment);
        setIdx++;
    }
}

template <typename T_ta>
MemDevSet<T_ta>::MemDevSet(Neon::DeviceType                               devType,
                           const Neon::set::DataSet<Neon::sys::DeviceID>& devIdx,
                           const Neon::Allocator&                         allocType,
                           const Neon::set::DataSet<uint64_t>&            nElementVec,
                           Neon::sys::MemAlignment                        alignment)
{
    m_storage = std::make_shared<Neon::set::DataSet<Neon::sys::MemDevice<T_ta>>>(static_cast<int>(devIdx.size()));

    int setIdx = 0;
    for (auto&& id : devIdx) {
        vecRef()[setIdx] = Neon::sys::MemDevice<T_ta>(devType, id, allocType, nElementVec[setIdx], alignment);
        setIdx++;
    }
}

template <typename T_ta>
MemDevSet<T_ta>::MemDevSet(int                                            cardinality,
                           Neon::memLayout_et::order_e                    order,
                           Neon::memLayout_et::padding_e                  padding,
                           Neon::DeviceType                               devType,
                           const Neon::set::DataSet<Neon::sys::DeviceID>& devIds,
                           Neon::Allocator                                allocType,
                           const Neon::set::DataSet<uint64_t>&            nElementVec,
                           Neon::sys::MemAlignment                        alignment)
{
    m_storage = std::make_shared<Neon::set::DataSet<Neon::sys::MemDevice<T_ta>>>(static_cast<int>(devIds.size()));
    int setIdx = 0;

    for (auto&& id : devIds) {
        vecRef()[setIdx] = Neon::sys::MemDevice<T_ta>(cardinality, order, padding, devType, id, allocType, nElementVec[setIdx], alignment);
        setIdx++;
    }
}


template <typename T_ta>
MemDevSet<T_ta>::MemDevSet(int numEntries)
{ /* Nothing to be done here*/
    m_storage = std::make_shared<Neon::set::DataSet<Neon::sys::MemDevice<T_ta>>>(numEntries);
}


template <typename T_ta>
MemDevSet<T_ta>::MemDevSet(Neon::DeviceType devType, Neon::sys::DeviceID devIdx, Neon::set::DataSet<size_t> sizes, Neon::set::DataSet<char*> addrs)
{
    m_storage = std::make_shared<Neon::set::DataSet<Neon::sys::MemDevice<T_ta>>>(sizes.size());
    for (size_t id = 0; id < addrs.size(); id++) {
        Neon::sys::MemDevice<T_ta> tmp(devType, devIdx, sizes[id], addrs[id]);
        set(id, tmp);
    }
}

template <typename T_ta>
MemDevSet<T_ta>::MemDevSet(Neon::DeviceType devType, Neon::set::DataSet<size_t> sizes, Neon::set::DataSet<char*> addrs)
    : MemDevSet(devType, Neon::sys::DeviceID(0), sizes, addrs)
{
    /* Empty, we handover the initialization to another constructor */
}

template <typename T_ta>
MemDevSet<T_ta>::MemDevSet(Neon::set::DataSet<Neon::sys::MemDevice<T_ta>> memVec)
{
    vecRef() = memVec;
}

template <typename T_ta>
void MemDevSet<T_ta>::set(SetIdx id, Neon::sys::MemDevice<T_ta>& mem)
{
    NEON_MEM_SET_CHECK_ID(id.idx());
    vecRef()[id.idx()] = mem;
}

template <typename T_ta>
template <Neon::Access e>
std::enable_if_t<e == Neon::Access::read, const Neon::sys::MemDevice<T_ta>&>
MemDevSet<T_ta>::get(SetIdx id) const
{
    NEON_MEM_SET_CHECK_ID(id.idx());
    return vecRef()[id.idx()];
}

template <typename T_ta>
template <Neon::Access e>
std::enable_if_t<e == Neon::Access::readWrite, Neon::sys::MemDevice<T_ta>&>
MemDevSet<T_ta>::get(SetIdx id)
{
    NEON_MEM_SET_CHECK_ID(id.idx());
    return vecRef()[id.idx()];
}

template <typename T_ta>
void MemDevSet<T_ta>::release()
{
    vecRef() = Neon::set::DataSet<Neon::sys::MemDevice<T_ta>>(0);
}

template <typename T_ta>
int32_t MemDevSet<T_ta>::setCardinality() const
{
    return static_cast<int32_t>(vecRef().size());
}

template <typename T_ta>
managedMode_t MemDevSet<T_ta>::managedMode(SetIdx idx) const
{
    NEON_MEM_SET_CHECK_ID(idx.idx());
    return vecRef()[idx.idx()].managedMode();
}

template <typename T_ta>
const Neon::DeviceType& MemDevSet<T_ta>::devType(SetIdx idx) const
{
    NEON_MEM_SET_CHECK_ID(idx.idx());
    return vecRef()[idx.idx()].devType();
}

template <typename T_ta>
const Neon::sys::DeviceID& MemDevSet<T_ta>::devId(SetIdx idx) const
{
    NEON_MEM_SET_CHECK_ID(idx.idx());
    return vecRef()[idx.idx()].devIdx();
}

template <typename T_ta>
auto MemDevSet<T_ta>::allocType() const -> Neon::Allocator
{
    Neon::Allocator aggregatedType;
    aggregatedType = vecRef()[0].allocType();
    for (int idx = 0; idx < vecRef().size(); idx++) {
        Neon::Allocator ith = vecRef()[idx].allocType();
        if (aggregatedType != ith) {
            return Neon::Allocator::MIXED_MEM;
        }
    }
    return aggregatedType;
}

template <typename T_ta>
auto MemDevSet<T_ta>::size(SetIdx idx) const -> size_t
{
    NEON_MEM_SET_CHECK_ID(idx.idx());
    return vecRef()[idx.idx()].size();
}


}  // namespace set
}  // End of namespace Neon
