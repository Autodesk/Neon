#pragma once

#include <string>

#include "Neon/core/types/Allocator.h"
#include "Neon/core/types/memOptions.h"
#include "Neon/sys/devices/DevInterface.h"
#include "Neon/sys/memory/MemAlignment.h"

namespace Neon {
namespace set {
class DevSet;
}
namespace sys {


struct [[deprecated("This feature is going to be replaced by a new API for Neon 1.0")]] memConf_t
{
    friend set::DevSet;

   private:
    enum state_e
    {
        initDone,
        initToDo
    };


    state_e                       m_state = {initToDo};
    Neon::DeviceType              m_devEt = {Neon::DeviceType::NONE};
    Neon::Allocator               m_allocEt = {Neon::Allocator::NULL_MEM};
    Neon::memLayout_et::order_e   m_orderEt = {Neon::memLayout_et::order_e::structOfArrays};
    Neon::memLayout_et::padding_e m_paddingEt = {Neon::memLayout_et::padding_e::OFF};
    Neon::sys::MemAlignment       m_memAlignment;

   public:
    /**
     *
     */
    ~memConf_t() = default;

    /**
     *
     */
    memConf_t() = default;

    /**
     *
     * @param devEt
     */
    memConf_t(Neon::DeviceType devEt)
    {
        m_devEt = devEt;
        switch (m_devEt) {
            case Neon::DeviceType::CPU: {
                m_allocEt = Neon::Allocator::MALLOC;
                m_state = initDone;
                break;
            }
            case Neon::DeviceType::CUDA: {
                m_allocEt = Neon::Allocator::CUDA_MEM_DEVICE;
                m_state = initDone;
                break;
            }
            default: {
                NeonException exception("memConf_t");
                exception << "Option not supported. ";
                NEON_THROW(exception);
            }
        }
    }

    /**
     *
     * @param devEt
     * @param allocEt
     */
    memConf_t(Neon::DeviceType devEt,
              Neon::Allocator  allocEt)
    {
        m_devEt = devEt;
        m_allocEt = allocEt;
        m_state = initDone;
    }

    memConf_t(Neon::DeviceType            devEt,
              Neon::memLayout_et::order_e order)
    {
        m_devEt = devEt;
        m_orderEt = order;
        m_state = initDone;
    }

    /**
     *
     * @param devEt
     * @param allocEt
     * @param order
     */
    memConf_t(Neon::DeviceType            devEt,
              Neon::Allocator             allocEt,
              Neon::memLayout_et::order_e order)
    {
        m_devEt = devEt;
        m_allocEt = allocEt;
        m_orderEt = order;
        m_state = initDone;
    }

    /**
     *
     * @param devEt
     * @param allocEt
     * @param orderEt
     * @param alignment
     */
    memConf_t(Neon::DeviceType            devEt,
              Neon::Allocator             allocEt,
              Neon::memLayout_et::order_e orderEt,
              Neon::sys::MemAlignment     alignment)
    {
        m_devEt = devEt;
        m_allocEt = allocEt;
        m_orderEt = orderEt;
        m_memAlignment = alignment;
        m_state = initDone;
    }

    /**
     *
     * @param devEt
     * @param allocEt
     * @param orderEt
     * @param alignment
     * @param paddingEt
     */
    memConf_t(Neon::DeviceType              devEt,
              Neon::Allocator               allocEt,
              Neon::memLayout_et::order_e   orderEt,
              Neon::sys::MemAlignment       alignment,
              Neon::memLayout_et::padding_e paddingEt)
    {
        m_devEt = devEt;
        m_allocEt = allocEt;
        m_orderEt = orderEt;
        m_memAlignment = alignment;
        m_paddingEt = paddingEt;
        m_state = initDone;
    }

    /**
     *
     * @return
     */
    Neon::DeviceType devEt() const
    {
        return m_devEt;
    }

    /**
     *
     * @return
     */
    Neon::Allocator allocEt() const
    {
        return m_allocEt;
    }

    /**
     *
     * @return
     */
    auto padding()->Neon::memLayout_et::padding_e&
    {
        return m_paddingEt;
    }

    /**
     *
     * @return
     */
    auto padding() const->const Neon::memLayout_et::padding_e&
    {
        return m_paddingEt;
    }

    /**
     *
     * @return
     */
    Neon::memLayout_et::order_e order() const
    {
        return m_orderEt;
    }

    /**
     *
     * @return
     */
    const Neon::sys::MemAlignment& alignment() const
    {
        return m_memAlignment;
    }
};

}  // namespace sys
}  // namespace Neon
