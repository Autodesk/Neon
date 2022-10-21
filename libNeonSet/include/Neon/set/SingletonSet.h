#pragma once
#include <vector>

#include "Neon/sys/devices/DevInterface.h"

namespace Neon {
namespace set {

template <typename T_ta>
struct SingletonSet
{
   private:
    std::shared_ptr<T_ta> m_data;
    /**
     * Returns a mutable reference to the data hold by the object
     */
    auto h_val()
        -> T_ta&
    {
        return *(m_data.get());
    }

    /**
     * Returns a constant reference to the data hold by the object
     */
    auto h_val()
        const
        -> const T_ta&
    {
        return *(m_data.get());
    }

   public:
    using self_t = SingletonSet<T_ta>;
    using global_t = self_t;
    using local_t = T_ta;
    using element_t = T_ta;

    /**
     *
     */
    SingletonSet()
    {
        m_data = std::make_shared<T_ta>();
    };

    auto self() -> SingletonSet&
    {
        return *this;
    }

    auto self() const -> const SingletonSet&
    {
        return *this;
    }
    /**
     *
     */
    SingletonSet(const SingletonSet&) = default;

    /**
     *
     * @param nDev
     * @param val
     */
    SingletonSet(const T_ta& val)
    {
        m_data = std::make_shared<T_ta>(val);
    }

    /**
     * Returns a unique identifier for the data set.
     * @return
     */
    auto uid() ->Neon::set::dataDependency::MultiXpuDataUid
    {
        T_ta*                addr = m_data.get();
       Neon::set::dataDependency::MultiXpuDataUid uidRes = (size_t)addr;
        return uidRes;
    }

    /**
     * Returns a unique identifier for the data set.
     * @return
     */
    auto uid() const ->Neon::set::dataDependency::MultiXpuDataUid
    {
        T_ta*                addr = m_data.get();
       Neon::set::dataDependency::MultiXpuDataUid uidRes = (size_t)addr;
        return uidRes;
    }

    /**
     * Setting the value stored by the singleton
     * @param val
     */
    auto setValue(const T_ta val) -> void
    {
        h_val() = val;
    }

    /**
     * Returning the value stored by the singleton
     * @return
     */
    auto getValue() const -> const T_ta&
    {
        return h_val();
    }


    /**
     * Returning the value stored by the singleton
     * @return
     */
    auto getValue() -> const T_ta&
    {
        return h_val();
    }

    /**
     *
     * @return
     */
    auto compute() const -> const SingletonSet&
    {
        return *this;
    }

    /**
     *
     * @return
     */
    auto local([[maybe_unused]] const Neon::DataView& dataView = Neon::DataView::STANDARD) -> SingletonSet&
    {
        return *this;
    }

    /**
     *
     * @return
     */
    auto local([[maybe_unused]] Neon::DeviceType,
               [[maybe_unused]] SetIdx,
               [[maybe_unused]] const Neon::DataView& dataView = Neon::DataView::STANDARD) -> T_ta&
    {
        return h_val();
    }

    /**
     *
     * @return
     */
    auto local([[maybe_unused]] Neon::DeviceType, SetIdx, [[maybe_unused]] const Neon::DataView& dataView = Neon::DataView::STANDARD) const -> const T_ta&
    {
        return h_val();
    }

    /**
     *
     * @param idx
     * @return
     */
    auto operator[](int64_t) const -> const T_ta&
    {
        return h_val();
    }

    /**
     *
     * @param idx
     * @return
     */
    auto operator[](int64_t) -> T_ta&
    {
        return h_val();
    }

    template <typename newType_ta>
    auto newType() const -> SingletonSet<newType_ta>
    {
        newType_ta               newVal = newType_ta(h_val());
        SingletonSet<newType_ta> newSingletonSet(newVal);
        return newSingletonSet;
    }
};


}  // namespace set
}  // namespace Neon
