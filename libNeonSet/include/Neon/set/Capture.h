#pragma once
#include <functional>
#include "Neon/core/types/DataView.h"

namespace Neon {
namespace set {

#if NEON_WORK_IN_PROGRESS

/**
 * Mechanism to convert global fields into local when capturing fields in kernels.
 */
struct Capture_t
{
    friend Backend;

   private:
    Neon::dev_et::enum_e m_devType;  /** type of device */
    int                  m_setIdx;   /** device id */
    Neon::DataView       m_dataView; /** data view */

   public:
    /**
     * Internal constructor
     * TODO: conver to private
     * @param devType
     * @param setIdx
     * @param dataView
     */
    Capture_t(Neon::dev_et::enum_e devType, int setIdx, Neon::DataView dataView)
    {
        m_devType = devType;
        m_setIdx = setIdx;
        m_dataView = dataView;
    }

    /**
     * operator for the capture action. Enabled only for mutable fields
     * @tparam Field_t
     * @param field
     * @return
     */
    template <typename Field_t>
    auto
    operator()(Field_t& field) const -> std::enable_if<!std::is_const_v<Field_t>,
                                                       typename Field_t::local_t&>
    {
        return field.local(m_devType, m_setIdx, m_dataView);
    }

    /**
     * operator for the capture action. Enabled only for const fields
     * @tparam Field_t
     * @param field
     * @return
     */
    template <typename Field_t>
    auto operator()(const Field_t& field) const -> std::enable_if<std::is_const_v<Field_t>,
                                                                  const typename Field_t::local_t&>
    {
        return field.local(m_devType, m_setIdx, m_dataView);
    }
};
#endif
}  // namespace set
}  // namespace Neon