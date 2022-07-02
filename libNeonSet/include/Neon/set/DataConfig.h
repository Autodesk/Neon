#pragma once

#include "Neon/core/types/DataUse.h"
#include "Neon/set/Backend.h"
#include "Neon/sys/memory/memConf.h"

namespace Neon {
namespace set {

/**
 *
 */
class DataConfig
{
   private:
    Neon::DataUse        m_dataUseE{Neon::DataUse::IO_COMPUTE};
    Neon::Backend        m_backendConfig{};
    Neon::sys::memConf_t m_cpuMemConf{Neon::DeviceType::CPU};
    Neon::sys::memConf_t m_gpuMemConf{Neon::DeviceType::CUDA};

   public:
    static Neon::sys::memConf_t g_cpuNullMemConf;
    static Neon::sys::memConf_t g_gpuNullMemConf;

   public:
    DataConfig() = default;
    DataConfig(const Neon::Backend& bk, Neon::DataUse);
    DataConfig(const Neon::Backend&        bk,
                 Neon::DataUse               dataUse,
                 const Neon::sys::memConf_t& cpu,
                 const Neon::sys::memConf_t& gpu);

    /**
     *
     * @param devType
     * @return
     */
    auto memConfig(Neon::DeviceType devType)
        const
        -> const Neon::sys::memConf_t&;

    /**
     *
     * @return
     */
    auto dataUse()
        const
        -> const Neon::DataUse&;

    /**
     *
     * @return
     */
    auto backendConfig()
        const
        -> const Neon::Backend&;

    /**
     *
     * @param devE
     * @return
     */
    auto isEnabled(Neon::DeviceType devE) const -> bool;
};

}  // namespace set
}  // namespace Neon
