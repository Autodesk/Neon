#include "Neon/set/DataConfig.h"

namespace Neon {
namespace set {

Neon::sys::memConf_t DataConfig::g_cpuNullMemConf = Neon::sys::memConf_t{Neon::DeviceType::CPU, Neon::Allocator::NULL_MEM};
Neon::sys::memConf_t DataConfig::g_gpuNullMemConf = Neon::sys::memConf_t{Neon::DeviceType::CUDA, Neon::Allocator::NULL_MEM};


DataConfig::DataConfig(const Neon::Backend& bk,
                           Neon::DataUse        dataUse)
{
    m_backendConfig = bk;
    m_dataUseE = dataUse;
}


DataConfig::DataConfig(const Neon::Backend&        bk,
                           Neon::DataUse               dataUse,
                           const Neon::sys::memConf_t& cpu,
                           const Neon::sys::memConf_t& gpu)
{
    m_backendConfig = bk;
    m_dataUseE = dataUse;
    m_cpuMemConf = cpu;
    m_gpuMemConf = gpu;
}


auto DataConfig::memConfig(Neon::DeviceType devType)
    const
    -> const Neon::sys::memConf_t&
{

    switch (devType) {
        case Neon::DeviceType::CPU: {
            if (!isEnabled(devType)) {
                return DataConfig::g_cpuNullMemConf;
            }
            return m_cpuMemConf;
        }
        case Neon::DeviceType::CUDA: {
            if (!isEnabled(devType)) {
                return DataConfig::g_gpuNullMemConf;
            }
            return m_gpuMemConf;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("DataConfig_t");
        }
    }
}


auto DataConfig::dataUse()
    const
    -> const Neon::DataUse&
{
    return m_dataUseE;
}


auto DataConfig::backendConfig()
    const
    -> const Neon::Backend&
{
    return m_backendConfig;
}


auto DataConfig::isEnabled(Neon::DeviceType devE) const -> bool
{
    switch (devE) {
        case Neon::DeviceType::CPU: {
            if (m_backendConfig.devType() == devE || this->dataUse() == Neon::DataUse::IO_COMPUTE) {
                return true;
            }
            return false;
        }
        case Neon::DeviceType::CUDA: {
            if (m_backendConfig.devType() == devE) {
                return true;
            }
            return false;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
}

}  // namespace set
}  // namespace Neon
