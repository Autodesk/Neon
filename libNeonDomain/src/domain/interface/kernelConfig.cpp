#include "Neon/domain/interface/KernelConfig.h"

#include <assert.h>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>


namespace Neon {
namespace domain {

KernelConfig::KernelConfig(int streamId)
{
    Neon::set::KernelConfig::expertSetStream(streamId);
}


KernelConfig::KernelConfig(int           streamId,
                           Neon::Runtime runtime)
{
    Neon::set::KernelConfig::expertSetStream(streamId);
    m_runtime = runtime;
}

KernelConfig::KernelConfig(int                           streamId,
                           Neon::Runtime                 runtime,
                           const Neon::set::BlockConfig& blockConfig)
{
    Neon::set::KernelConfig::expertSetStream(streamId);
    m_runtime = runtime;
    m_blockConfig = blockConfig;
}

KernelConfig::KernelConfig(int            streamId,
                           Neon::Runtime  runtime,
                           Neon::DataView dataView)
{
    Neon::set::KernelConfig::expertSetStream(streamId);
    m_runtime = runtime;
    Neon::set::KernelConfig::expertSetDataView(dataView);
}


KernelConfig::KernelConfig(int            streamId,
                           Neon::DataView dataView)
{
    Neon::set::KernelConfig::expertSetStream(streamId);
    Neon::set::KernelConfig::expertSetDataView(dataView);
}

/**
 *
 * @param blockSize
 */
KernelConfig::KernelConfig(int                           streamId,
                           Neon::Runtime                 runtime,
                           Neon::DataView                dataView,
                           const Neon::set::BlockConfig& blockConfig)
{
    Neon::set::KernelConfig::expertSetStream(streamId);
    m_runtime = runtime;
    Neon::set::KernelConfig::expertSetDataView(dataView);
    m_blockConfig = blockConfig;
}


auto KernelConfig::runtime() const -> Neon::Runtime
{
    return m_runtime;
}

auto KernelConfig::blockConfig() const -> const Neon::set::BlockConfig&
{
    return m_blockConfig;
}

}  // namespace domain
}  // namespace Neon
