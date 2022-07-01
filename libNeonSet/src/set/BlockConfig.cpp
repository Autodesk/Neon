#include "Neon/set/BlockConfig.h"

#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

namespace Neon {
namespace set {

BlockConfig::BlockConfig(std::function<size_t(const Neon::index_3d&)> f)
{
    m_shMem.set(f);
}

BlockConfig::BlockConfig(const Neon::index_3d& blockSize)
{
    m_blockMode = blockMode_e::user;
    m_blockSize = blockSize;
}

BlockConfig::BlockConfig(const Neon::index_3d& blockSize,
                             size_t                sharedMemSize)
{
    m_blockMode = blockMode_e::user;
    m_blockSize = blockSize;
    m_shMem.set(sharedMemSize);
}

auto BlockConfig::blockMode() const -> const blockMode_e&
{
    return m_blockMode;
}

auto BlockConfig::blockSize() const -> const Neon::index_3d&
{
    if (m_blockMode == user) {
        return m_blockSize;
    } else {
        NEON_THROW_UNSUPPORTED_OPERATION();
    }
}

auto BlockConfig::sharedMemory(const Neon::index_3d& block) const ->  size_t
{
    return m_shMem.size(block);
}

auto BlockConfig::setBlockSize(Neon::index_3d blockSize, size_t sharedMemSize) -> void
{
    m_blockMode = user;
    m_blockSize = blockSize;
    m_shMem.set(sharedMemSize);
}

namespace globalDefaults {

BlockConfig g_blockConfig;

}  // namespace globalDefaults

auto BlockConfig::defaultConfig() -> const BlockConfig&
{
    return globalDefaults::g_blockConfig;
}


}  // namespace set
}  // namespace Neon
