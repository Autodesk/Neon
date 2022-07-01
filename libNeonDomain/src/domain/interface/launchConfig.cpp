#include "Neon/domain/interface/LaunchConfig.h"

#include <assert.h>
#include <functional>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>


namespace Neon {
namespace domain {

LaunchConfig_t::LaunchConfig_t(const Neon::Backend& backend)
{
    m_backend = &backend;
}

LaunchConfig_t::LaunchConfig_t(const LaunchConfigStorage_t& storage)
{
    m_backend = &storage.backend();
    m_blockConfig = &storage.blockConfig();
    m_indexing = storage.indexing();
}

LaunchConfig_t::LaunchConfig_t(const LaunchConfigStorage_t& storage,
                               Neon::DataView               indexing)
{
    m_backend = &storage.backend();
    m_blockConfig = &storage.blockConfig();
    m_indexing = indexing;
}


LaunchConfig_t::LaunchConfig_t(const Neon::Backend&     backend,
                               const Neon::set::BlockConfig& blockConfig)
{
    m_backend = &backend;
    m_blockConfig = &blockConfig;
}

/**
 *
 * @param backend
 */
LaunchConfig_t::LaunchConfig_t(const Neon::Backend& backend,
                               Neon::DataView              indexing)
{
    m_backend = &backend;
    m_indexing = indexing;
}

/**
 *
 * @param blockSize
 */
LaunchConfig_t::LaunchConfig_t(const Neon::Backend&     backend,
                               Neon::DataView                  indexing,
                               const Neon::set::BlockConfig& blockConfig)
{
    m_backend = &backend;
    m_indexing = indexing;
    m_blockConfig = &blockConfig;
}


auto LaunchConfig_t::backend() const -> const Neon::Backend&
{
    if (m_blockConfig != nullptr) {
        return *m_backend;
    } else {
        assert(m_backend != nullptr);
        return *m_backend;
    }
}

auto LaunchConfig_t::blockConfig() const -> const Neon::set::BlockConfig&
{
    if (m_blockConfig != nullptr) {
        return *m_blockConfig;
    } else {
        return Neon::set::BlockConfig::defaultConfig();
    }
}

auto LaunchConfig_t::indexing() const -> const Neon::DataView&
{
    return m_indexing;
}


LaunchConfigStorage_t::LaunchConfigStorage_t(const Neon::Backend& backend)
{
    m_backend = backend;
}

LaunchConfigStorage_t::LaunchConfigStorage_t(const Neon::Backend& backend, const Neon::set::BlockConfig& blockConfig)
{
    m_backend = backend;
    m_blockConfig = blockConfig;
}

LaunchConfigStorage_t::LaunchConfigStorage_t(const Neon::Backend& backend,
                                             Neon::DataView              indexing)
{
    m_backend = backend;
    m_indexing = indexing;
}

LaunchConfigStorage_t::LaunchConfigStorage_t(const Neon::Backend&     backend,
                                             Neon::DataView                  indexing,
                                             const Neon::set::BlockConfig& blockConfig)
{
    m_backend = backend;
    m_indexing = indexing;
    m_blockConfig = blockConfig;
}

auto LaunchConfigStorage_t::backend() const -> const Neon::Backend&
{
    return m_backend;
}

auto LaunchConfigStorage_t::indexing() const -> const Neon::DataView&
{
    return m_indexing;
}

auto LaunchConfigStorage_t::blockConfig() const -> const Neon::set::BlockConfig&
{
    return m_blockConfig;
}

auto LaunchConfigStorage_t::LaunchConfig() const -> const LaunchConfig_t&
{
    return m_launchConfig;
}

}  // namespace grids
}  // namespace Neon
