#pragma once

#include "Neon/set/Backend.h"
#include "Neon/set/BlockConfig.h"

namespace Neon {
namespace domain {

class LaunchConfigStorage_t;

class LaunchConfig_t
{

   private:
    const Neon::Backend*            m_backend{nullptr};
    const Neon::set::BlockConfig* m_blockConfig{&Neon::set::BlockConfig::defaultConfig()};
    Neon::DataView                  m_indexing{Neon::DataView::STANDARD};
    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------
   public:
    /**
     * Empty constructor
     */
    LaunchConfig_t() = default;

    /**
     *
     */
    LaunchConfig_t(const LaunchConfigStorage_t&);

    /**
     *
     */
    LaunchConfig_t(const LaunchConfigStorage_t&, Neon::DataView);

    /**
     *
     * @param backend
     */
    LaunchConfig_t(const Neon::Backend& backend);

    /**
     *
     * @param blockSize
     */
    LaunchConfig_t(const Neon::Backend&            backend,
                   const Neon::set::BlockConfig& blockConfig);

    /**
     *
     * @param backend
     */
    LaunchConfig_t(const Neon::Backend& backend,
                   Neon::DataView);

    /**
     *
     * @param blockSize
     */
    LaunchConfig_t(const Neon::Backend&            backend,
                   Neon::DataView                  indexing,
                   const Neon::set::BlockConfig& blockConfig);

    /**
     *
     * @return
     */
    auto backend() const -> const Neon::Backend&;

    /**
     *
     * @return
     */
    auto blockConfig() const -> const Neon::set::BlockConfig&;

    auto indexing() const -> const Neon::DataView&;
};


class LaunchConfigStorage_t
{
   private:
    Neon::Backend            m_backend{};
    Neon::set::BlockConfig   m_blockConfig{};
    LaunchConfig_t           m_launchConfig{m_backend, m_blockConfig};
    Neon::DataView           m_indexing{Neon::DataView::STANDARD};

   public:
    LaunchConfigStorage_t() = default;
    /**
     *
     * @param backend
     */
    LaunchConfigStorage_t(const Neon::Backend& backend);

    /**
     *
     * @param blockSize
     */
    LaunchConfigStorage_t(const Neon::Backend& backend, const Neon::set::BlockConfig& blockConfig);

    /**
     *
     * @param backend
     */
    LaunchConfigStorage_t(const Neon::Backend& backend,
                          Neon::DataView       indexing);

    /**
     *
     * @param blockSize
     */
    LaunchConfigStorage_t(const Neon::Backend&            backend,
                          Neon::DataView                  indexing,
                          const Neon::set::BlockConfig& blockConfig);

    /**
     *
     * @return
     */
    auto backend() const -> const Neon::Backend&;

    /**
     *
     * @return
     */
    auto indexing() const -> const Neon::DataView&;

    /**
     *
     * @return
     */
    auto blockConfig() const -> const Neon::set::BlockConfig&;

    /**
     *
     * @return
     */
    auto LaunchConfig() const -> const LaunchConfig_t&;
};

}  // namespace grids
}  // namespace Neon
