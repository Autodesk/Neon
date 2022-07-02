#pragma once

#include "Neon/set/Backend.h"
#include "Neon/set/BlockConfig.h"
#include "Neon/set/DevSet.h"

namespace Neon {
namespace domain {

/**
 * Class storing parametric aspects of running a kernel in a grid.
 * The standard user can use the LaunchConfig_t::system() to chose the default configurations
 */
class KernelConfig : public Neon::set::KernelConfig
{
   private:
    Neon::Runtime          m_runtime{Neon::Runtime::system};
    Neon::set::BlockConfig m_blockConfig{Neon::set::BlockConfig::defaultConfig()};

   public:
    virtual ~KernelConfig() = default;

    /**
     * Empty constructor
     */
    KernelConfig() = default;

    /**
     * Only the streamId is configured explicitally.
     * All other options are set to default.
     * @param streamId
     */
    KernelConfig(int streamId);

    /**
     *
     * @param backend
     */
    KernelConfig(int streamId, Neon::Runtime);

    /**
     *
     * @param blockSize
     */
    KernelConfig(int                           streamId,
                 Neon::Runtime                 runtime,
                 const Neon::set::BlockConfig& blockConfig);

    /**
     *
     * @param backend
     */
    KernelConfig(int           streamId,
                 Neon::Runtime runtime,
                 Neon::DataView);

    /**
     *
     */
    KernelConfig(int streamId,
                 Neon::DataView);

    /**
     *
     * @param blockSize
     */
    KernelConfig(int                           streamId,
                 Neon::Runtime                 runtime,
                 Neon::DataView                dataView,
                 const Neon::set::BlockConfig& blockConfig);

    /**
     *
     * @return
     */
    auto runtime() const -> Neon::Runtime;

    /**
     *
     * @return
     */
    auto blockConfig() const -> const Neon::set::BlockConfig&;
};


}  // namespace domain
}  // namespace Neon
