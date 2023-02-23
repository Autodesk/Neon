#pragma once

#include <functional>
#include "Neon/core/core.h"

namespace Neon {
namespace set {

class BlockConfig
{
   public:
    enum blockMode_e
    {
        user,
        system
    };

   private:
    struct shMemSize_t
    {
        enum e
        {
            implicitFun,
            explicitVal
        };
        size_t                                       m_val{0};
        std::function<size_t(const Neon::index_3d&)> m_fun;
        shMemSize_t::e                              m_mode = {explicitVal};

        auto set(std::function<size_t(const Neon::index_3d&)> fun) -> void
        {
            m_fun = fun;
            m_mode = e::implicitFun;
        }
        auto set(size_t s) -> void
        {
            m_val = s;
            m_mode = e::explicitVal;
        }

        auto size(const Neon::index_3d& block) const -> size_t
        {
            switch (m_mode) {
                case e::implicitFun: {
                    return m_fun(block);
                }
                case e::explicitVal: {
                    return m_val;
                }
                default:{
                    NEON_THROW_UNSUPPORTED_OPTION();
                }
            }
        }
    };

    blockMode_e    m_blockMode{blockMode_e::system};
    Neon::index_3d m_blockSize{0, 0, 0};
    shMemSize_t    m_shMem;
    //--------------------------------------------------------------------------
    // INITIALIZATION
    //--------------------------------------------------------------------------
   public:
    /**
     * Empty constructor
     */
    BlockConfig() = default;

    explicit BlockConfig(std::function<size_t(const Neon::index_3d& blockSize)>);

    /**
     *
     * @param blockSize
     */
    explicit BlockConfig(const Neon::index_3d& blockSize);

    /**
     *
     * @param blockSize
     * @param sharedMemSize
     */
    BlockConfig(const Neon::index_3d& blockSize, size_t sharedMemSize);


    /**
     * Returns the cuda block to be used
     * @return
     */
    auto blockMode() const -> const blockMode_e&;

    /**
     * Returns the cuda shared memory size to be used
     * @return
     */
    auto sharedMemory(const Neon::index_3d& block) const ->  size_t;

    /**
     *
     * @return
     */
    auto blockSize() const -> const Neon::index_3d&;

    /**
     *
     * @param blockSize
     * @param sharedMem
     */
    auto setBlockSize(Neon::index_3d blockSize, size_t sharedMem = 0) -> void;

    static auto defaultConfig() -> const BlockConfig&;
};

}  // namespace set
}  // namespace Neon
