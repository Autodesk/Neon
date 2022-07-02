#pragma once
#include <functional>

#include "Neon/domain/internal/eGrid/eInternals/builder/dsFrame.h"
#include "Partitioning.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/set/DevSet.h"
#include "Partitioning.h"

namespace Neon::domain::internal::eGrid {

namespace internals {

struct dsBuilder_t
{

   private:
    partitioning_et            m_schema = {};
    std::shared_ptr<dsFrame_t> m_frame{nullptr};

   public:
    dsBuilder_t() = default;
    dsBuilder_t(const Neon::set::DevSet& devSet,
                const Neon::index_3d&    domain,
                const std::function<bool(const Neon::index_3d&)>&,
                int                          nPartitions,
                const Neon::domain::Stencil& stencil);

    auto frame()
        const
        -> const std::shared_ptr<dsFrame_t>&
    {
        return m_frame;
    }

   private:
    /**
     * Compute partitioning
     */
    auto p_compute_partition(const Neon::set::DevSet&                          devSet,
                             const Neon::index_3d&                             sizeDomain,
                             const std::function<bool(const Neon::index_3d&)>& inOut,
                             int                                               nPartitions,
                             const Neon::domain::Stencil&)
        -> void;
};

}  // namespace internals
}  // namespace Neon::domain::internal::eGrid