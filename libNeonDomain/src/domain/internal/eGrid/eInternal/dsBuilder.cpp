#include "Neon/domain/internal/eGrid/eInternals/dsBuilder.h"
#include "Neon/domain/internal/eGrid/eInternals/builder/flatPartitioning.h"

namespace Neon::domain::internal::eGrid {
namespace internals {
dsBuilder_t::dsBuilder_t(const Neon::set::DevSet&                          devSet,
                         const Neon::index_3d&                             sizeDomain,
                         const std::function<bool(const Neon::index_3d&)>& inOut,
                         int                                               nPartitions,
                         const Neon::domain::Stencil&                      stencil)
{
    p_compute_partition(devSet, sizeDomain, inOut, nPartitions, stencil);
}


void dsBuilder_t::p_compute_partition(const Neon::set::DevSet&                          devSet,
                                      const Neon::index_3d&                             sizeDomain,
                                      const std::function<bool(const Neon::index_3d&)>& inOut,
                                      int                                               nPartitions,
                                      const Neon::domain::Stencil&                      stencil)
{

    switch (m_schema.schema) {
        case decltype(m_schema)::FLAT: {
            /*
             *     flatPartitioning_t(const Neon::index_3d&                      domain,
                       std::function<bool(const Neon::index_3d&)> inOut,
                       int                                        nPartitions,
                       partitioning_et                            prtSchema,
                       const Neon::domain::stencil_t&                           stencil)
             */
            flatPartitioning_t flat(devSet, sizeDomain, inOut, nPartitions, stencil);
            m_frame = flat.getFrame();
            // m_frame->exportTopology_vti("frame_test.vti");
            return;
        }
        case decltype(m_schema)::UNDEFINED:
        default: {
            Neon::NeonException exc("eSparse");
            exc << "Undefined partitioning schema";
            NEON_THROW(exc);
        }
    };
}

}  // namespace internals
}  // namespace Neon::domain::internal::eGrid