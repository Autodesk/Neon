#include "Neon/skeleton/internal/dependencyTools/DependencyAnalyser.h"


namespace Neon::skeleton::internal {


DependencyAnalyser::DependencyAnalyser(DataUId_t uid, DataIdx_t idx)
{
    m_uid = uid;
    m_idx = idx;
}

auto DependencyAnalyser::update(ContainerIdx newKernel, Access_e newOp)
    -> std::vector<Dependency>
{
    switch (newOp) {
        case Access_e::READ: {
            if (m_parsedW.size() == 0) {
                // Parsing a Read operation with no write before
                // We are at the beginning of the dependencies
                // a. just add the kernel in the parsed read kernels
                m_parsedR.push_back(newKernel);
                return std::vector<Dependency>(0);
            }
            if (m_parsedW.size() == 1) {
                // Parsing a Read operation with one write before
                // We have a RaW dependency
                // a. add the new kernel to the read queue
                // b. return a RaW between the new R and the old W
                m_parsedR.push_back(newKernel);
                auto       t0W = m_parsedW[0];
                auto       t1R = newKernel;
                Dependency d(t1R, Dependencies_e::RAW, m_uid, t0W);
                return {d};
            } else {
                // Violation of the state machine integrity
                NEON_THROW_UNSUPPORTED_OPERATION("");
            }
            break;
        }
        case Access_e::WRITE: {
            if (m_parsedR.size() == 0 && m_parsedW.size() == 0) {
                // Parsing a kernel for the first time and it is a W
                // Return an empty dependency
                m_parsedW.push_back(newKernel);
                return std::vector<Dependency>(0);
            }
            if (m_parsedR.size() != 0 && m_parsedW.size() <= 1) {
                // Parsing a W kernel after
                // .. none or one W old kernel
                // .. one or more R old kernels

                // Fire a WaR dependency between the new W and the old Rs
                std::vector<Dependency> res;
                for (auto t0R : m_parsedR) {
                    auto t1W = newKernel;
                    if (t1W == t0R) {
                        continue;
                    }
                    Dependency d(t1W, Dependencies_e::WAR, m_uid, t0R);
                    res.push_back(d);
                }
                m_parsedR.clear();
                m_parsedW = std::vector<ContainerIdx>(1, newKernel);
                return res;
            }

            if (m_parsedR.size() == 0 && m_parsedW.size() <= 1) {
                // Parsing a W kernel after a W old kernel
                NEON_WARNING("Skeleton: WaW dependency detected.");
                auto       t0W = m_parsedW[0];
                auto       t1W = newKernel;
                Dependency d(t1W, Dependencies_e::WAW, m_uid, t0W);
                m_parsedW = std::vector<ContainerIdx>(1, newKernel);
                return std::vector<Dependency>(1, d);
                ;
            }
            break;
        }
        case Access_e::NONE: {
            // Error
            NEON_THROW_UNSUPPORTED_OPTION("");
        }
    }
    NEON_THROW_UNSUPPORTED_OPTION("");
}


}  // namespace Neon::skeleton::internal