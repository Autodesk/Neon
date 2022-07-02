#include "Neon/skeleton/internal/dependencyTools/UserDataManager.h"

namespace Neon::skeleton::internal {

auto UserDataManager::helpGetIdx(DataUId_t uid) -> DataIdx_t
{
    auto count = m_uid2Idx.count(uid);
    if (count == 0) {
        DataIdx_t idx = m_depAnalyserVec.size();
        m_depAnalyserVec.emplace_back(uid, idx);
        m_uid2Idx[uid] = idx;
        return idx;
    }
    return m_uid2Idx[uid];
}

auto UserDataManager::updateStatus(ContainerIdx newKernel,
                                   Access_e     op,
                                   DataUId_t    uid) -> std::vector<Dependency>
{
    auto idx = helpGetIdx(uid);
    auto depVec = m_depAnalyserVec[idx].update(newKernel, op);
    return depVec;
}

}  // namespace Neon::skeleton::internal
