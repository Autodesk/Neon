#include "Neon/skeleton/internal/dependencyTools/UserDataManager.h"

namespace Neon::skeleton::internal {

auto UserDataManager::helpGetIdx(Neon::set::dataDependency::MultiXpuDataUid uid)
    -> Neon::set::dataDependency::MultiXpuDataIdx
{
    auto count = mUid2Idx.count(uid);
    if (count == 0) {
        Neon::set::dataDependency::MultiXpuDataIdx idx = mDepAnalyserVec.size();
        mDepAnalyserVec.emplace_back(uid, idx);
        mUid2Idx[uid] = idx;
        return idx;
    }
    return mUid2Idx[uid];
}

auto UserDataManager::updateStatus(Neon::set::container::GraphInfo::NodeUid nodeUid,
                                   Neon::set::dataDependency::AccessType                           op,
                                   Neon::set::dataDependency::MultiXpuDataUid dataUid)
    -> std::vector<DataDependency>
{
    auto idx = helpGetIdx(dataUid);
    auto depVec = mDepAnalyserVec[idx].update(nodeUid, op);
    return depVec;
}

}  // namespace Neon::skeleton::internal
