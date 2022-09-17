#include "Neon/skeleton/internal/dependencyTools/UserDataManager.h"

namespace Neon::skeleton::internal {

auto UserDataManager::helpGetIdx(Neon::internal::dataDependency::DataUId uid)
    -> Neon::internal::dataDependency::DataIdx
{
    auto count = mUid2Idx.count(uid);
    if (count == 0) {
        Neon::internal::dataDependency::DataIdx idx = mDepAnalyserVec.size();
        mDepAnalyserVec.emplace_back(uid, idx);
        mUid2Idx[uid] = idx;
        return idx;
    }
    return mUid2Idx[uid];
}

auto UserDataManager::updateStatus(Neon::set::container::GraphData::Uid nodeUid,
                                   Neon::internal::dataDependency::AccessType                           op,
                                   Neon::internal::dataDependency::DataUId                            dataUid)
    -> std::vector<DataDependency>
{
    auto idx = helpGetIdx(dataUid);
    auto depVec = mDepAnalyserVec[idx].update(nodeUid, op);
    return depVec;
}

}  // namespace Neon::skeleton::internal
