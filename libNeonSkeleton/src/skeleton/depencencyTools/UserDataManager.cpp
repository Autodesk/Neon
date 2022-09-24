#include "Neon/skeleton/internal/dependencyTools/UserDataManager.h"

namespace Neon::skeleton::internal {

auto UserDataManager::helpGetIdx(Neon::internal::dataDependency::MdObjUid uid)
    -> Neon::internal::dataDependency::MdObjIdx
{
    auto count = mUid2Idx.count(uid);
    if (count == 0) {
        Neon::internal::dataDependency::MdObjIdx idx = mDepAnalyserVec.size();
        mDepAnalyserVec.emplace_back(uid, idx);
        mUid2Idx[uid] = idx;
        return idx;
    }
    return mUid2Idx[uid];
}

auto UserDataManager::updateStatus(Neon::set::container::GraphInfo::NodeUid nodeUid,
                                   Neon::internal::dataDependency::AccessType                           op,
                                   Neon::internal::dataDependency::MdObjUid dataUid)
    -> std::vector<DataDependency>
{
    auto idx = helpGetIdx(dataUid);
    auto depVec = mDepAnalyserVec[idx].update(nodeUid, op);
    return depVec;
}

}  // namespace Neon::skeleton::internal
