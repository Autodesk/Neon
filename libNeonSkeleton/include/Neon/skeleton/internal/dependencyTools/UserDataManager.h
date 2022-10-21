#pragma once
#include <map>
#include "Neon/set/Backend.h"
#include "Neon/set/container/Graph.h"
#include "Neon/set/dependency/Alias.h"
#include "Neon/set/dependency/DataDependencyType.h"
#include "Neon/skeleton/internal/dependencyTools/DependencyAnalyser.h"

namespace Neon::skeleton::internal {


/**
 * Keep track of all the data used by all kernels
 *
 * Each field is track by a local indexing.
 * The indexing is determined by the map mUid2Idx
 *
 * The local index (DataIdx) is used to access information on the user data
 * that is stored in the vector mDepAnalyserVec.
 */
struct UserDataManager
{
    std::vector<DependencyAnalyser> mDepAnalyserVec;
    std::map<Neon::set::dataDependency::MdObjUid,
             Neon::set::dataDependency::MdObjIdx>
        mUid2Idx;

   private:
    /**
     * helper function to retrieve a relative index from a unique identifier
     * @param uid
     * @return
     */
    auto helpGetIdx(Neon::set::dataDependency::MdObjUid uid)
        -> Neon::set::dataDependency::MdObjIdx;

   public:
    /**
     * Update data status.
     * It returns a vector if dependencies if any is detected.
     *
     * @param newKernel
     * @param op
     * @param uid
     * @return
     */
    auto updateStatus(Neon::set::container::GraphInfo::NodeUid       newKernel,
                      Neon::set::dataDependency::AccessType op,
                      Neon::set::dataDependency::MdObjUid uid) -> std::vector<DataDependency>;
};

}  // namespace Neon::skeleton::internal
