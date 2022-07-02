#pragma once
#include <map>
#include "Neon/set/Backend.h"
#include "Neon/set/dependencyTools/Alias.h"
#include "Neon/set/dependencyTools/enum.h"
#include "Neon/skeleton/internal/dependencyTools/DependencyAnalyser.h"

namespace Neon::skeleton::internal {


/**
 * Keep track of all the data used by all kernels
 *
 * Each field is track by a local indexing.
 * The indexing is determined by the map m_uid2Idx
 *
 * The local index (DataIdx_t) is used to access information on the user data
 * that is stored in the vector m_depAnalyserVec.
 */
struct UserDataManager
{
    std::vector<DependencyAnalyser> m_depAnalyserVec;
    std::map<DataUId_t, DataIdx_t>    m_uid2Idx;

   private:
    /**
     * helper function to retrieve a relative index from a unique identifier
     * @param uid
     * @return
     */
    auto helpGetIdx(DataUId_t uid) -> DataIdx_t;

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
    auto updateStatus(ContainerIdx newKernel,
                      Access_e     op,
                      DataUId_t    uid) -> std::vector<Dependency>;
};

}  // namespace internal
