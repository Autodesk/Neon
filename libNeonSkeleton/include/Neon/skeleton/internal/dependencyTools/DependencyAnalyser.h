#pragma once
#include "Neon/set/Backend.h"
#include "Neon/skeleton/internal/dependencyTools/DataDependency.h"

namespace Neon::skeleton::internal {


/**
 * Stores type of operations on data for each kernels while user code is "parsed"
 * It is used to construct the user kernel dependency graph
 */
struct DependencyAnalyser
{
   private:
    std::vector<Neon::set::container::GraphInfo::NodeUid> mParsedR{};
    std::vector<Neon::set::container::GraphInfo::NodeUid> mParsedW{};

    Neon::internal::dataDependency::MdObjUid mUid;
    Neon::internal::dataDependency::MdObjIdx mIdx;

   public:
    DependencyAnalyser() = delete;
    DependencyAnalyser(Neon::internal::dataDependency::MdObjUid,
                       Neon::internal::dataDependency::MdObjIdx);

    auto update(Neon::set::container::GraphInfo::NodeUid       newKernel,
                Neon::internal::dataDependency::AccessType newOp)
        -> std::vector<DataDependency>;
};

}  // namespace Neon::skeleton::internal