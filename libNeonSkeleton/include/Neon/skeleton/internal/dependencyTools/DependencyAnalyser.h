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
    std::vector<Neon::set::container::GraphData::Uid> mParsedR{};
    std::vector<Neon::set::container::GraphData::Uid> mParsedW{};

    Neon::internal::dataDependency::DataUId mUid;
    Neon::internal::dataDependency::DataIdx mIdx;

   public:
    DependencyAnalyser() = delete;
    DependencyAnalyser(Neon::internal::dataDependency::DataUId,
                       Neon::internal::dataDependency::DataIdx);

    auto update(Neon::set::container::GraphData::Uid       newKernel,
                Neon::internal::dataDependency::AccessType newOp)
        -> std::vector<DataDependency>;
};

}  // namespace Neon::skeleton::internal