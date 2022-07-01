#pragma once
#include "Neon/set/Backend.h"
#include "Neon/skeleton/internal/dependencyTools/Alias.h"
#include "Neon/skeleton/internal/dependencyTools/Dependency.h"
#include "Neon/set/dependencyTools/enum.h"

namespace Neon::skeleton::internal {


/**
 * Stores type of operations on data for each kernels while user code is "parsed"
 * It is used to construct the user kernel dependency graph
 */
// TODO rename DataDependencyAnalizer_t
struct DependencyAnalyser
{
   private:
    std::vector<ContainerIdx> m_parsedR{};
    std::vector<ContainerIdx> m_parsedW{};

    DataUId_t m_uid;
    DataIdx_t m_idx;

   public:
    DependencyAnalyser() = delete;
    DependencyAnalyser(DataUId_t, DataIdx_t);
    auto update(ContainerIdx newKernel, Access_e newOp) -> std::vector<Dependency>;
};

}