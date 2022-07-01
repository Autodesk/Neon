#pragma once

#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"

#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"
#include "Neon/set/memory/memSet.h"

#include "Neon/domain/internal/aGrid/aPartition.h"
#include "Neon/set/MemoryOptions.h"


namespace Neon::domain::internal::aGrid {
class aGrid /** Forward declaration for aField */;



template <typename T, int C = 0>
class Storage
{
   public:
    Storage();

    using Self = Storage;
    using Partition = aPartition<T, C>;
    using Grid = aGrid;

    Neon::set::MemSet_t<T> rawMem;

    auto getPartition(Neon::Execution, Neon::DataView, Neon::SetIdx) -> Partition&;

    auto getPartition(Neon::Execution, Neon::DataView, Neon::SetIdx) const-> const Partition&;

    auto getPartitionSet(Neon::Execution, Neon::DataView) -> Neon::set::DataSet<Partition>&;

   private:
    /**
     * This multi-level array returns a DataSet of Partition
     * given a DataView and an Execution types
     */
    std::array<std::array<Neon::set::DataSet<Partition>,
                          Neon::DataViewUtil::nConfig>,
               Neon::ExecutionUtils::numConfigurations>
        partitions;
};
}  // namespace Neon::domain::array
