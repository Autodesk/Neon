#pragma once
#include "Neon/core/core.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/types/memSetOptions.h"

#include "Neon/sys/memory/MemDevice.h"

#include "Neon/set/Capture.h"
#include "Neon/set/Containter.h"
#include "Neon/set/DataConfig.h"
#include "Neon/set/DevSet.h"

#include "Neon/domain/interface/GridBase.h"
#include "Neon/domain/interface/KernelConfig.h"
#include "Neon/domain/interface/LaunchConfig.h"
#include "Neon/domain/interface/Stencil.h"
#include "Neon/domain/interface/common.h"
#include "Neon/domain/internal/eGrid/eInternals/dsBuilder.h"
#include "Neon/domain/patterns/PatternScalar.h"
#include "eField.h"

namespace Neon::domain::internal::eGrid {


struct eStorage
{
    auto getCount(DataView dw) -> Neon::set::DataSet<count_t>&
    {
        return count[DataViewUtil::toInt(dw)];
    }

    auto getCountPerDevice(DataView dw, Neon::SetIdx setIdx) -> count_t&
    {
        const int dwIdx = DataViewUtil::toInt(dw);
        return count[dwIdx][setIdx.idx()];
    }

    auto getPartitionIndexSpace(DataView dw) -> Neon::set::DataSet<ePartitionIndexSpace>&
    {
        return partitionIndexSpace[DataViewUtil::toInt(dw)];
    }
    // INPUT
    eField<Neon::index_t, index_3d::num_axis> inverseMappingFieldMirror;
    bool                                      inverseMappingEnabled = {false};
    // COMPUTED
    internals::dsBuilder_t builder;

   private:
    std::array<Neon::set::DataSet<ePartitionIndexSpace>, Neon::DataViewUtil::nConfig> partitionIndexSpace;
    std::array<Neon::set::DataSet<count_t>, Neon::DataViewUtil::nConfig>              count;
};

}  // namespace Neon::domain::internal::eGrid
