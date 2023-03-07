#include "Neon/domain/tools/partitioning/SpanDecomposition.h"

namespace Neon::domain::tools::partitioning {

auto SpanDecomposition::getNumBlockPerPartition() const -> const Neon::set::DataSet<int64_t>&
{
    return mNumBlocks;
}

auto SpanDecomposition::getFirstZSliceIdx() const -> const Neon::set::DataSet<int32_t>&
{
    return mZFirstIdx;
}

auto SpanDecomposition::getLastZSliceIdx() const -> const Neon::set::DataSet<int32_t>&
{
    return mZLastIdx;
}

}  // namespace Neon::domain::tools::partitioning