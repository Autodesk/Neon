#include "Neon/domain/tools/partitioning/SpanDecomposition.h"

namespace Neon::domain::tool::partitioning {

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
auto SpanDecomposition::toString(Neon::Backend const& bk) const -> std::string
{
    std::stringstream s;
    bk.forEachDeviceSeq([&](Neon::SetIdx const& setIdx) {
        s << "\t" << setIdx << " blocks: " << this->getNumBlockPerPartition()[setIdx]
          << " first z " << this->getFirstZSliceIdx()[setIdx]
          << " last z " << this->getLastZSliceIdx()[setIdx]
          << " count " << this->getLastZSliceIdx()[setIdx] - this->getFirstZSliceIdx()[setIdx] + 1 << "\n";
    });
    return s.str();
}

}  // namespace Neon::domain::tool::partitioning