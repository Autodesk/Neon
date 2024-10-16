#include "Neon/domain/tools/Partitioner1D.h"

namespace Neon::domain::tool {


auto Partitioner1D::getSpanClassifier()
    const -> partitioning::SpanClassifier const&
{
    return *(mData->mSpanClassifier);
}

auto Partitioner1D::getSpanLayout()
    const -> partitioning::SpanLayout const&
{
    return *(mData->mSpanLayout);
}

auto Partitioner1D::getDecomposition() const -> partitioning::SpanDecomposition const&
{
    return *(mData->spanDecomposition.get());
}

}  // namespace Neon::domain::tools