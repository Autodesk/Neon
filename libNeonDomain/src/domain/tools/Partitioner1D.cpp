#include "Neon/domain/tools/Partitioner1D.h"

namespace Neon::domain::tools {


auto Partitioner1D::getSpanClassifier()
    const -> partitioning::SpanClassifier const&
{
    return mSpanClassifier;
}

auto Partitioner1D::getSpanLayout()
    const -> partitioning::SpanLayout const&
{
    return mPartitionSpan;
}

}  // namespace Neon::domain::tools