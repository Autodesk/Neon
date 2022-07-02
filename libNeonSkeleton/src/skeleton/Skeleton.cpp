#include "Neon/skeleton/Skeleton.h"

namespace Neon {
namespace skeleton {


Skeleton::Skeleton(const Neon::Backend& bk)
{
    setBackend(bk);
    m_inited = true;
}

void Skeleton::setBackend(const Neon::Backend& bk)
{
    mBackend = bk;
}

}  // namespace skeleton
}  // namespace Neon