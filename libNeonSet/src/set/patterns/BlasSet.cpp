#include "Neon/set/patterns/BlasSet.h"

namespace Neon::set::patterns {
template class BlasSet<float>;
template class BlasSet<double>;
template class BlasSet<int32_t>;
template class BlasSet<int64_t>;
template class BlasSet<uint32_t>;
template class BlasSet<uint64_t>;
}  // namespace Neon::set::patterns