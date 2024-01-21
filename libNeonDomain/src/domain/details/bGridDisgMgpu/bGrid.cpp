#include "Neon/domain/details/bGridDisgMgpu/bGrid.h"

namespace Neon::domain::details::bGridMgpu {

template class bGrid<StaticBlock<defaultBlockSize, defaultBlockSize, defaultBlockSize>>;

}  // namespace Neon::domain::details::bGrid