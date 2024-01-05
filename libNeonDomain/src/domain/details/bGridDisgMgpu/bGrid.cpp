#include "Neon/domain/details/bGridDisgMgpu/bGrid.h"

namespace Neon::domain::details::bGridMgpu {

template class bGrid<Neon::domain::details::bGridMgpu::StaticBlock<8,8,8>>;

}  // namespace Neon::domain::details::bGrid