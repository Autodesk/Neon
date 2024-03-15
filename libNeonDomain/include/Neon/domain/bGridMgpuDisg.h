#pragma once
#include "Neon/domain/details/bGridDisgMgpu//bGridDisgMgpu.h"

namespace Neon {

template <typename SBlock>
using bGridMgpuGenricBlock = Neon::domain::details::bGridDisgMgpu::bGridDisgMgpu<SBlock>;
using bGridMgpu = bGridMgpuGenricBlock<Neon::domain::details::bGridDisgMgpu::BlockDefault>;

}  // namespace Neon