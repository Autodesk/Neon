#pragma once
#include "Neon/domain/details/bGridDisgMgpu//bGrid.h"

namespace Neon {
using bGridMgpu = Neon::domain::details::bGridMgpu::bGrid<Neon::domain::details::bGridMgpu::BlockDefault>;
}