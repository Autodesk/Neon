#pragma once
#include "Neon/domain/details/bGridDisgMgpu//bGridDisgMgpu.h"

namespace Neon {
using bGridMgpu = Neon::domain::details::bGridDisgMgpu::bGridDisgMgpu<Neon::domain::details::bGridDisgMgpu::BlockDefault>;
}