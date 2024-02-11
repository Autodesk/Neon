#pragma once
#include "Neon/domain/details/bGridDisg/bGridDisg.h"

namespace Neon {
using bGridDisg = Neon::domain::details::disaggregated::bGridDisg::bGridDisg<Neon::domain::details::StaticBlock<4, 4, 4>>;
}