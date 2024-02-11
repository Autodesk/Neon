#pragma once
#include "Neon/domain/details/bGridDisg/bGridDisg.h"

namespace Neon {

template <typename SBlock>
using bGridDisgGenericBlock = Neon::domain::details::disaggregated::bGridDisg::bGridDisg<SBlock>;
using bGridDisg = bGridDisgGenericBlock<Neon::domain::details::disaggregated::bGridDisg::BlockDefault>;

}