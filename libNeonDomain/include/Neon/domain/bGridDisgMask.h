#pragma once
#include "Neon/domain/details/bGridDisgMask/bGridMask.h"

namespace Neon {

template <typename SBlock>
using bGridMaskGenericBlock = Neon::domain::details::disaggregated::bGridMask::bGridMask<SBlock>;
using bGridMask = bGridMaskGenericBlock<Neon::domain::details::disaggregated::bGridMask::BlockDefault>;

}  // namespace Neon