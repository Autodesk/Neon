#pragma once
#include "Neon/domain/details/bGridDisgBlockMask/bGridBlockMask.h"

namespace Neon {

template <typename SBlock>
using bGridBlockMaskGenericBlock = Neon::domain::details::disaggregated::bGridBlockMask::bGridBlockMask<SBlock>;
using bGridBlockMask = bGridBlockMaskGenericBlock<Neon::domain::details::disaggregated::bGridBlockMask::BlockDefault>;

}  // namespace Neon