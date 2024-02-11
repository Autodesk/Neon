#pragma once
#include "Neon/domain/details/bGridDisgMask/bGridMask.h"

namespace Neon {
using bGridMask = Neon::domain::details::disaggregated::bGridMask::bGridMask<
    Neon::domain::details::disaggregated::bGridMask::BlockDefault>;
}