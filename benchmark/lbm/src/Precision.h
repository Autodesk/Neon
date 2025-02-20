#pragma once

#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Neon/set/memory/memSet.h"

template <typename StorageFP,
          typename ComputeFP>
struct Precision
{
    using Storage = StorageFP;
    using Compute = ComputeFP;
};
