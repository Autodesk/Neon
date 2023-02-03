#pragma once

#include "Neon/set/Backend.h"

namespace Neon {
template <Neon::Execution execution,
          typename UserFunction>
auto Backend::forEachXpu(UserFunction function) const -> void
{
    const int nDev = getXpuCount();
    if constexpr (execution == Neon::Execution::seq) {
        for (int i = 0; i < nDev; i++) {
            function(i);
        }
    } else {
#pragma omp parallel for num_threads(nDev) default(none) shared(function)
        for (int i = 0; i < nDev; i++) {
            function(i);
        }
    }
}

}  // namespace Neon
