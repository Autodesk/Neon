#include "../../../libNeonSkeleton/tests/perf/SkeletonSyntheticBenchmarks/src/CLiCorrectness.h"
#include "Neon/domain/details/bGridDisg/BlockView.h"
#include "Neon/domain/details/ncclGrid/ncclGrid.h"
#include "Neon/set/Containter.h"

/**
 * A simple tutorial demonstrating the use of staggered grid in Neon.
 */
using Field = Neon::domain::details::ncclGrid::ncclField<int>;
auto laplaceTemplate(const Field& filedA,
                     Field&       fieldB)
    -> Neon::set::Container;