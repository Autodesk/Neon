#pragma once
#include "gUt_map_container.cu.h"
#include "gtest/gtest.h"

TEST(gUt_map, container)
{
    int nGpus = 3;
    NEON_INFO("kContainer - eGrid_t");
    runAllTestConfiguration("eGrid_t", basicOp_kContainer_ns::runKernel<eGrid_t, int64_t, 0>, nGpus);
}
