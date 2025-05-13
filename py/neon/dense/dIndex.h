#pragma once

#include "Neon/domain/details/dGrid/dIndex.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using NeonDenseIdx = ::Neon::domain::details::dGrid::dIndex;

//// create dense index
//CUDA_CALLABLE inline auto neon_idx_3d(int x, int y, int z) -> NeonDenseIdx
//{
//    return NeonDenseIdx(x, y, z);
//}
//
//CUDA_CALLABLE inline auto neon_init(NeonDenseIdx& idx, int x, int y, int z) -> void
//{
//    idx.setLocation().x= x;
//    idx.setLocation().y= y;
//    idx.setLocation().z= z;
//}
//

CUDA_CALLABLE inline auto neon_get_x(NeonDenseIdx& idx) -> int
{
    return idx.getLocation().x  ;
}

CUDA_CALLABLE inline auto neon_get_y(NeonDenseIdx& idx) -> int
{
    return idx.getLocation().y;
}

CUDA_CALLABLE inline auto neon_get_z(NeonDenseIdx& idx) -> int
{
    return idx.getLocation().z;
}

// print dense index
CUDA_CALLABLE inline auto neon_print(const NeonDenseIdx& a) -> void
{
    printf("neon_print - NeonDenseIdx(%d, %d, %d)\n", a.getLocation().x,  a.getLocation().y, a.getLocation().z);
}

}
