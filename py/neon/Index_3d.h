#pragma once

#include "Neon/domain/details/dGrid/dIndex.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using NeonIndex3d = ::Neon::index_3d;

// create dense index
CUDA_CALLABLE inline auto neon_idx_3d(int x, int y, int z) -> NeonIndex3d
{
    return NeonIndex3d(x, y, z);
}

CUDA_CALLABLE inline auto neon_init(NeonIndex3d& idx, int x, int y, int z) -> void
{
    idx.x= x;
    idx.y= y;
    idx.z= z;
}

CUDA_CALLABLE inline auto neon_get_x(NeonIndex3d& idx) -> int
{
    return idx.x  ;
}

CUDA_CALLABLE inline auto neon_is_equal(NeonIndex3d& idx, int x, int y, int z) -> bool
{
    //printf("(%d %d %d ) (%d %d %d)\n", idx.x, idx.y,idx.z, x, y, z);
    return idx.x == x && idx.y == y && idx.z == z;
}

CUDA_CALLABLE inline auto neon_get_y(NeonIndex3d& idx) -> int
{
    return idx.y;
}

CUDA_CALLABLE inline auto neon_get_z(NeonIndex3d& idx) -> int
{
    return idx.z;
}

// print dense index
CUDA_CALLABLE inline auto neon_print(const NeonIndex3d& a) -> void
{
    printf("neon_print - NeonIndex3d(%d, %d, %d)\n", a.x,  a.y, a.z);
}

CUDA_CALLABLE inline auto neon_cuda_info() -> void
{
    printf("neon_print - CUDA - Grid dim (%d, %d, %d) block dim (%d, %d, %d) block (%d, %d, %d) thread (%d %d %d)\n",
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    blockIdx.x,  blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

}
