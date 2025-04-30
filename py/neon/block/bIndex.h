#pragma once

#include "Neon/domain/details/bGrid/bIndex.h"

// TODO: currently, all types and builtins need to be in the wp:: namespace
namespace wp
{

// import types into this namespace
using NeonBlockIdx = ::Neon::domain::details::bGrid::bIndex<Neon::domain::details::bGrid::BlockDefault>;

CUDA_CALLABLE inline auto neon_print(const NeonBlockIdx& a) -> void
{
    printf("neon_print - NeonBlockIdx(%d, %d, %d, %d)\n", a.getDataBlockIdx(),a.getInDataBlockIdx().x,   a.getInDataBlockIdx().y, a.getInDataBlockIdx().z);
}
}
