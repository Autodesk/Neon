#pragma once

#include <cstdio>

#include "Neon/domain/details/bGrid/bSpan.h"
#include "../Index_3d.h"
#include "./bIndex.h"

namespace wp
{
// using NeonBlockSpan = ::Neon::domain::details::dGrid::dSpan;

class NeonBlockSpan : public ::Neon::domain::details::bGrid::bSpan<Neon::domain::details::bGrid::BlockDefault>
{
public:

    // ... that's why we need to initialize it from bytes
    NeonBlockSpan(const char* bytes, size_t n)
    {
        assert(n == sizeof(*this));
        memcpy(this, bytes, n);
    }

    // NOTE: need default constructor for adjoint vars
    NeonBlockSpan()
    {
    }
};

CUDA_CALLABLE inline auto neon_set(NeonBlockSpan& span, bool& is_valid)
 -> NeonBlockIdx
{
    NeonBlockIdx index;
    using DummyType = int;
    is_valid = span.setAndValidateGPUDevice(index);

    return index;
}

//CUDA_CALLABLE inline auto neon_set(NeonBlockSpan& span, int x, int y, int z)
// -> NeonBlockIdx
//{
//    NeonBlockIdx index;
//    span.setAndValidate_warp(index, x,y,z);
//    return index;
//}
}
