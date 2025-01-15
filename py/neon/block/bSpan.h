#pragma once

#include <cstdio>

#include "Neon/domain/details/bGrid/bSpan.h"
#include "../Index_3d.h"
#include "./bIndex.h"

namespace wp
{
// using NeonDenseSpan = ::Neon::domain::details::dGrid::dSpan;

class NeonBlockSpan : public ::Neon::domain::details::bGrid::bSpan
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
//
//// print
//CUDA_CALLABLE inline auto neon_print(const NeonDenseSpan& a) -> void
//{
//    Neon::index_3d dim = a.helpGetDim();
////    printf("NeonDenseSpan(%d, %d, %d, {%d, %d, %d})\n",
////        int(a.helpGetDataView()),
////        a.helpGetZHaloRadius(),
////        a.helpGetZBoundaryRadius(),
////        dim.x, dim.y, dim.z);
//}
//
//CUDA_CALLABLE inline auto neon_set(NeonDenseSpan& span, bool& is_valid)
// -> NeonDenseIdx
//{
//    NeonDenseIdx index;
//    using DummyType = int;
//    is_valid = span.template setAndValidate_warp<DummyType>(index);
//
//    return index;
//}
//
//CUDA_CALLABLE inline auto neon_set(NeonDenseSpan& span, int x, int y, int z)
// -> NeonDenseIdx
//{
//    NeonDenseIdx index;
//    span.setAndValidate_warp(index, x,y,z);
//    return index;
//}


}
