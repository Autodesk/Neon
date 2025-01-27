#pragma once

#include <cstdio>

#include "Neon/domain/details/bGrid/bSpan.h"
//#include "Neon/domain/details/dGrid/dSpan.h"

#include "../Index_3d.h"
#include "./bIndex.h"

namespace wp
{
// using NeonBlockSpan = ::Neon::domain::details::dGrid::dSpan;
namespace neon {
namespace warp{
   using BlockDefault = Neon::domain::details::bGrid::BlockDefault;
  using  bSpan = ::Neon::domain::details::bGrid::template bSpan<BlockDefault>;
  }
 }
class NeonBlockSpan : public ::wp::neon::warp::bSpan
{
public:

    // ... that's why we need to initialize it from bytes
    NeonBlockSpan(const char* bytes, size_t n)
    {
//        printf("bSpan(%ld, %ld)\n", sizeof(neon::warp::bSpan), n);
//        //printf("dSpan(%ld, %ld)\n", sizeof(Neon::domain::details::dGrid::dSpan), n);
//
//        printf("mFirstDataBlockOffset(%ld, %ld)\n", sizeof(this->mFirstDataBlockOffset), n);
//        printf("mActiveMask(%ld, %ld)\n", sizeof(this->mActiveMask), n);
//        printf("mDataView(%ld, %ld)\n", sizeof(this->mDataView), n);
//
//        printf("NeonBlockSpan(%ld, %ld)\n", sizeof(*this), n);
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
