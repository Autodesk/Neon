

// $Id$

// $Log$

#pragma once

#include <cmath>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>

//#include <cuda.h>
//#include <cuda_runtime_api.h>

#include "Neon/core/types/BasicTypes.h"
#include "Neon/core/types/Exceptions.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/types/memOptions.h"
#include "Neon/core/types/mode.h"
#include "Neon/core/types/vec/vec4d_generic.h"

namespace Neon {

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE Vec_4d<IntegerType_ta, true, false>::Vec_4d()
{
    static_assert(sizeof(self_t) == sizeof(element_t) * axis_e::num_axis, "");
};


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false>::Vec_4d(const self_t& other)
{
    set(other);
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false>::Vec_4d(const IntegerType_ta& xyzw)
{
    set(xyzw);
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false>::Vec_4d(const IntegerType_ta other[Vec_4d<IntegerType_ta, true, false>::num_axis])
{
    set(other);
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_ONLY inline Vec_4d<IntegerType_ta, true, false>::Vec_4d(std::initializer_list<IntegerType_ta> other)
{
    if (other.size() != self_t::num_axis && other.size() != 1) {
        NeonException exp("Vec_4d");
        exp << "initializer_list of length different than 1 or 4 ( was" << other.size() << ")";
        NEON_THROW(exp);
    }
    if (other.size() == 1) {
        x = *other.begin();
        y = x;
        z = x;
        w = x;
        return;
    }

    const element_t* begin = other.begin();
    x = begin[0];
    y = begin[1];
    z = begin[2];
    w = begin[3];
    return;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false>::Vec_4d(IntegerType_ta px, IntegerType_ta py, IntegerType_ta pz, IntegerType_ta pw)
{
    set(px, py, pz, pw);
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false>& Vec_4d<IntegerType_ta, true, false>::operator=(const self_t& other)
{
    this->set(other);
    return *this;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_4d<IntegerType_ta, true, false>::set(element_t px, element_t py, element_t pz, element_t pw)
{
    x = px;
    y = py;
    z = pz;
    w = pw;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_4d<IntegerType_ta, true, false>::set(IntegerType_ta p[Vec_4d<IntegerType_ta, true, false>::num_axis])
{
    x = p[0];
    y = p[1];
    z = p[2];
    w = p[3];
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_4d<IntegerType_ta, true, false>::set(const self_t& other)
{
    x = other.x;
    y = other.y;
    z = other.z;
    w = other.w;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_4d<IntegerType_ta, true, false>::set(const element_t& xyzw)
{
    x = xyzw;
    y = xyzw;
    z = xyzw;
    w = xyzw;
}


//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline IntegerType_ta Vec_4d<IntegerType_ta, true, false>::rMax() const
{
    element_t themax = x;
    themax = (y > themax ? y : themax);
    themax = (z > themax ? z : themax);
    themax = (w > themax ? w : themax);
    return themax;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline IntegerType_ta Vec_4d<IntegerType_ta, true, false>::rMin() const
{
    element_t themin = x;
    themin = (y < themin ? y : themin);
    themin = (z < themin ? z : themin);
    themin = (w < themin ? w : themin);
    return themin;
}


template <typename IntegerType_ta>
inline IntegerType_ta Vec_4d<IntegerType_ta, true, false>::rAbsMax() const
{
    element_t themax = std::abs(x);

    element_t tmp = std::abs(y);
    themax = (tmp > themax ? tmp : themax);

    tmp = std::abs(z);
    themax = (tmp > themax ? tmp : themax);

    tmp = std::abs(w);
    themax = (tmp > themax ? tmp : themax);
    return themax;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline IntegerType_ta Vec_4d<IntegerType_ta, true, false>::rSum() const
{
    element_t redux = x + y + z + w;
    return redux;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline IntegerType_ta Vec_4d<IntegerType_ta, true, false>::rMul() const
{
    element_t redux = x * y * z * w;
    return redux;
}

template <typename IntegerType_ta>
template <typename OtherBaseType_ta>
NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta Vec_4d<IntegerType_ta, true, false>::rMulTyped() const
{
    OtherBaseType_ta redux = static_cast<OtherBaseType_ta>(x) * static_cast<OtherBaseType_ta>(y) * static_cast<OtherBaseType_ta>(z) * static_cast<OtherBaseType_ta>(w);
    return redux;
}

template <typename IntegerType_ta>
template <typename OtherBaseType_ta>
NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta Vec_4d<IntegerType_ta, true, false>::rSumTyped() const
{
    OtherBaseType_ta redux = static_cast<OtherBaseType_ta>(x) + static_cast<OtherBaseType_ta>(y) + static_cast<OtherBaseType_ta>(z) + static_cast<OtherBaseType_ta>(w);
    return redux;
}


//---- [MEMORY SECTION] --------------------------------------------------------------------------------------------
//---- [MEMORY SECTION] --------------------------------------------------------------------------------------------
//---- [MEMORY SECTION] --------------------------------------------------------------------------------------------

template <typename IntegerType_ta>
template <Neon::memLayout_et::order_e order_ta, typename OtherIndexType_ta>
NEON_CUDA_HOST_DEVICE inline size_t Vec_4d<IntegerType_ta, true, false>::mCardDenseJump(const Integer_4d<OtherIndexType_ta>& dimGrid) const
{
    switch (order_ta) {
        case Neon::memLayout_et::order_e::structOfArrays: {
            return size_t(c) +
                   size_t(x) * size_t(dimGrid.c) +
                   size_t(y) * size_t(dimGrid.c) * size_t(dimGrid.x) +
                   size_t(z) * size_t(dimGrid.c) * size_t(dimGrid.x) * size_t(dimGrid.y);
        }
        case Neon::memLayout_et::order_e::arrayOfStructs: {
            return size_t(x) +
                   size_t(y) * size_t(dimGrid.x) +
                   size_t(z) * size_t(dimGrid.x) * size_t(dimGrid.y) +
                   size_t(c) * size_t(dimGrid.x) * size_t(dimGrid.y) * size_t(dimGrid.z);
            ;
        }
    }
}

template <typename IntegerType_ta>
template <typename OtherIndexType_ta>
NEON_CUDA_HOST_DEVICE inline size_t Vec_4d<IntegerType_ta, true, false>::mCardDenseJump(Neon::memLayout_et::order_e orderE, const Integer_4d<OtherIndexType_ta>& dimGrid) const
{
    switch (orderE) {
        case Neon::memLayout_et::order_e::structOfArrays: {
            return mCardDenseJump<Neon::memLayout_et::order_e::structOfArrays>(dimGrid);
        }
        case Neon::memLayout_et::order_e::arrayOfStructs: {
            return mCardDenseJump<Neon::memLayout_et::order_e::arrayOfStructs>(dimGrid);
        }
    }
}

template <typename IntegerType_ta>
template <typename OtherIndexType_ta>
NEON_CUDA_HOST_DEVICE inline size_t Vec_4d<IntegerType_ta, true, false>::mPitch(const Integer_4d<OtherIndexType_ta>& dimGrid) const
{
    return size_t(x) +
           size_t(y) * size_t(dimGrid.x) +
           size_t(z) * size_t(dimGrid.x) * size_t(dimGrid.y) +
           size_t(w) * size_t(dimGrid.x) * size_t(dimGrid.y) * size_t(dimGrid.z);
}

template <typename IntegerType_ta>
template <typename MemotyType_ta>
NEON_CUDA_HOST_DEVICE inline size_t Vec_4d<IntegerType_ta, true, false>::mSize() const
{
    return size_t(x) * size_t(y) * size_t(z) * size_t(w) * sizeof(MemotyType_ta);
}

//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------


template <typename IntegerType_ta>
inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::abs() const
{
    return Vec_4d<element_t>(std::abs(x), std::abs(y), std::abs(z), std::abs(w));
}


template <typename IntegerType_ta>
inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::pow2() const
{
    return Vec_4d<element_t>(x * x, y * y, z * z, w * w);
}


template <typename IntegerType_ta>
inline IntegerType_ta Vec_4d<IntegerType_ta, true, false>::norm() const
{
    return std::sqrt(x * x + y * y + z * z + w * w);
}


//---- [Index SECTION] ----------------------------------------------------------------------------------------------
//---- [Index SECTION] ----------------------------------------------------------------------------------------------
//---- [Index SECTION] ----------------------------------------------------------------------------------------------

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline index_t Vec_4d<IntegerType_ta, true, false>::idxOfMax() const
{
    element_t themax = x;
    index_t   indexMax = 0;
    for (int index = 1; index < 4; index++) {
        if (themax < v[index]) {
            themax = v[index];
            indexMax = index;
        }
    }
    return indexMax;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline index_t Vec_4d<IntegerType_ta, true, false>::idxOfMin() const
{
    element_t themin = x;
    index_t   indexMin = 0;
    for (int index = 1; index < 4; index++) {
        if (themin > v[index]) {
            themin = v[index];
            indexMin = index;
        }
    }
    return indexMin;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<index_t> Vec_4d<IntegerType_ta, true, false>::idxMinMask() const
{
    Vec_4d<index_t> mask(0);
    const index_t   index = this->iOfMin();
    mask.v[index] = 1;
    return mask;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<int32_t> Vec_4d<IntegerType_ta, true, false>::idxOrderByMax() const
{
    Vec_4d<int32_t> ordered(0, 1, 2, 3);
    if (v[0] < v[1]) {
        ordered.v[0] = 1;
        ordered.v[1] = 0;
    }
    if (v[ordered.v[1]] < v[ordered.v[2]]) {
        int32_t tmp = ordered.v[1];
        ordered.v[1] = ordered.v[2];
        ordered.v[2] = tmp;
    }
    if (v[ordered.v[2]] < v[ordered.v[3]]) {
        int32_t tmp = ordered.v[2];
        ordered.v[2] = ordered.v[3];
        ordered.v[3] = tmp;
    }
    return ordered;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline int32_t Vec_4d<IntegerType_ta, true, false>::countZeros() const
{
    int32_t nZeros = 0;
    nZeros += (x == 0 ? 1 : 0);
    nZeros += (y == 0 ? 1 : 0);
    nZeros += (z == 0 ? 1 : 0);
    nZeros += (w == 0 ? 1 : 0);
    return nZeros;
}

//---- [Operators SECTION] ----------------------------------------------------------------------------------------------
//---- [Operators SECTION] ----------------------------------------------------------------------------------------------
//---- [Operators SECTION] ----------------------------------------------------------------------------------------------


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false>& Vec_4d<IntegerType_ta, true, false>::operator+=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x += B.x;
    A.y += B.y;
    A.z += B.z;
    A.w += B.w;
    ////
    return A;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false>& Vec_4d<IntegerType_ta, true, false>::operator-=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x -= B.x;
    A.y -= B.y;
    A.z -= B.z;
    A.w -= B.w;
    ////
    return A;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false>& Vec_4d<IntegerType_ta, true, false>::operator*=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x *= B.x;
    A.y *= B.y;
    A.z *= B.z;
    A.w *= B.w;
    ////
    return A;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false>& Vec_4d<IntegerType_ta, true, false>::operator/=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x /= B.x;
    A.y /= B.y;
    A.z /= B.z;
    A.w /= B.w;
    ////
    return A;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator+(const int32_t b) const
{
    const Vec_4d& A = *this;
    self_t        C(A.x + (element_t)b, A.y + (element_t)b, A.z + (element_t)b, A.w + (element_t)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator+(const int64_t b) const
{
    const Vec_4d& A = *this;
    self_t        C(A.x + (element_t)b, A.y + (element_t)b, A.z + (element_t)b, A.w + (element_t)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator+(const float b) const
{
    const Vec_4d& A = *this;
    self_t        C(A.x + (element_t)b, A.y + (element_t)b, A.z + (element_t)b, A.w + (element_t)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator+(const double b) const
{
    const Vec_4d& A = *this;
    self_t        C(A.x + (element_t)b, A.y + (element_t)b, A.z + (element_t)b, A.w + (element_t)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator+(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_4d& A = *this;
    self_t        C(A.x + B.x, A.y + B.y, A.z + B.z, A.w + B.w);
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator-(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_4d<element_t>& A = *this;
    Vec_4d<element_t>        C(A.x - B.x, A.y - B.y, A.z - B.z, A.w - B.w);
    return C;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator%(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_4d<element_t>& A = *this;
    Vec_4d<element_t>        C(A.x % B.x, A.y % B.y, A.z % B.z, A.w % B.w);
    return C;
}

template <typename IntegerType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator*(const Vec_4d<K_tt>& B) const
{
    const Vec_4d<element_t>& A = *this;
    Vec_4d<element_t>        C((element_t)(A.x * B.x), (element_t)(A.y * B.y), (element_t)(A.z * B.z), (element_t)(A.w * B.w));
    return C;
}

template <typename IntegerType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator*(const K_tt& alpha) const
{
    const Vec_4d<element_t>& A = *this;
    const auto               alpha_c = static_cast<element_t>(alpha);
    Vec_4d<element_t>        C(A.x * alpha_c, A.y * alpha_c, A.z * alpha_c, A.w * alpha_c);
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator/(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_4d<element_t>& A = *this;
    Vec_4d<element_t>        C(A.x / B.x, A.y / B.y, A.z / B.z, A.w / B.w);
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_4d<IntegerType_ta, true, false>::operator>(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_4d<element_t>& A = *this;
    return (A.x > B.x) && (A.y > B.y) && (A.z > B.z) && (A.w > B.w);
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_4d<IntegerType_ta, true, false>::operator<(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_4d<element_t>& A = *this;
    return (A.x < B.x) && (A.y < B.y) && (A.z < B.z) && (A.w < B.w);
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_4d<IntegerType_ta, true, false>::operator>=(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<element_t>& A = *this;
    return A.x >= B.x && A.y >= B.y && A.z >= B.z && A.w >= B.w;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_4d<IntegerType_ta, true, false>::operator<=(const Integer_4d<element_t>& B) const
{
    const Vec_4d<element_t>& A = *this;
    return A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_4d<IntegerType_ta, true, false>::operator==(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_4d<element_t>& A = *this;
    return A.x == B.x && A.y == B.y && A.z == B.z && A.w == B.w;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_4d<IntegerType_ta, true, false>::operator==(const IntegerType_ta other[Vec_4d<IntegerType_ta, true, false>::num_axis]) const
{
    const Vec_4d<element_t>& A = *this;
    return A.x == other[0] && A.y == other[1] && A.z == other[2] && A.w == other[3];
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_4d<IntegerType_ta, true, false>::operator==(const IntegerType_ta otherScalar) const
{
    const Vec_4d<element_t>& A = *this;
    return A.x == otherScalar && A.y == otherScalar && A.z == otherScalar && A.w == otherScalar;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_4d<IntegerType_ta, true, false>::operator!=(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_4d<element_t>& A = *this;
    return !(A == B);
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_4d<IntegerType_ta, true, false>::operator!=(const IntegerType_ta other[Vec_4d<IntegerType_ta, true, false>::num_axis]) const
{
    const Vec_4d<element_t>& A = *this;
    return A.x != other[0] || A.y != other[1] || A.z != other[2] || A.w != other[3];
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator>>(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_4d<element_t>& A = *this;
    Vec_4d<element_t>        C(A.x > B.x ? A.x : B.x,
                        A.y > B.y ? A.y : B.y,
                        A.z > B.z ? A.z : B.z,
                        A.w > B.w ? A.w : B.w);
    return C;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_4d<IntegerType_ta, true, false> Vec_4d<IntegerType_ta, true, false>::operator<<(const Vec_4d<IntegerType_ta, true, false>& B) const
{
    const Vec_4d<element_t>& A = *this;
    Vec_4d<element_t>        C(A.x < B.x ? A.x : B.x,
                        A.y < B.y ? A.y : B.y,
                        A.z < B.z ? A.z : B.z,
                        A.w < B.w ? A.w : B.w);
    return C;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_4d<IntegerType_ta, true, false>::operator()(IntegerType_ta _x, IntegerType_ta _y, IntegerType_ta _z, IntegerType_ta _w)
{
    this->x = _x;
    this->y = _y;
    this->z = _z;
    this->w = _w;
    return;
}

template <typename IntegerType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_4d<K_tt> Vec_4d<IntegerType_ta, true, false>::newType() const
{
    const Vec_4d<element_t>& A = *this;
    Vec_4d<K_tt>             C;
    ////
    //// DREAMERTODO try to add a static cast...
    C.x = static_cast<K_tt>(A.x);
    C.y = static_cast<K_tt>(A.y);
    C.z = static_cast<K_tt>(A.z);
    C.w = static_cast<K_tt>(A.w);
    ////
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE void Vec_4d<IntegerType_ta, true, false>::to_printf(bool newLine) const
{
    if (newLine) {
        printf("(%d, %d, %d, %d)\n", x, y, z, w);
    } else {
        printf("(%d, %d, %d, %d)", x, y, z, w);
    }
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE void Vec_4d<IntegerType_ta, true, false>::to_printfLD(bool newLine) const
{
    if (newLine) {
        printf("(%ld, %ld, %ld, %ld)\n", x, y, z, w);
    } else {
        printf("(%ld, %ld, %ld, %ld)", x, y, z, w);
    }
}

template <typename IntegerType_ta>
std::string Vec_4d<IntegerType_ta, true, false>::to_string(const std::string& prefix) const
{
    std::string msg = prefix;
    msg += std::string("(");
    msg += std::to_string(x);
    msg += std::string(", ");
    msg += std::to_string(y);
    msg += std::string(", ");
    msg += std::to_string(z);
    msg += std::string(", ");
    msg += std::to_string(w);
    msg += std::string(")");
    return msg;
}

template <typename IntegerType_ta>
std::string Vec_4d<IntegerType_ta, true, false>::to_string(int tab_num) const
{
    std::string prefix = std::string(tab_num, '\t');
    return this->to_string(prefix);
}

template <typename IntegerType_ta>
std::string Vec_4d<IntegerType_ta, true, false>::to_stringForComposedNames() const
{
    std::string msg = std::string("");
    msg += std::to_string(x);
    msg += std::string("_");
    msg += std::to_string(y);
    msg += std::string("_");
    msg += std::to_string(z);
    msg += std::string("_");
    msg += std::to_string(w);
    return msg;
}

template <typename IntegerType_ta>
template <Neon::computeMode_t::computeMode_e computeMode_ta>
void Vec_4d<IntegerType_ta, true, false>::forEach(const self_t& len, std::function<void(const self_t& idx)> lambda)
{
    if (computeMode_ta == Neon::computeMode_t::par) {
#if _MSC_VER <= 1916
#pragma omp parallel for simd collapse(4)
#endif
        for (element_t w = 0; w < len.w; w++) {
            for (element_t z = 0; z < len.z; z++) {
                for (element_t y = 0; y < len.y; y++) {
                    for (element_t x = 0; x < len.x; x++) {
                        const self_t idx(x, y, z, w);
                        lambda(idx);
                    }
                }
            }
        }
    } else {
        for (element_t w = 0; w < len.w; w++) {
            for (element_t z = 0; z < len.z; z++) {
                for (element_t y = 0; y < len.y; y++) {
                    for (element_t x = 0; x < len.x; x++) {
                        const self_t idx(x, y, z, w);
                        lambda(idx);
                    }
                }
            }
        }
    }
}

template <typename IntegerType_ta>
template <Neon::computeMode_t::computeMode_e computeMode_ta>
void Vec_4d<IntegerType_ta, true, false>::forEach(const self_t& len, std::function<void(element_t idxX, element_t idxY, element_t idxZ, element_t idxW)> lambda)
{
    if (computeMode_ta == Neon::computeMode_t::par) {
#if _MSC_VER <= 1916
#pragma omp parallel for simd collapse(4)
#endif
        for (element_t w = 0; w < len.w; w++) {
            for (element_t z = 0; z < len.z; z++) {
                for (element_t y = 0; y < len.y; y++) {
                    for (element_t x = 0; x < len.x; x++) {
                        lambda(x, y, z, w);
                    }
                }
            }
        }
    } else {
        for (element_t w = 0; w < len.w; w++) {
            for (element_t z = 0; z < len.z; z++) {
                for (element_t y = 0; y < len.y; y++) {
                    for (element_t x = 0; x < len.x; x++) {
                        lambda(x, y, z, w);
                    }
                }
            }
        }
    }
}


template <typename IntegerType_ta>
std::ostream& operator<<(std::ostream& out, const Vec_4d<IntegerType_ta, true, false>& p)
{
    out << p.to_string();
    return out;
}

}  // namespace Neon
