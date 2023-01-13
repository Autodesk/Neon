

// $Id$

// $Log$

#pragma once

#include <cmath>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>

#include "Neon/core/types/BasicTypes.h"
#include "Neon/core/types/Exceptions.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/types/mode.h"
//#include "Neon/core/types/vec/vec2d_generic.h"
#include "Neon/core/types/vec/vec2d_integer.tdecl.h"
#include "Neon/core/types/vec/vecAlias.h"

namespace Neon {

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE Vec_2d<IntegerType_ta, true, false>::Vec_2d()
{
    static_assert(sizeof(self_t) == sizeof(eValue_t) * axis_e::num_axis, "");
};


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false>::Vec_2d(const self_t& other)
{
    set(other);
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false>::Vec_2d(const eValue_t& xy)
{
    set(xy);
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false>::Vec_2d(const eValue_t other[axis_e::num_axis])
{
    set(other);
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_ONLY inline Vec_2d<IntegerType_ta, true, false>::Vec_2d(std::initializer_list<eValue_t> other)
{
    if (other.size() != axis_e::num_axis && other.size() != 1) {
        NeonException exp("Vec_2d");
        exp << "initializer_list of length different than 1 or 3 ( was" << other.size() << ")";
        NEON_THROW(exp);
    }
    if (other.size() == 1) {
        x = *other.begin();
        y = x;
        return;
    }

    const eValue_t* begin = other.begin();
    x = begin[0];
    y = begin[1];
    return;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false>::Vec_2d(eValue_t px, eValue_t py)
{
    set(px, py);
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false>& Vec_2d<IntegerType_ta, true, false>::operator=(const self_t& other)
{
    this->set(other);
    return *this;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_2d<IntegerType_ta, true, false>::set(eValue_t px, eValue_t py)
{
    x = px;
    y = py;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_2d<IntegerType_ta, true, false>::set(eValue_t p[axis_e::num_axis])
{
    x = p[0];
    y = p[1];
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_2d<IntegerType_ta, true, false>::set(const self_t& other)
{
    x = other.x;
    y = other.y;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_2d<IntegerType_ta, true, false>::set(const eValue_t& xyz)
{
    x = xyz;
    y = xyz;
}


//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline IntegerType_ta Vec_2d<IntegerType_ta, true, false>::rMax() const
{
    eValue_t themax = x;
    themax = (y > themax ? y : themax);
    return themax;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline IntegerType_ta Vec_2d<IntegerType_ta, true, false>::rMin() const
{
    eValue_t themin = x;
    themin = (y < themin ? y : themin);
    return themin;
}


template <typename IntegerType_ta>
inline IntegerType_ta Vec_2d<IntegerType_ta, true, false>::rAbsMax() const
{
    eValue_t themax = std::abs(x);
    eValue_t tmp = std::abs(y);
    themax = (tmp > themax ? tmp : themax);
    return themax;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline IntegerType_ta Vec_2d<IntegerType_ta, true, false>::rSum() const
{
    eValue_t redux = x + y;
    return redux;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline IntegerType_ta Vec_2d<IntegerType_ta, true, false>::rMul() const
{
    eValue_t redux = x * y;
    return redux;
}

template <typename IntegerType_ta>
template <typename OtherBaseType_ta>
NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta Vec_2d<IntegerType_ta, true, false>::rMulTyped() const
{
    OtherBaseType_ta redux = static_cast<OtherBaseType_ta>(x) * static_cast<OtherBaseType_ta>(y);
    return redux;
}

template <typename IntegerType_ta>
template <typename OtherBaseType_ta>
NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta Vec_2d<IntegerType_ta, true, false>::rSumTyped() const
{
    OtherBaseType_ta redux = static_cast<OtherBaseType_ta>(x) + static_cast<OtherBaseType_ta>(y);
    return redux;
}


//---- [MEMORY SECTION] --------------------------------------------------------------------------------------------
//---- [MEMORY SECTION] --------------------------------------------------------------------------------------------
//---- [MEMORY SECTION] --------------------------------------------------------------------------------------------

template <typename IntegerType_ta>
template <typename OtherIndexType_ta>
NEON_CUDA_HOST_DEVICE inline size_t Vec_2d<IntegerType_ta, true, false>::mPitch(const Integer_2d<OtherIndexType_ta>& dimGrid) const
{
    return size_t(x) + size_t(y) * size_t(dimGrid.x);
}

template <typename IntegerType_ta>
template <typename OtherIndexType_ta>
NEON_CUDA_HOST_DEVICE inline size_t Vec_2d<IntegerType_ta, true, false>::mPitch(const OtherIndexType_ta dimX) const
{
    return (size_t)x + (size_t)y * (size_t)dimX;
}


template <typename IntegerType_ta>
template <typename MemotyType_ta>
NEON_CUDA_HOST_DEVICE inline size_t Vec_2d<IntegerType_ta, true, false>::mSize() const
{
    return size_t(x) * size_t(y) * sizeof(MemotyType_ta);
}


//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------


template <typename IntegerType_ta>
inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::abs() const
{
    return Vec_2d<eValue_t>(std::abs(x), std::abs(y));
}


template <typename IntegerType_ta>
inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::pow2() const
{
    return Vec_2d<eValue_t>(x * x, y * y);
}


template <typename IntegerType_ta>
inline IntegerType_ta Vec_2d<IntegerType_ta, true, false>::norm() const
{
    return static_cast<IntegerType_ta>(std::sqrt(x * x + y * y));
}

//---- [Index SECTION] ----------------------------------------------------------------------------------------------
//---- [Index SECTION] ----------------------------------------------------------------------------------------------
//---- [Index SECTION] ----------------------------------------------------------------------------------------------

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline index_t Vec_2d<IntegerType_ta, true, false>::idxOfMax() const
{
    eValue_t themax = x;
    index_t  indexMax = 0;
    for (int index = 1; index < axis_e::num_axis; index++) {
        if (themax < v[index]) {
            themax = v[index];
            indexMax = index;
        }
    }
    return indexMax;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline index_t Vec_2d<IntegerType_ta, true, false>::idxOfMin() const
{
    eValue_t themin = x;
    index_t  indexMin = 0;
    for (int index = 1; index < axis_e::num_axis; index++) {
        if (themin > v[index]) {
            themin = v[index];
            indexMin = index;
        }
    }
    return indexMin;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<index_t> Vec_2d<IntegerType_ta, true, false>::idxMinMask() const
{
    Vec_2d<index_t> mask(0);
    const index_t   index = this->iOfMin();
    mask.v[index] = 1;
    return mask;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<int32_t> Vec_2d<IntegerType_ta, true, false>::idxOrderByMax() const
{
    Vec_2d<int32_t> ordered(0, 1);
    if (v[0] < v[1]) {
        ordered.v[0] = 1;
        ordered.v[1] = 0;
    }

    return ordered;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline int32_t Vec_2d<IntegerType_ta, true, false>::countZeros() const
{
    int32_t nZeros = 0;
    nZeros += (x == 0 ? 1 : 0);
    nZeros += (y == 0 ? 1 : 0);
    return nZeros;
}

//---- [Operators SECTION] ----------------------------------------------------------------------------------------------
//---- [Operators SECTION] ----------------------------------------------------------------------------------------------
//---- [Operators SECTION] ----------------------------------------------------------------------------------------------


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false>& Vec_2d<IntegerType_ta, true, false>::operator+=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x += B.x;
    A.y += B.y;
    ////
    return A;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false>& Vec_2d<IntegerType_ta, true, false>::operator-=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x -= B.x;
    A.y -= B.y;
    ////
    return A;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false>& Vec_2d<IntegerType_ta, true, false>::operator*=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x *= B.x;
    A.y *= B.y;
    ////
    return A;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false>& Vec_2d<IntegerType_ta, true, false>::operator/=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x /= B.x;
    A.y /= B.y;
    ////
    return A;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator+(const int32_t b) const
{
    const Vec_2d& A = *this;
    self_t        C(A.x + (eValue_t)b, A.y + (eValue_t)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator+(const int64_t b) const
{
    const Vec_2d& A = *this;
    self_t        C(A.x + (eValue_t)b, A.y + (eValue_t)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator+(const float b) const
{
    const Vec_2d& A = *this;
    self_t        C(A.x + (eValue_t)b, A.y + (eValue_t)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator+(const double b) const
{
    const Vec_2d& A = *this;
    self_t        C(A.x + (eValue_t)b, A.y + (eValue_t)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator+(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d& A = *this;
    self_t        C(A.x + B.x, A.y + B.y);
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator-(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x - B.x, A.y - B.y);
    return C;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator%(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x % B.x, A.y % B.y);
    return C;
}

template <typename IntegerType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator*(const Vec_2d<K_tt>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    //Vec_2d<Integer>        C((Integer)(A.x * B.x), (Integer)(A.y * B.y), (Integer)(A.z * B.z));
    Vec_2d<eValue_t> C((eValue_t)(A.x * B.x), (eValue_t)(A.y * B.y));
    return C;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator*(const int32_t& alpha) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x * alpha, A.y * alpha);
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator/(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x / B.x, A.y / B.y);
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<IntegerType_ta, true, false>::operator>(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return (A.x > B.x) && (A.y > B.y);
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<IntegerType_ta, true, false>::operator<(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return (A.x < B.x) && (A.y < B.y);
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<IntegerType_ta, true, false>::operator>=(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return A.x >= B.x && A.y >= B.y;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<IntegerType_ta, true, false>::operator<=(const Integer_2d<eValue_t>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return A.x <= B.x && A.y <= B.y;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<IntegerType_ta, true, false>::operator==(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return A.x == B.x && A.y == B.y;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<IntegerType_ta, true, false>::operator==(const IntegerType_ta other[axis_e::num_axis]) const
{
    const Vec_2d<eValue_t>& A = *this;
    return A.x == other[0] && A.y == other[1];
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<IntegerType_ta, true, false>::operator!=(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return !(A == B);
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<IntegerType_ta, true, false>::operator!=(const IntegerType_ta other[axis_e::num_axis]) const
{
    const Vec_2d<eValue_t>& A = *this;
    return A.x != other[0] || A.y != other[1];
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator>>(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x > B.x ? A.x : B.x, A.y > B.y ? A.y : B.y);
    return C;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<IntegerType_ta, true, false> Vec_2d<IntegerType_ta, true, false>::operator<<(const Vec_2d<IntegerType_ta, true, false>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x < B.x ? A.x : B.x, A.y < B.y ? A.y : B.y);
    return C;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_2d<IntegerType_ta, true, false>::operator()(IntegerType_ta x, IntegerType_ta y)
{
    this->x = x;
    this->y = y;
    return;
}

template <typename IntegerType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_2d<K_tt> Vec_2d<IntegerType_ta, true, false>::newType() const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<K_tt>            C;
    ////
    //// DREAMERTODO try to add a static cast...
    C.x = static_cast<K_tt>(A.x);
    C.y = static_cast<K_tt>(A.y);
    ////
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE void Vec_2d<IntegerType_ta, true, false>::to_printf(bool newLine) const
{
    if (newLine) {
        printf("(%d, %d, %d)\n", x, y);
    } else {
        printf("(%d, %d, %d)", x, y);
    }
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE void Vec_2d<IntegerType_ta, true, false>::to_printfLD(bool newLine) const
{
    if (newLine) {
        printf("(%ld, %ld, %ld)\n", x, y);
    } else {
        printf("(%ld, %ld, %ld)", x, y);
    }
}

template <typename IntegerType_ta>
std::string Vec_2d<IntegerType_ta, true, false>::to_string(const std::string& prefix) const
{
    std::string msg = prefix;
    msg += std::string("(");
    msg += std::to_string(x);
    msg += std::string(", ");
    msg += std::to_string(y);
    msg += std::string(")");
    return msg;
}

template <typename IntegerType_ta>
std::string Vec_2d<IntegerType_ta, true, false>::to_string(int tab_num) const
{
    std::string prefix = std::string(tab_num, '\t');
    return this->to_string(prefix);
}

template <typename IntegerType_ta>
std::string Vec_2d<IntegerType_ta, true, false>::to_stringForComposedNames() const
{
    std::string msg = std::string("");
    msg += std::to_string(x);
    msg += std::string("_");
    msg += std::to_string(y);
    return msg;
}

template <typename IntegerType_ta>
template <Neon::computeMode_t::computeMode_e computeMode_ta>
void Vec_2d<IntegerType_ta, true, false>::forEach(const eValue_t& len, std::function<void(const self_t& idx)> lambda)
{
    if (computeMode_ta == Neon::computeMode_t::par) {
#if _MSC_VER <= 1916
#pragma omp parallel for simd collapse(2)
#endif
        for (eValue_t y = 0; y < len.y; y++) {
            for (eValue_t x = 0; x < len.x; x++) {
                const self_t idx(x, y);
                lambda(idx);
            }
        }

    } else {
        for (eValue_t y = 0; y < len.y; y++) {
            for (eValue_t x = 0; x < len.x; x++) {
                const self_t idx(x, y);
                lambda(idx);
            }
        }
    }
}

template <typename IntegerType_ta>
template <Neon::computeMode_t::computeMode_e computeMode_ta>
void Vec_2d<IntegerType_ta, true, false>::forEach(const eValue_t& len, std::function<void(eValue_t idxX, eValue_t idxY)> lambda)
{
    if (computeMode_ta == Neon::computeMode_t::par) {
#if _MSC_VER <= 1916
#pragma omp parallel for simd collapse(2)
#endif
        for (eValue_t y = 0; y < len.y; y++) {
            for (eValue_t x = 0; x < len.x; x++) {
                lambda(x, y);
            }
        }
    } else {
        for (eValue_t y = 0; y < len.y; y++) {
            for (eValue_t x = 0; x < len.x; x++) {
                lambda(x, y);
            }
        }
    }
}


template <typename IntegerType_ta>
std::ostream& operator<<(std::ostream& out, const Vec_2d<IntegerType_ta, true, false>& p)
{
    out << p.to_string();
    return out;
}

}  // namespace Neon
