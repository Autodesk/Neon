/**

   System:          Neon
   Component Name:  Core
   Language:        C++

   Licensed Material - Property of Autodesk Corp

   AUTODESK Copyright 2017. All rights reserved.

   Address:
            Autodesk Research
            210 King Street East, Suite 500
            Toronto, ON M5A 1J7
            Canada
**/

// $Id$
/**
 * @date   June, 2017
 * @brief  A class to represent 2d tuple data types.
 *
 * The class provides
 *      3. vector math operations
 *      4. conversion tools
 * */
// $Log$

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <type_traits>

//#include <cuda.h>
//#include <cuda_runtime_api.h>

#include "Neon/core/types/BasicTypes.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/types/vec/vec2d_generic.h"

#include "Neon/core/types/vec/vec2d_real.tdecl.h"

namespace Neon {

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE Vec_2d<RealType_ta, false, true>::Vec_2d()
{
    static_assert(sizeof(self_t) == sizeof(eValue_t) * axis_e::num_axis, "");
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>::Vec_2d(const RealType_ta& xyz)
{
    set(xyz);
}


template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>::Vec_2d(const RealType_ta other[axis_e::num_axis])
{
    set(other);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>::Vec_2d(RealType_ta px, RealType_ta py)
{
    set(px, py);
}


template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>::Vec_2d(const self_t& xyz)
{
    set(xyz);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>& Vec_2d<RealType_ta, false, true>::operator=(const eValue_t other)
{
    set(other);
    return *this;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>& Vec_2d<RealType_ta, false, true>::operator=(const self_t& other)
{
    set(other);
    return *this;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>& Vec_2d<RealType_ta, false, true>::operator=(const eValue_t other[axis_e::num_axis])
{
    set(other);
    return *this;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_2d<RealType_ta, false, true>::set(const eValue_t val)
{
    x = val;
    y = val;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_2d<RealType_ta, false, true>::set(const eValue_t px, const eValue_t py)
{
    x = px;
    y = py;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_2d<RealType_ta, false, true>::set(const eValue_t p[axis_e::num_axis])
{
    x = p[0];
    y = p[1];
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_2d<RealType_ta, false, true>::set(const self_t& xyz)
{
    x = xyz.x;
    y = xyz.y;
}


//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_2d<RealType_ta, false, true>::rMax() const
{
    eValue_t themax = x;
    themax = (y > themax ? y : themax);
    return themax;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_2d<RealType_ta, false, true>::rMin() const
{
    eValue_t themin = x;
    themin = (y < themin ? y : themin);
    return themin;
}

template <typename RealType_ta>
inline RealType_ta Vec_2d<RealType_ta, false, true>::rAbsMax() const
{
    eValue_t themax = std::abs(x);
    eValue_t tmp = std::abs(y);
    themax = (tmp > themax ? tmp : themax);

    return themax;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_2d<RealType_ta, false, true>::rSum() const
{
    eValue_t redux = x + y;
    return redux;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_2d<RealType_ta, false, true>::rMul() const
{
    eValue_t redux = x * y;
    return redux;
}

template <typename RealType_ta>
template <typename OtherBaseType_ta>
NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta Vec_2d<RealType_ta, false, true>::rMulTyped() const
{
    OtherBaseType_ta redux = static_cast<OtherBaseType_ta>(x) * static_cast<OtherBaseType_ta>(y);
    return redux;
}

template <typename RealType_ta>
template <typename OtherBaseType_ta>
NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta Vec_2d<RealType_ta, false, true>::rSumTyped() const
{
    OtherBaseType_ta redux = static_cast<OtherBaseType_ta>(x) + static_cast<OtherBaseType_ta>(y);
    return redux;
}


//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
template <typename RealType_ta>
inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::abs() const
{
    return Vec_2d<eValue_t>(std::abs(x), std::abs(y));
}

template <typename RealType_ta>
inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::pow2() const
{
    return Vec_2d<eValue_t>(x * x, y * y);
}

template <typename RealType_ta>
template <typename exponentType_t>
inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::pow(exponentType_t exp) const
{
    return self_t(std::pow(x, exp), std::pow(y, exp));
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_2d<RealType_ta, false, true>::normSq() const
{
    return x * x + y * y;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_2d<RealType_ta, false, true>::norm() const
{
    return std::sqrt(x * x + y * y);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::getNormalized() const
{
    const eValue_t normInverse = 1.0 / this->norm();
    return self_t(x * normInverse, y * normInverse);
}


//---- [Index SECTION] ----------------------------------------------------------------------------------------------
//---- [Index SECTION] ----------------------------------------------------------------------------------------------
//---- [Index SECTION] ----------------------------------------------------------------------------------------------

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline index_t Vec_2d<RealType_ta, false, true>::idxOfMax() const
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

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline index_t Vec_2d<RealType_ta, false, true>::idxOfMin() const
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

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<index_t> Vec_2d<RealType_ta, false, true>::idxMinMask() const
{
    Vec_2d<index_t> mask(0);
    const index_t   index = this->iOfMin();
    mask.v[index] = 1;
    return mask;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<int32_t> Vec_2d<RealType_ta, false, true>::idxOrderByMax() const
{
    Vec_2d<int32_t> ordered(0, 1);
    if (v[0] < v[1]) {
        ordered.v[0] = 1;
        ordered.v[1] = 0;
    }
    return ordered;
}


//---- [Operators SECTION] ----------------------------------------------------------------------------------------------
//---- [Operators SECTION] ----------------------------------------------------------------------------------------------
//---- [Operators SECTION] ----------------------------------------------------------------------------------------------

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>& Vec_2d<RealType_ta, false, true>::operator+=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x += B.x;
    A.y += B.y;
    ////
    return A;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>& Vec_2d<RealType_ta, false, true>::operator-=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x -= B.x;
    A.y -= B.y;
    ////
    return A;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>& Vec_2d<RealType_ta, false, true>::operator*=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x *= B.x;
    A.y *= B.y;
    ////
    return A;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true>& Vec_2d<RealType_ta, false, true>::operator/=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x /= B.x;
    A.y /= B.y;
    ////
    return A;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator+(const int32_t b) const
{
    const Vec_2d& A = *this;
    self_t        C(A.x + (eValue_t)b, A.y + (eValue_t)b);
    return C;
}
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator+(const int64_t b) const
{
    const Vec_2d& A = *this;
    self_t        C(A.x + (eValue_t)b, A.y + (eValue_t)b);
    return C;
}
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator+(const float b) const
{
    const Vec_2d& A = *this;
    self_t        C(A.x + (eValue_t)b, A.y + (eValue_t)b);
    return C;
}
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator+(const double b) const
{
    const Vec_2d& A = *this;
    self_t        C(A.x + (eValue_t)b, A.y + (eValue_t)b);
    return C;
}
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator+(const Vec_2d& B) const
{
    const Vec_2d& A = *this;
    self_t        C(A.x + B.x, A.y + B.y);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator-(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x - B.x, A.y - B.y);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator%(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x % B.x, A.y % B.y);
    return C;
}

template <typename RealType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator*(const Vec_2d<K_tt>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    //Vec_2d<Integer>        C((Integer)(A.x * B.x), (Integer)(A.y * B.y), (Integer)(A.z * B.z));
    Vec_2d<eValue_t> C((eValue_t)(A.x * B.x), (eValue_t)(A.y * B.y));
    return C;
}

// [TODO]@Max("Convert to something with enable if basic type")
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator*(const RealType_ta& alpha) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x * alpha, A.y * alpha);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator/(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x / B.x, A.y / B.y);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<RealType_ta, false, true>::operator>(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return (A.x > B.x) && (A.y > B.y);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<RealType_ta, false, true>::operator<(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return (A.x < B.x) && (A.y < B.y);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<RealType_ta, false, true>::operator>=(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return A.x >= B.x && A.y >= B.y;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<RealType_ta, false, true>::operator<=(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return A.x <= B.x && A.y <= B.y;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<RealType_ta, false, true>::operator==(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return A.x == B.x && A.y == B.y;
}
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_2d<RealType_ta, false, true>::operator!=(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    return !(A == B);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator>>(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x > B.x ? A.x : B.x, A.y > B.y ? A.y : B.y);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_2d<RealType_ta, false, true> Vec_2d<RealType_ta, false, true>::operator<<(const Vec_2d<RealType_ta, false, true>& B) const
{
    const Vec_2d<eValue_t>& A = *this;
    Vec_2d<eValue_t>        C(A.x < B.x ? A.x : B.x, A.y < B.y ? A.y : B.y);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_2d<RealType_ta, false, true>::operator()(RealType_ta x, RealType_ta y)
{
    this->x = x;
    this->y = y;
    return;
}

//---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------
//---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------
//---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------


template <typename RealType_ta>
template <typename target_gp_tt>
NEON_CUDA_HOST_DEVICE inline target_gp_tt Vec_2d<RealType_ta, false, true>::convert() const
{
    const Vec_2d<eValue_t>& A = *this;
    target_gp_tt            C;
    ////
    //// DREAMERTODO try to add a static cast...
    C.x = (A.x);
    C.y = (A.y);
    ////
    return C;
}

template <typename RealType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_2d<K_tt> Vec_2d<RealType_ta, false, true>::newType() const
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

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE void Vec_2d<RealType_ta, false, true>::to_printf(bool newLine) const
{
    if (newLine) {
        printf("(%f, %f, %f)\n", x, y);
    } else {
        printf("(%f, %f, %f)", x, y);
    }
}

template <typename RealType_ta>
std::string Vec_2d<RealType_ta, false, true>::to_string(const std::string& prefix) const
{
    std::string msg = prefix;
    msg += std::string("(");
    msg += std::to_string(x);
    msg += std::string(", ");
    msg += std::to_string(y);
    msg += std::string(")");
    return msg;
}

template <typename RealType_ta>
std::string Vec_2d<RealType_ta, false, true>::to_string(int tab_num) const
{
    std::string prefix = std::string(tab_num, '\t');
    return this->to_string(prefix);
}

template <typename RealType_ta>
std::string Vec_2d<RealType_ta, false, true>::to_stringForComposedNames() const
{
    std::string msg = std::string("");
    msg += std::to_string(x);
    msg += std::string("_");
    msg += std::to_string(y);
    msg += std::string("_");
    return msg;
}


template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE RealType_ta Vec_2d<RealType_ta, false, true>::dot(const self_t& a, const self_t& b)
{
    eValue_t dotRes = 0;
    dotRes += a.x * b.x;
    dotRes += a.y * b.y;
    return dotRes;
}

template <typename RealType_ta>
std::ostream& operator<<(std::ostream& out, const Vec_2d<RealType_ta, false, true>& p)
{
    out << p.to_string();
    return out;
}


}  // namespace Neon
