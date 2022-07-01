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
 * @brief  A class to represent 3d tuple data types.
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
#include "Neon/core/types/vec/vec3d_generic.h"

namespace Neon {

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE Vec_3d<RealType_ta, false, true>::Vec_3d()
{
    static_assert(sizeof(self_t) == sizeof(Integer) * axis_e::num_axis, "");
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>::Vec_3d(const RealType_ta& xyz)
{
    set(xyz);
}


template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>::Vec_3d(const RealType_ta other[Vec_3d<RealType_ta, false, true>::num_axis])
{
    set(other);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>::Vec_3d(RealType_ta px, RealType_ta py, RealType_ta pz)
{
    set(px, py, pz);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>::Vec_3d(const self_t& xyz)
{
    set(xyz);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>& Vec_3d<RealType_ta, false, true>::operator=(const Integer other)
{
    set(other);
    return *this;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>& Vec_3d<RealType_ta, false, true>::operator=(const self_t& other)
{
    set(other);
    return *this;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>& Vec_3d<RealType_ta, false, true>::operator=(const Integer other[Vec_3d<RealType_ta, false, true>::num_axis])
{
    set(other);
    return *this;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_3d<RealType_ta, false, true>::set(const Integer val)
{
    x = val;
    y = val;
    z = val;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_3d<RealType_ta, false, true>::set(const Integer px, const Integer py, const Integer pz)
{
    x = px;
    y = py;
    z = pz;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_3d<RealType_ta, false, true>::set(const Integer p[Vec_3d<RealType_ta, false, true>::num_axis])
{
    x = p[0];
    y = p[1];
    z = p[2];
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_3d<RealType_ta, false, true>::set(const self_t& xyz)
{
    x = xyz.x;
    y = xyz.y;
    z = xyz.z;
}


//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_3d<RealType_ta, false, true>::rMax() const
{
    Integer themax = x;
    themax = (y > themax ? y : themax);
    themax = (z > themax ? z : themax);
    return themax;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_3d<RealType_ta, false, true>::rMin() const
{
    Integer themin = x;
    themin = (y < themin ? y : themin);
    themin = (z < themin ? z : themin);
    return themin;
}

template <typename RealType_ta>
inline RealType_ta Vec_3d<RealType_ta, false, true>::rAbsMax() const
{
    Integer themax = std::abs(x);
    Integer tmp = std::abs(y);
    themax = (tmp > themax ? tmp : themax);
    tmp = std::abs(z);
    themax = (tmp > themax ? tmp : themax);
    return themax;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_3d<RealType_ta, false, true>::rSum() const
{
    Integer redux = x + y + z;
    return redux;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_3d<RealType_ta, false, true>::rMul() const
{
    Integer redux = x * y * z;
    return redux;
}

template <typename RealType_ta>
template <typename OtherBaseType_ta>
NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta Vec_3d<RealType_ta, false, true>::rMulTyped() const
{
    OtherBaseType_ta redux = static_cast<OtherBaseType_ta>(x) * static_cast<OtherBaseType_ta>(y) * static_cast<OtherBaseType_ta>(z);
    return redux;
}

template <typename RealType_ta>
template <typename OtherBaseType_ta>
NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta Vec_3d<RealType_ta, false, true>::rSumTyped() const
{
    OtherBaseType_ta redux = static_cast<OtherBaseType_ta>(x) + static_cast<OtherBaseType_ta>(y) + static_cast<OtherBaseType_ta>(z);
    return redux;
}


//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
template <typename RealType_ta>
inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::abs() const
{
    return Vec_3d<Integer>(std::abs(x), std::abs(y), std::abs(z));
}

template <typename RealType_ta>
inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::pow2() const
{
    return Vec_3d<Integer>(x * x, y * y, z * z);
}

template <typename RealType_ta>
template <typename exponentType_t>
inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::pow(exponentType_t exp) const
{
    return Vec_3d<Integer>(std::pow(x, exp), std::pow(y, exp), std::pow(z, exp));
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_3d<RealType_ta, false, true>::normSq() const
{
    return x * x + y * y + z * z;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline RealType_ta Vec_3d<RealType_ta, false, true>::norm() const
{
    return std::sqrt(x * x + y * y + z * z);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::getNormalized() const
{
    const Integer normInverse = RealType_ta(1.0) / this->norm();
    return self_t(x * normInverse, y * normInverse, z * normInverse);
}


//---- [Index SECTION] ----------------------------------------------------------------------------------------------
//---- [Index SECTION] ----------------------------------------------------------------------------------------------
//---- [Index SECTION] ----------------------------------------------------------------------------------------------

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline index_t Vec_3d<RealType_ta, false, true>::idxOfMax() const
{
    Integer  themax = x;
    index_t  indexMax = 0;
    for (int index = 1; index < 3; index++) {
        if (themax < v[index]) {
            themax = v[index];
            indexMax = index;
        }
    }
    return indexMax;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline index_t Vec_3d<RealType_ta, false, true>::idxOfMin() const
{
    Integer  themin = x;
    index_t  indexMin = 0;
    for (int index = 1; index < 3; index++) {
        if (themin > v[index]) {
            themin = v[index];
            indexMin = index;
        }
    }
    return indexMin;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<index_t> Vec_3d<RealType_ta, false, true>::idxMinMask() const
{
    Vec_3d<index_t> mask(0);
    const index_t   index = this->iOfMin();
    mask.v[index] = 1;
    return mask;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<int32_t> Vec_3d<RealType_ta, false, true>::idxOrderByMax() const
{
    Vec_3d<int32_t> ordered(0, 1, 2);
    if (v[0] < v[1]) {
        ordered.v[0] = 1;
        ordered.v[1] = 0;
    }
    if (v[ordered.v[1]] < v[ordered.v[2]]) {

        int32_t tmp = ordered.v[1];
        ordered.v[1] = ordered.v[2];
        ordered.v[2] = tmp;
    }
    return ordered;
}


//---- [Operators SECTION] ----------------------------------------------------------------------------------------------
//---- [Operators SECTION] ----------------------------------------------------------------------------------------------
//---- [Operators SECTION] ----------------------------------------------------------------------------------------------

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>& Vec_3d<RealType_ta, false, true>::operator+=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x += B.x;
    A.y += B.y;
    A.z += B.z;
    ////
    return A;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>& Vec_3d<RealType_ta, false, true>::operator-=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x -= B.x;
    A.y -= B.y;
    A.z -= B.z;
    ////
    return A;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>& Vec_3d<RealType_ta, false, true>::operator*=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x *= B.x;
    A.y *= B.y;
    A.z *= B.z;
    ////
    return A;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true>& Vec_3d<RealType_ta, false, true>::operator/=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x /= B.x;
    A.y /= B.y;
    A.z /= B.z;
    ////
    return A;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator+(const int32_t b) const
{
    const Vec_3d& A = *this;
    self_t        C(A.x + (Integer)b, A.y + (Integer)b, A.z + (Integer)b);
    return C;
}
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator+(const int64_t b) const
{
    const Vec_3d& A = *this;
    self_t        C(A.x + (Integer)b, A.y + (Integer)b, A.z + (Integer)b);
    return C;
}
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator+(const float b) const
{
    const Vec_3d& A = *this;
    self_t        C(A.x + (Integer)b, A.y + (Integer)b, A.z + (Integer)b);
    return C;
}
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator+(const double b) const
{
    const Vec_3d& A = *this;
    self_t        C(A.x + (Integer)b, A.y + (Integer)b, A.z + (Integer)b);
    return C;
}
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator+(const Vec_3d& B) const
{
    const Vec_3d& A = *this;
    self_t        C(A.x + B.x, A.y + B.y, A.z + B.z);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator-(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x - B.x, A.y - B.y, A.z - B.z);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator%(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x % B.x, A.y % B.y, A.z % B.z);
    return C;
}

template <typename RealType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator*(const Vec_3d<K_tt>& B) const
{
    const Vec_3d<Integer>& A = *this;
    //Vec_3d<Integer>        C((Integer)(A.x * B.x), (Integer)(A.y * B.y), (Integer)(A.z * B.z));
    Vec_3d<Integer> C((Integer)(A.x * B.x), (Integer)(A.y * B.y), (Integer)(A.z * B.z));
    return C;
}

// [TODO]@Max("Convert to something with enable if basic type")
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator*(const RealType_ta& alpha) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x * static_cast<Integer>(alpha), A.y * static_cast<Integer>(alpha), A.z * static_cast<Integer>(alpha));
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator/(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x / B.x, A.y / B.y, A.z / B.z);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<RealType_ta, false, true>::operator>(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return (A.x > B.x) && (A.y > B.y) && (A.z > B.z);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<RealType_ta, false, true>::operator<(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return (A.x < B.x) && (A.y < B.y) && (A.z < B.z);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<RealType_ta, false, true>::operator>=(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return A.x >= B.x && A.y >= B.y && A.z >= B.z;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<RealType_ta, false, true>::operator<=(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return A.x <= B.x && A.y <= B.y && A.z <= B.z;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<RealType_ta, false, true>::operator==(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return A.x == B.x && A.y == B.y && A.z == B.z;
}
template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<RealType_ta, false, true>::operator!=(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return !(A == B);
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator>>(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x > B.x ? A.x : B.x, A.y > B.y ? A.y : B.y, A.z > B.z ? A.z : B.z);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<RealType_ta, false, true> Vec_3d<RealType_ta, false, true>::operator<<(const Vec_3d<RealType_ta, false, true>& B) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x < B.x ? A.x : B.x, A.y < B.y ? A.y : B.y, A.z < B.z ? A.z : B.z);
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_3d<RealType_ta, false, true>::operator()(RealType_ta _x, RealType_ta _y, RealType_ta _z)
{
    this->x = _x;
    this->y = _y;
    this->z = _z;
    return;
}

//---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------
//---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------
//---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------


template <typename RealType_ta>
template <typename target_gp_tt>
NEON_CUDA_HOST_DEVICE inline target_gp_tt Vec_3d<RealType_ta, false, true>::convert() const
{
    const Vec_3d<Integer>& A = *this;
    target_gp_tt            C;
    ////
    //// DREAMERTODO try to add a static cast...
    C.x = (A.x);
    C.y = (A.y);
    C.z = (A.z);
    ////
    return C;
}

template <typename RealType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_3d<K_tt> Vec_3d<RealType_ta, false, true>::newType() const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<K_tt>            C;
    ////
    //// DREAMERTODO try to add a static cast...
    C.x = static_cast<K_tt>(A.x);
    C.y = static_cast<K_tt>(A.y);
    C.z = static_cast<K_tt>(A.z);
    ////
    return C;
}

template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE void Vec_3d<RealType_ta, false, true>::to_printf(bool newLine) const
{
    if (newLine) {
        printf("(%f, %f, %f)\n", x, y, z);
    } else {
        printf("(%f, %f, %f)", x, y, z);
    }
}

template <typename RealType_ta>
std::string Vec_3d<RealType_ta, false, true>::to_string(const std::string& prefix) const
{
    std::string msg = prefix;
    msg += std::string("(");
    msg += std::to_string(x);
    msg += std::string(", ");
    msg += std::to_string(y);
    msg += std::string(", ");
    msg += std::to_string(z);
    msg += std::string(")");
    return msg;
}

template <typename RealType_ta>
std::string Vec_3d<RealType_ta, false, true>::to_string(int tab_num) const
{
    std::string prefix = std::string(tab_num, '\t');
    return this->to_string(prefix);
}

template <typename RealType_ta>
std::string Vec_3d<RealType_ta, false, true>::to_stringForComposedNames() const
{
    std::string msg = std::string("");
    msg += std::to_string(x);
    msg += std::string("_");
    msg += std::to_string(y);
    msg += std::string("_");
    msg += std::to_string(z);
    return msg;
}


template <typename RealType_ta>
NEON_CUDA_HOST_DEVICE RealType_ta Vec_3d<RealType_ta, false, true>::dot(const self_t& a, const self_t& b)
{
    Integer dotRes = 0;
    dotRes += a.x * b.x;
    dotRes += a.y * b.y;
    dotRes += a.z * b.z;
    return dotRes;
}

template <typename RealType_ta>
std::ostream& operator<<(std::ostream& out, const Vec_3d<RealType_ta, false, true>& p)
{
    out << p.to_string();
    return out;
}


}  // namespace Neon
