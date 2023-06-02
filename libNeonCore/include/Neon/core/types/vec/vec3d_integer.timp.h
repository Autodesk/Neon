#pragma once

#include <cmath>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>

#include "Neon/core/types/Macros.h"
// #include <cuda.h>
// #include <cuda_runtime_api.h>

#include "Neon/core/types/BasicTypes.h"
#include "Neon/core/types/Exceptions.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/types/mode.h"
#include "Neon/core/types/vec/vec3d_generic.h"

namespace Neon {

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE constexpr Vec_3d<IntegerType_ta, true, false>::Vec_3d()
{
    static_assert(sizeof(self_t) == sizeof(Integer) * axis_e::num_axis, "");
};


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr Vec_3d<IntegerType_ta, true, false>::Vec_3d(const self_t& other)
    : x(other.x), y(other.y), z(other.z)
{
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr Vec_3d<IntegerType_ta, true, false>::Vec_3d(const IntegerType_ta& xyz)
{
    set(xyz);
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr Vec_3d<IntegerType_ta, true, false>::Vec_3d(const IntegerType_ta other[Vec_3d<IntegerType_ta, true, false>::num_axis])
{
    set(other);
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_ONLY inline constexpr Vec_3d<IntegerType_ta, true, false>::Vec_3d(std::initializer_list<IntegerType_ta> other)
{
    if (other.size() != self_t::num_axis && other.size() != 1) {
        NeonException exp("Vec_3d");
        exp << "initializer_list of length different than 1 or 3 ( was" << other.size() << ")";
        NEON_THROW(exp);
    }
    if (other.size() == 1) {
        x = *other.begin();
        y = x;
        z = x;
        return;
    }

    const Integer* begin = other.begin();
    x = begin[0];
    y = begin[1];
    z = begin[2];
    return;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr Vec_3d<IntegerType_ta, true, false>::Vec_3d(IntegerType_ta px, IntegerType_ta py, IntegerType_ta pz)
    : x(px), y(py), z(pz)
{
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr Vec_3d<IntegerType_ta, true, false>& Vec_3d<IntegerType_ta, true, false>::operator=(const self_t& other)
{
    this->set(other);
    return *this;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr void Vec_3d<IntegerType_ta, true, false>::set(Integer px, Integer py, Integer pz)
{
    x = px;
    y = py;
    z = pz;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr void Vec_3d<IntegerType_ta, true, false>::set(IntegerType_ta p[Vec_3d<IntegerType_ta, true, false>::num_axis])
{
    x = p[0];
    y = p[1];
    z = p[2];
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr void Vec_3d<IntegerType_ta, true, false>::set(const self_t& other)
{
    x = other.x;
    y = other.y;
    z = other.z;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr void Vec_3d<IntegerType_ta, true, false>::set(const Integer& xyz)
{
    x = xyz;
    y = xyz;
    z = xyz;
}


//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
//---- [REDUCE SECTION] --------------------------------------------------------------------------------------------

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr IntegerType_ta Vec_3d<IntegerType_ta, true, false>::rMax() const
{
    Integer themax = x;
    themax = (y > themax ? y : themax);
    themax = (z > themax ? z : themax);
    return themax;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline constexpr IntegerType_ta Vec_3d<IntegerType_ta, true, false>::rMin() const
{
    Integer themin = x;
    themin = (y < themin ? y : themin);
    themin = (z < themin ? z : themin);
    return themin;
}


template <typename IntegerType_ta>
inline IntegerType_ta Vec_3d<IntegerType_ta, true, false>::rAbsMax() const
{
    Integer themax = std::abs(x);
    Integer tmp = std::abs(y);
    themax = (tmp > themax ? tmp : themax);
    tmp = std::abs(z);
    themax = (tmp > themax ? tmp : themax);
    return themax;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline IntegerType_ta Vec_3d<IntegerType_ta, true, false>::rSum() const
{
    Integer redux = x + y + z;
    return redux;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline IntegerType_ta Vec_3d<IntegerType_ta, true, false>::rMul() const
{
    Integer redux = x * y * z;
    return redux;
}

template <typename IntegerType_ta>
template <typename OtherBaseType_ta>
NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta Vec_3d<IntegerType_ta, true, false>::rMulTyped() const
{
    OtherBaseType_ta redux = static_cast<OtherBaseType_ta>(x) * static_cast<OtherBaseType_ta>(y) * static_cast<OtherBaseType_ta>(z);
    return redux;
}

template <typename IntegerType_ta>
template <typename OtherBaseType_ta>
NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta Vec_3d<IntegerType_ta, true, false>::rSumTyped() const
{
    OtherBaseType_ta redux = static_cast<OtherBaseType_ta>(x) + static_cast<OtherBaseType_ta>(y) + static_cast<OtherBaseType_ta>(z);
    return redux;
}


//---- [MEMORY SECTION] --------------------------------------------------------------------------------------------
//---- [MEMORY SECTION] --------------------------------------------------------------------------------------------
//---- [MEMORY SECTION] --------------------------------------------------------------------------------------------

template <typename IntegerType_ta>
template <typename OtherIndexType_ta>
NEON_CUDA_HOST_DEVICE inline size_t Vec_3d<IntegerType_ta, true, false>::mPitch(const Integer_3d<OtherIndexType_ta>& dimGrid) const
{
    return size_t(x) + size_t(y) * size_t(dimGrid.x) + size_t(z) * size_t(dimGrid.x) * size_t(dimGrid.y);
}

template <typename IntegerType_ta>
template <typename OtherIndexType_ta>
NEON_CUDA_HOST_DEVICE inline size_t Vec_3d<IntegerType_ta, true, false>::mPitch(const OtherIndexType_ta dimX,
                                                                                const OtherIndexType_ta dimY) const
{
    return (size_t)x + (size_t)y * (size_t)dimX + (size_t)z * (size_t)dimX * (size_t)dimY;
}


template <typename IntegerType_ta>
template <typename MemotyType_ta>
NEON_CUDA_HOST_DEVICE inline size_t Vec_3d<IntegerType_ta, true, false>::mSize() const
{
    return size_t(x) * size_t(y) * size_t(z) * sizeof(MemotyType_ta);
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline typename Vec_3d<IntegerType_ta, true, false>::self_t Vec_3d<IntegerType_ta, true, false>::mapTo3dIdx(size_t linear1D_idx) const
{
    self_t res;
    res.x = static_cast<IntegerType_ta>(linear1D_idx) % x;
    res.y = (static_cast<IntegerType_ta>(linear1D_idx) / x) % y;
    res.z = static_cast<IntegerType_ta>(linear1D_idx) / (y * x);
    return res;
}

//---- [CUDA SECTION] ----------------------------------------------------------------------------------------------
//---- [CUDA SECTION] ----------------------------------------------------------------------------------------------
//---- [CUDA SECTION] ----------------------------------------------------------------------------------------------


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::cudaGridDim(const self_t& blockDim) const
{
    const self_t& denseGridDim = *this;
    return ((denseGridDim - 1) / blockDim) + 1;
}

//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
//---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------

template <typename IntegerType_ta>
bool Vec_3d<IntegerType_ta, true, false>::isInsideBox(const self_t& low, const self_t& hi) const
{
    bool lowOk = true;
    lowOk = lowOk && (low.x <= x);
    lowOk = lowOk && (low.y <= y);
    lowOk = lowOk && (low.z <= z);

    bool hiOk = true;
    hiOk = hiOk && (x <= hi.x);
    hiOk = hiOk && (y <= hi.y);
    hiOk = hiOk && (z <= hi.z);

    const bool isIn = lowOk && hiOk;

    return isIn;
}


template <typename IntegerType_ta>
inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::abs() const
{
    return Vec_3d<Integer>(std::abs(x), std::abs(y), std::abs(z));
}


template <typename IntegerType_ta>
inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::pow2() const
{
    return Vec_3d<Integer>(x * x, y * y, z * z);
}


template <typename IntegerType_ta>
inline IntegerType_ta Vec_3d<IntegerType_ta, true, false>::norm() const
{
    return static_cast<IntegerType_ta>(std::sqrt(x * x + y * y + z * z));
}


//---- [Index SECTION] ----------------------------------------------------------------------------------------------
//---- [Index SECTION] ----------------------------------------------------------------------------------------------
//---- [Index SECTION] ----------------------------------------------------------------------------------------------

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline index_t Vec_3d<IntegerType_ta, true, false>::idxOfMax() const
{
    Integer themax = x;
    index_t indexMax = 0;
    for (int index = 1; index < 3; index++) {
        if (themax < v[index]) {
            themax = v[index];
            indexMax = index;
        }
    }
    return indexMax;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline index_t Vec_3d<IntegerType_ta, true, false>::idxOfMin() const
{
    Integer themin = x;
    index_t indexMin = 0;
    for (int index = 1; index < 3; index++) {
        if (themin > v[index]) {
            themin = v[index];
            indexMin = index;
        }
    }
    return indexMin;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<index_t> Vec_3d<IntegerType_ta, true, false>::idxMinMask() const
{
    Vec_3d<index_t> mask(0);
    const index_t   index = this->iOfMin();
    mask.v[index] = 1;
    return mask;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<int32_t> Vec_3d<IntegerType_ta, true, false>::idxOrderByMax() const
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


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline int32_t Vec_3d<IntegerType_ta, true, false>::countZeros() const
{
    int32_t nZeros = 0;
    nZeros += (x == 0 ? 1 : 0);
    nZeros += (y == 0 ? 1 : 0);
    nZeros += (z == 0 ? 1 : 0);
    return nZeros;
}

//---- [Operators SECTION] ----------------------------------------------------------------------------------------------
//---- [Operators SECTION] ----------------------------------------------------------------------------------------------
//---- [Operators SECTION] ----------------------------------------------------------------------------------------------


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false>& Vec_3d<IntegerType_ta, true, false>::operator+=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x += B.x;
    A.y += B.y;
    A.z += B.z;
    ////
    return A;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false>& Vec_3d<IntegerType_ta, true, false>::operator-=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x -= B.x;
    A.y -= B.y;
    A.z -= B.z;
    ////
    return A;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false>& Vec_3d<IntegerType_ta, true, false>::operator*=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x *= B.x;
    A.y *= B.y;
    A.z *= B.z;
    ////
    return A;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false>& Vec_3d<IntegerType_ta, true, false>::operator/=(const self_t& B)
{
    self_t& A = *this;
    ////
    A.x /= B.x;
    A.y /= B.y;
    A.z /= B.z;
    ////
    return A;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator+(const int32_t b) const
{
    const Vec_3d& A = *this;
    self_t        C(A.x + (Integer)b, A.y + (Integer)b, A.z + (Integer)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator+(const int64_t b) const
{
    const Vec_3d& A = *this;
    self_t        C(A.x + (Integer)b, A.y + (Integer)b, A.z + (Integer)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator+(const float b) const
{
    const Vec_3d& A = *this;
    self_t        C(A.x + (Integer)b, A.y + (Integer)b, A.z + (Integer)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator+(const double b) const
{
    const Vec_3d& A = *this;
    self_t        C(A.x + (Integer)b, A.y + (Integer)b, A.z + (Integer)b);
    return C;
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator+(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d& A = *this;
    self_t        C(A.x + B.x, A.y + B.y, A.z + B.z);
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator-(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x - B.x, A.y - B.y, A.z - B.z);
    return C;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator%(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x % B.x, A.y % B.y, A.z % B.z);
    return C;
}

template <typename IntegerType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator*(const Vec_3d<K_tt>& B) const
{
    const Vec_3d<Integer>& A = *this;
    // Vec_3d<Integer>        C((Integer)(A.x * B.x), (Integer)(A.y * B.y), (Integer)(A.z * B.z));
    Vec_3d<Integer> C((Integer)(A.x * B.x), (Integer)(A.y * B.y), (Integer)(A.z * B.z));
    return C;
}

template <typename IntegerType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator*(const K_tt& alpha) const
{
    const Vec_3d<Integer>& A = *this;
    const auto             alpha_c = static_cast<Integer>(alpha);
    Vec_3d<Integer>        C(A.x * alpha_c, A.y * alpha_c, A.z * alpha_c);
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator/(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x / B.x, A.y / B.y, A.z / B.z);
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator-() const
{
    Vec_3d<Integer> res(-x, -y, -z);
    return res;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<IntegerType_ta, true, false>::operator>(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return (A.x > B.x) && (A.y > B.y) && (A.z > B.z);
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<IntegerType_ta, true, false>::operator<(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return (A.x < B.x) && (A.y < B.y) && (A.z < B.z);
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<IntegerType_ta, true, false>::operator>=(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return A.x >= B.x && A.y >= B.y && A.z >= B.z;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<IntegerType_ta, true, false>::operator<=(const Integer_3d<Integer>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return A.x <= B.x && A.y <= B.y && A.z <= B.z;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<IntegerType_ta, true, false>::operator==(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return A.x == B.x && A.y == B.y && A.z == B.z;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<IntegerType_ta, true, false>::operator==(const IntegerType_ta other[Vec_3d<IntegerType_ta, true, false>::num_axis]) const
{
    const Vec_3d<Integer>& A = *this;
    return A.x == other[0] && A.y == other[1] && A.z == other[2];
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<IntegerType_ta, true, false>::operator==(const IntegerType_ta otherScalar) const
{
    const Vec_3d<Integer>& A = *this;
    return A.x == otherScalar && A.y == otherScalar && A.z == otherScalar;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<IntegerType_ta, true, false>::operator!=(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<Integer>& A = *this;
    return !(A == B);
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline bool Vec_3d<IntegerType_ta, true, false>::operator!=(const IntegerType_ta other[Vec_3d<IntegerType_ta, true, false>::num_axis]) const
{
    const Vec_3d<Integer>& A = *this;
    return A.x != other[0] || A.y != other[1] || A.z != other[2];
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator>>(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x > B.x ? A.x : B.x, A.y > B.y ? A.y : B.y, A.z > B.z ? A.z : B.z);
    return C;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline Vec_3d<IntegerType_ta, true, false> Vec_3d<IntegerType_ta, true, false>::operator<<(const Vec_3d<IntegerType_ta, true, false>& B) const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<Integer>        C(A.x < B.x ? A.x : B.x, A.y < B.y ? A.y : B.y, A.z < B.z ? A.z : B.z);
    return C;
}

template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE inline void Vec_3d<IntegerType_ta, true, false>::operator()(IntegerType_ta _x, IntegerType_ta _y, IntegerType_ta _z)
{
    this->x = _x;
    this->y = _y;
    this->z = _z;
    return;
}

template <typename IntegerType_ta>
template <typename K_tt>
NEON_CUDA_HOST_DEVICE inline Vec_3d<K_tt> Vec_3d<IntegerType_ta, true, false>::newType() const
{
    const Vec_3d<Integer>& A = *this;
    Vec_3d<K_tt>           C;
    ////
    //// DREAMERTODO try to add a static cast...
    C.x = static_cast<K_tt>(A.x);
    C.y = static_cast<K_tt>(A.y);
    C.z = static_cast<K_tt>(A.z);
    ////
    return C;
}


template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE void Vec_3d<IntegerType_ta, true, false>::to_printf(bool newLine) const
{
    if (newLine) {
        printf("(%d, %d, %d)\n", x, y, z);
    } else {
        printf("(%d, %d, %d)", x, y, z);
    }
}
template <typename IntegerType_ta>
NEON_CUDA_HOST_DEVICE void Vec_3d<IntegerType_ta, true, false>::to_printfLD(bool newLine) const
{
    if (newLine) {
        printf("(%ld, %ld, %ld)\n", x, y, z);
    } else {
        printf("(%ld, %ld, %ld)", x, y, z);
    }
}

template <typename IntegerType_ta>
std::string Vec_3d<IntegerType_ta, true, false>::to_string(const std::string& prefix) const
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

template <typename IntegerType_ta>
std::string Vec_3d<IntegerType_ta, true, false>::to_string(int tab_num) const
{
    std::string prefix = std::string(tab_num, '\t');
    return this->to_string(prefix);
}

template <typename IntegerType_ta>
std::string Vec_3d<IntegerType_ta, true, false>::to_stringForComposedNames() const
{
    std::string msg = std::string("");
    msg += std::to_string(x);
    msg += std::string("_");
    msg += std::to_string(y);
    msg += std::string("_");
    msg += std::to_string(z);
    return msg;
}

template <typename IntegerType_ta>
template <Neon::computeMode_t::computeMode_e computeMode_ta>
void Vec_3d<IntegerType_ta, true, false>::forEach(const self_t& len, std::function<void(const self_t& idx)> lambda)
{
    if (computeMode_ta == Neon::computeMode_t::par) {
#if _MSC_VER <= 1916
#pragma omp parallel for simd collapse(3)
#endif
        for (Integer z = 0; z < len.z; z++) {
            for (Integer y = 0; y < len.y; y++) {
                for (Integer x = 0; x < len.x; x++) {
                    const self_t idx(x, y, z);
                    lambda(idx);
                }
            }
        }
    } else {
        for (Integer z = 0; z < len.z; z++) {
            for (Integer y = 0; y < len.y; y++) {
                for (Integer x = 0; x < len.x; x++) {
                    const self_t idx(x, y, z);
                    lambda(idx);
                }
            }
        }
    }
}

template <typename IntegerType_ta>
template <Neon::computeMode_t::computeMode_e computeMode_ta>
void Vec_3d<IntegerType_ta, true, false>::forEach(const self_t&                                                 len,
                                                  std::function<void(Integer idxX, Integer idxY, Integer idxZ)> lambda)
{
    if constexpr (computeMode_ta == Neon::computeMode_t::par) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for
#else
#pragma omp parallel for simd collapse(3)
#endif
        for (Integer z = 0; z < len.z; z++) {
            for (Integer y = 0; y < len.y; y++) {
                for (Integer x = 0; x < len.x; x++) {
                    lambda(x, y, z);
                }
            }
        }
    } else {
        for (Integer z = 0; z < len.z; z++) {
            for (Integer y = 0; y < len.y; y++) {
                for (Integer x = 0; x < len.x; x++) {
                    lambda(x, y, z);
                }
            }
        }
    }
}

template <typename IntegerType_ta>
template <Neon::computeMode_t::computeMode_e computeMode_ta, class Lambda>
auto Vec_3d<IntegerType_ta, true, false>::forEach(const Lambda& lambda) const
    -> std::enable_if_t<std::is_invocable_v<Lambda, Vec_3d<IntegerType_ta, true, false>> ||
                            std::is_invocable_v<Lambda, Integer, Integer, Integer>,
                        void>
{
    if constexpr (std::is_invocable_v<Lambda, Integer, Integer, Integer>) {
        if constexpr (computeMode_ta == Neon::computeMode_t::par) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for
#else
#pragma omp parallel for simd collapse(3)
#endif
            for (Integer zIdx = 0; zIdx < this->z; zIdx++) {
                for (Integer yIdx = 0; yIdx < this->y; yIdx++) {
                    for (Integer xIdx = 0; xIdx < this->x; xIdx++) {
                        lambda(xIdx, yIdx, zIdx);
                    }
                }
            }
        } else {
            for (Integer zIdx = 0; zIdx < this->z; zIdx++) {
                for (Integer yIdx = 0; yIdx < this->y; yIdx++) {
                    for (Integer xIdx = 0; xIdx < this->x; xIdx++) {
                        lambda(xIdx, yIdx, zIdx);
                    }
                }
            }
        }
        return;
    }
    if constexpr (std::is_invocable_v<Lambda, Vec_3d<IntegerType_ta, true, false>>) {
        if constexpr (computeMode_ta == Neon::computeMode_t::par) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for
#else
#pragma omp parallel for simd collapse(3)
#endif
            for (Integer zIdx = 0; zIdx < this->z; zIdx++) {
                for (Integer yIdx = 0; yIdx < this->y; yIdx++) {
                    for (Integer xIdx = 0; xIdx < this->x; xIdx++) {
                        Vec_3d<IntegerType_ta, true, false> point(xIdx, yIdx, zIdx);
                        lambda(point);
                    }
                }
            }
        } else {
            for (Integer zIdx = 0; zIdx < this->z; zIdx++) {
                for (Integer yIdx = 0; yIdx < this->y; yIdx++) {
                    for (Integer xIdx = 0; xIdx < this->x; xIdx++) {
                        Vec_3d<IntegerType_ta, true, false> point(xIdx, yIdx, zIdx);
                        lambda(point);
                    }
                }
            }
        }
        return;
    }
}


template <typename IntegerType_ta>
std::ostream& operator<<(std::ostream& out, const Vec_3d<IntegerType_ta, true, false>& p)
{
    out << p.to_string();
    return out;
}

}  // namespace Neon
