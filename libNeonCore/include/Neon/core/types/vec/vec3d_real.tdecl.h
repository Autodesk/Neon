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

// #include <cuda.h>
// #include <cuda_runtime_api.h>

#include "Neon/core/types/BasicTypes.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/types/vec/vec3d_generic.h"

namespace Neon {

/**
 * Partial specialization for floating point types (float, double, long double, ...)
 */
template <typename RealType_ta>
class Vec_3d<RealType_ta, false, true>
{
   public:
    using Integer = RealType_ta;
    using self_t = Vec_3d<Integer, false, true>;

    static_assert(!std::is_integral<RealType_ta>::value, "");
    static_assert(std::is_floating_point<RealType_ta>::value, "");


    enum axis_e
    {
        x_axis = 0,
        y_axis = 1,
        z_axis = 2,
        num_axis = 3
    };

    Integer x;
    Integer y;
    Integer z;


    /**
     * Default constructor.
     */
    NEON_CUDA_HOST_DEVICE Vec_3d();

    /**
     * Default destructor.
     */
    ~Vec_3d() = default;

    /**
     * All component of the 3d tuple are set to the same scalar value.
     *   @param[in] xyz: selected value.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_3d(const Integer& xyz);

    /**
     * initialization through an 3 element array
     * @param other
     */
    NEON_CUDA_HOST_DEVICE inline explicit Vec_3d(const Integer other[self_t::num_axis]);

    /**
     * Creates a 3d tuple with specific values for each component.
     *   @param[in] px: value for the x component.
     *   @param[in] py: value for the y component.
     *   @param[in] pz: value for the z component.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_3d(Integer px, Integer py, Integer pz);

    /**
     * Copy constructor
     *   @param[in] xyz: element to be copied.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_3d(const self_t& xyz);

    NEON_CUDA_HOST_DEVICE inline Vec_3d& operator=(const Integer other);

    NEON_CUDA_HOST_DEVICE inline Vec_3d& operator=(const self_t& other);

    NEON_CUDA_HOST_DEVICE inline Vec_3d& operator=(const Integer other[self_t::num_axis]);

    NEON_CUDA_HOST_DEVICE inline void set(const Integer val);

    NEON_CUDA_HOST_DEVICE inline void set(const Integer px, const Integer py, const Integer pz);

    NEON_CUDA_HOST_DEVICE inline void set(const Integer p[self_t::num_axis]);

    NEON_CUDA_HOST_DEVICE inline void set(const self_t& xyz);

    NEON_CUDA_HOST_DEVICE inline auto constexpr getVectorView() -> Integer*;

    NEON_CUDA_HOST_DEVICE inline auto constexpr getVectorView() const -> const Integer*;

    //---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
    //---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
    //---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
    /**
     *   Extracts the max value stored by the 3d tuple.
     *   @return max value
     */
    NEON_CUDA_HOST_DEVICE inline Integer rMax() const;

    /**
     *   Extracts the min value stored by the 3d tuple.
     *   @return min value.
     */
    NEON_CUDA_HOST_DEVICE inline Integer rMin() const;

    /**
     *   Extracts the max absolute value stored by the 3d tuple.
     *   @return max absolute value
     */
    inline Integer rAbsMax() const;

    /**
     *   Reduce by sum: B = A.x + A.y + A.z
     *   @return A.x + A.y + A.z.
     */
    NEON_CUDA_HOST_DEVICE inline Integer rSum() const;

    /**
     *   Reduce by multiplication: A.x * A.y * A.z.
     *   @return A.x * A.y * A.z.
     */
    NEON_CUDA_HOST_DEVICE inline Integer rMul() const;

    /**
     *   Reduce by multiplication but data are converted to the new user type before the operation is computed.
     *   @return (newType)A.x * (newType)A.y * (newType)A.z.
     */
    template <typename OtherBaseType_ta>
    NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta rMulTyped() const;
    /**
     *   Reduce by sum, but data are converted to the new user type before the operation is computed.
     *   @return (newType)A.x + (newType)A.y + (newType)A.z.
     */
    template <typename OtherBaseType_ta>
    NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta rSumTyped() const;

    //---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
    //---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
    //---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------

    /**
     * Returns a 3d tuple where each component is the absolute value of the associated input component: {x,y,z}.abs() -> {|x|, |y|, |z|}
     * @return {|x|, |y|, |z|}
     * */
    inline self_t abs() const;

    /**
     * Returns a 3d tuple where each component is the power of 2 of the associated input component: {x,y,z}.getPow2() -> {x^2, y^2, z^2}
     * @return {x^2, y^2, z^2}
     */
    inline self_t pow2() const;

    template <typename exponentType_t>
    inline self_t pow(exponentType_t exp) const;

    /**
     * return the norm ov the 3d tuple.
     * @return {x,y,z}.norm() -> {x,y,z}
     * */
    NEON_CUDA_HOST_DEVICE inline Integer normSq() const;

    /**
     * return the norm ov the 3d tuple.
     * @return {x,y,z}.norm() -> |{x,y,z}|
     * */
    inline NEON_CUDA_HOST_DEVICE Integer norm() const;

    /**
     * Returns the normalized vector of the 3d tuple.
     * @return normalized 3d tuple.
     */
    inline NEON_CUDA_HOST_DEVICE self_t getNormalized() const;


    //---- [Index SECTION] ----------------------------------------------------------------------------------------------
    //---- [Index SECTION] ----------------------------------------------------------------------------------------------
    //---- [Index SECTION] ----------------------------------------------------------------------------------------------
    /**
     * Returns the index of 3d tuple element with the higher value.
     * @return component index (0, 1 or 2)
     */
    NEON_CUDA_HOST_DEVICE inline index_t idxOfMax() const;

    /**
     * Returns the index of 3d tuple element with the lower value.
     * @return component index (0, 1 or 2)
     */
    NEON_CUDA_HOST_DEVICE inline index_t idxOfMin() const;

    /**
     * Returns a mask which has a non zero element only for the 3d tuple component that stores the max value.
     * @return component index (0, 1 or 2)
     */
    NEON_CUDA_HOST_DEVICE inline Vec_3d<index_t> idxMinMask() const;
    /**
     * Returns a mask. The mask is computed by ordering the 3d tuple component values from higher to lower.
     * @return component index (0, 1 or 2)
     */
    NEON_CUDA_HOST_DEVICE inline Vec_3d<int32_t> idxOrderByMax() const;


    //---- [Operators SECTION] ----------------------------------------------------------------------------------------------
    //---- [Operators SECTION] ----------------------------------------------------------------------------------------------
    //---- [Operators SECTION] ----------------------------------------------------------------------------------------------

    /**
     * Operator +=
     */
    NEON_CUDA_HOST_DEVICE inline self_t& operator+=(const self_t& B);

    /**
     * Operator -=
     */
    NEON_CUDA_HOST_DEVICE inline self_t& operator-=(const self_t& B);

    /**
     * Operator *=
     */
    NEON_CUDA_HOST_DEVICE inline self_t& operator*=(const self_t& B);

    /**
     * Operator /=
     */
    NEON_CUDA_HOST_DEVICE inline self_t& operator/=(const self_t& B);

    NEON_CUDA_HOST_DEVICE inline self_t operator+(const int32_t b) const;

    NEON_CUDA_HOST_DEVICE inline self_t operator+(const int64_t b) const;

    NEON_CUDA_HOST_DEVICE inline self_t operator+(const float b) const;

    NEON_CUDA_HOST_DEVICE inline self_t operator+(const double b) const;

    NEON_CUDA_HOST_DEVICE inline self_t operator+(const Vec_3d& B) const;

    /**
     *   Compute the difference between two component A (this) and B (right hand side).
     *   @param[in] B: second element for the operation.
     *   @return Resulting point from the subtracting point B (represented by the input param B) form A
     */
    NEON_CUDA_HOST_DEVICE inline self_t operator-(const self_t& B) const;

    /**
     *   Compute the mod between two points A and B, component by component (A.x%B.x, A.y%B.y, A.z%B.z).
     *   @param[in] B: second point for the diff.
     *   @return Resulting point is C =(A.x % B.x, A.y % B.y, A.z % B.z)
     */
    NEON_CUDA_HOST_DEVICE inline self_t operator%(const self_t& B) const;

    /**
     *   Compute the multiplication between two points A and B, component by component (A.x*B.x, A.y*B.y, A.z*B.z).
     *   Be careful!!! if the type is int, the division will be an integer division!!!
     *   @param[in] B: second point for the division.
     *   @return Resulting point is C =(A.x / B.x, A.y / B.y, A.z / B.z)
     * */
    template <typename K_tt>
    NEON_CUDA_HOST_DEVICE inline self_t operator*(const Vec_3d<K_tt>& B) const;

    // [TODO]@Max("Convert to something with enable if basic type")
    NEON_CUDA_HOST_DEVICE inline self_t operator*(const Integer& alpha) const;

    /**
     *   Compute the division between two points A and B, component by component (A.x/B.x, A.y/B.y, A.z/B.z).
     *   Be careful!!! if the type is int, the division will be an integer division!!!
     *   @param[in] B: second point for the division.
     *   @return Resulting point is C =(A.x / B.x, A.y / B.y, A.z / B.z)
     */
    NEON_CUDA_HOST_DEVICE inline self_t operator/(const self_t& B) const;

    /**
     * Returns true if A.x > B.x && A.y > B.y && A.z > B.z
     *   @param[in] B: second point for the operation.
     *   @return Resulting point is C as C.v[i] = A.v[i] > B.v[i] ? A.v[i] : B.v[i]
     */
    NEON_CUDA_HOST_DEVICE inline bool operator>(const self_t& B) const;

    /**
     * Returns true if A.x > B.x && A.y > B.y && A.z > B.z
     *   @param[in] B: second point for the operation.
     *   @return Resulting point is C as C.v[i] = A.v[i] > B.v[i] ? A.v[i] : B.v[i]
     */
    NEON_CUDA_HOST_DEVICE inline bool operator<(const self_t& B) const;

    /**  Returns true if A.x >= B.x && A.y >= B.y && A.z >= B.z
     *   @param[in] B: second point for the operation.
     *   @return Resulting point is C as C.v[i] = A.v[i] > B.v[i] ? A.v[i] : B.v[i]
     */
    NEON_CUDA_HOST_DEVICE inline bool operator>=(const self_t& B) const;

    /**  Returns true if A.x <= B.x && A.y <= B.y && A.z <= B.z
     *   @param[in] B: second point for the operation.
     *   @return True if A.x <= B.x && A.y <= B.y && A.z <= B.z
     */
    NEON_CUDA_HOST_DEVICE inline bool operator<=(const self_t& B) const;

    /**  Returns true if A.x <= B.x && A.y <= B.y && A.z <= B.z
     *   @param[in] B: second point for the operation.
     *   @return True if A.x <= B.x && A.y <= B.y && A.z <= B.z
     */
    NEON_CUDA_HOST_DEVICE inline bool operator==(const self_t& B) const;

    NEON_CUDA_HOST_DEVICE inline bool operator!=(const self_t& B) const;

    /** Returns the most north-est-hi point buildable with A and B coordinates"
     * C = A >> B  is: C.v[i] = A.v[i] > B.v[i] ? A.v[i] : B.v[i]
     *   @param[in] B: second point for the operation.
     *   @return Resulting point is C as C.v[i] = A.v[i] > B.v[i] ? A.v[i] : B.v[i]
     */
    NEON_CUDA_HOST_DEVICE inline self_t operator>>(const self_t& B) const;

    /** Returns the most south-west-low point buildable form A and B coordinates"
     *  C = A << B  is: C.v[i] = A.v[i] < B.v[i] ? A.v[i] : B.v[i]
     *
     *  @param[in] B: second point for the operation.
     *  @return Resulting point is C as C.v[i] = A.v[i] < B.v[i] ? A.v[i] : B.v[i]
     */
    NEON_CUDA_HOST_DEVICE inline self_t operator<<(const self_t& B) const;

    NEON_CUDA_HOST_DEVICE inline void operator()(Integer x, Integer y, Integer z);

    //---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------
    //---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------
    //---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------

    template <typename target_gp_tt>
    NEON_CUDA_HOST_DEVICE inline target_gp_tt convert() const;

    template <typename K_tt>
    NEON_CUDA_HOST_DEVICE inline Vec_3d<K_tt> newType() const;

    NEON_CUDA_HOST_DEVICE void to_printf(bool newLine = true) const;

    std::string to_string(const std::string& prefix) const;

    std::string to_string(int tab_num = 0) const;

    std::string to_stringForComposedNames() const;

    NEON_CUDA_HOST_DEVICE static Integer dot(const self_t& a, const self_t& b);
};


template <typename T_ta>
std::ostream& operator<<(std::ostream& out, const Real_3d<T_ta>& p);


}  // namespace Neon
