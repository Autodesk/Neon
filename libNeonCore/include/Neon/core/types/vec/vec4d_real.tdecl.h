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
#include "Neon/core/types/vec/vec4d_generic.h"

namespace Neon {

/**
* Partial specialization for floating point types (float, double, long double, ...)
*/
template <typename RealType_ta>
class Vec_4d<RealType_ta, false, true>
{
   public:
    using element_t = RealType_ta;
    using self_t = Vec_4d<element_t, false, true>;

    static_assert(!std::is_integral<RealType_ta>::value, "");
    static_assert(std::is_floating_point<RealType_ta>::value, "");


    enum axis_e
    {
        x_axis = 0,
        y_axis = 1,
        z_axis = 2,
        w_axis = 3,
        num_axis = 4
    };

    union
    {
        element_t v[axis_e::num_axis]{
            static_cast<element_t>(.0), static_cast<element_t>(.0), static_cast<element_t>(.0), static_cast<element_t>(.0)};
        struct
        {
            union
            {
                element_t x;
                element_t r;
            };
            union
            {
                element_t y;
                element_t s;
            };
            union
            {
                element_t z;
                element_t t;
            };
            union
            {
                element_t w;
                element_t u;
            };
        };
    };


    /**
     * Default constructor.
     */
    NEON_CUDA_HOST_DEVICE Vec_4d();

    /**
     * Default destructor.
     */
    ~Vec_4d() = default;

    /**
     * All component of the 3d tuple are set to the same scalar value.
     *   @param[in] xyzw: selected value.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_4d(const element_t& xyzw);

    /**
     * initialization through an 3 element array
     * @param other 
     */
    NEON_CUDA_HOST_DEVICE inline explicit Vec_4d(const element_t other[self_t::num_axis]);

    /**
     * Creates a 3d tuple with specific values for each component.
     *   @param[in] px: value for the x component.
     *   @param[in] py: value for the y component.
     *   @param[in] pz: value for the z component.
	 *   @param[in] pw: value for the w component.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_4d(element_t px, element_t py, element_t pz, element_t pw);

    /**
     * Copy constructor  
     *   @param[in] xyzw: element to be copied.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_4d(const self_t& xyzw);

    NEON_CUDA_HOST_DEVICE inline Vec_4d& operator=(const element_t other);

    NEON_CUDA_HOST_DEVICE inline Vec_4d& operator=(const self_t& other);

    NEON_CUDA_HOST_DEVICE inline Vec_4d& operator=(const element_t other[self_t::num_axis]);

    NEON_CUDA_HOST_DEVICE inline void set(const element_t val);

    NEON_CUDA_HOST_DEVICE inline void set(const element_t px, const element_t py, const element_t pz, const element_t pw);

    NEON_CUDA_HOST_DEVICE inline void set(const element_t p[self_t::num_axis]);

    NEON_CUDA_HOST_DEVICE inline void set(const self_t& xyzw);

    //---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
    //---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
    //---- [REDUCE SECTION] --------------------------------------------------------------------------------------------
    /**
     *   Extracts the max value stored by the 3d tuple.
     *   @return max value
     */
    NEON_CUDA_HOST_DEVICE inline element_t rMax() const;

    /**
     *   Extracts the min value stored by the 3d tuple.
     *   @return min value.
     */
    NEON_CUDA_HOST_DEVICE inline element_t rMin() const;

    /**
     *   Extracts the max absolute value stored by the 3d tuple.
     *   @return max absolute value 
     */
    inline element_t rAbsMax() const;

    /**
     *   Reduce by sum: B = A.x + A.y + A.z + A.w
     *   @return A.x + A.y + A.z. + A.w.
     */
    NEON_CUDA_HOST_DEVICE inline element_t rSum() const;

    /**
     *   Reduce by multiplication: A.x * A.y * A.z. * A.w.
     *   @return A.x * A.y * A.z. * A.w.
     */
    NEON_CUDA_HOST_DEVICE inline element_t rMul() const;

    /**
     *   Reduce by multiplication but data are converted to the new user type before the operation is computed.
     *   @return (newType)A.x * (newType)A.y * (newType)A.z.  * (newType)A.w.
     */
    template <typename OtherBaseType_ta>
    NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta rMulTyped() const;
    /**
     *   Reduce by sum, but data are converted to the new user type before the operation is computed.
     *   @return (newType)A.x + (newType)A.y + (newType)A.z. + (newType)A.w.
     */
    template <typename OtherBaseType_ta>
    NEON_CUDA_HOST_DEVICE inline OtherBaseType_ta rSumTyped() const;

    //---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
    //---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
    //---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------

    /**
     * Returns a 3d tuple where each component is the absolute value of the associated input component: {x,y,z, w}.abs() -> {|x|, |y|, |z|, |w|}
     * @return {|x|, |y|, |z|, |w|}
     * */
    inline self_t abs() const;

    /**
     * Returns a 3d tuple where each component is the power of 2 of the associated input component: {x,y,z,w}.getPow2() -> {x^2, y^2, z^2, w^2}
     * @return {x^2, y^2, z^2, w^2}
     */
    inline self_t pow2() const;

    template <typename exponentType_t>
    inline self_t pow(exponentType_t exp) const;

    /**
     * return the norm ov the 3d tuple.
     * @return {x,y,z,w}.norm() -> {x,y,z,w}
     * */
    NEON_CUDA_HOST_DEVICE inline element_t normSq() const;

    /**
     * return the norm ov the 3d tuple.
     * @return {x,y,z,w}.norm() -> |{x,y,z,w}|
     * */
    inline NEON_CUDA_HOST_DEVICE element_t norm() const;

    /**
     * Returns the normalized vector of the 4d tuple.
     * @return normalized 4d tuple.
     */
    inline NEON_CUDA_HOST_DEVICE self_t getNormalized() const;


    //---- [Index SECTION] ----------------------------------------------------------------------------------------------
    //---- [Index SECTION] ----------------------------------------------------------------------------------------------
    //---- [Index SECTION] ----------------------------------------------------------------------------------------------
    /**
     * Returns the index of 4d tuple element with the higher value.
     * @return component index (0, 1, 2 or 3)
     */
    NEON_CUDA_HOST_DEVICE inline index_t idxOfMax() const;

    /**
     * Returns the index of 4d tuple element with the lower value.
     * @return component index (0, 1, 2 or 3)
     */
    NEON_CUDA_HOST_DEVICE inline index_t idxOfMin() const;

    /**
     * Returns a mask which has a non zero element only for the 4d tuple component that stores the max value.
     * @return component index (0, 1, 2 or 3)
     */
    NEON_CUDA_HOST_DEVICE inline Vec_4d<index_t> idxMinMask() const;
    /**
     * Returns a mask. The mask is computed by ordering the 4d tuple component values from higher to lower.
     * @return component index (0, 1, 2 or 3)
     */
    NEON_CUDA_HOST_DEVICE inline Vec_4d<int32_t> idxOrderByMax() const;


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

    NEON_CUDA_HOST_DEVICE inline self_t operator+(const Vec_4d& B) const;

    /**
     *   Compute the difference between two component A (this) and B (right hand side).
     *   @param[in] B: second element for the operation.
     *   @return Resulting point from the subtracting point B (represented by the input param B) form A
     */
    NEON_CUDA_HOST_DEVICE inline self_t operator-(const self_t& B) const;

    /**
        *   Compute the mod between two points A and B, component by component (A.x%B.x, A.y%B.y, A.z%B.z, A.w%B.w).
        *   @param[in] B: second point for the diff.
        *   @return Resulting point is C =(A.x % B.x, A.y % B.y, A.z % B.z, A.w % B.w)
        */
    NEON_CUDA_HOST_DEVICE inline self_t operator%(const self_t& B) const;

    /**
     *   Compute the multiplication between two points A and B, component by component (A.x*B.x, A.y*B.y, A.z*B.z, A.w*B.w).
     *   Be careful!!! if the type is int, the division will be an integer division!!!
     *   @param[in] B: second point for the division.
     *   @return Resulting point is C =(A.x / B.x, A.y / B.y, A.z / B.z, A.w / B.w)
     * */
    template <typename K_tt>
    NEON_CUDA_HOST_DEVICE inline self_t operator*(const Vec_4d<K_tt>& B) const;

    // [TODO]@Max("Convert to something with enable if basic type")
    NEON_CUDA_HOST_DEVICE inline self_t operator*(const element_t& alpha) const;

    /**
     *   Compute the division between two points A and B, component by component (A.x/B.x, A.y/B.y, A.z/B.z, A.w/B.w).
     *   Be careful!!! if the type is int, the division will be an integer division!!!
     *   @param[in] B: second point for the division.
     *   @return Resulting point is C =(A.x / B.x, A.y / B.y, A.z / B.z, A.w / B.w)
     */
    NEON_CUDA_HOST_DEVICE inline self_t operator/(const self_t& B) const;

    /**
     * Returns true if A.x > B.x && A.y > B.y && A.z > B.z && A.w > B.w
     *   @param[in] B: second point for the operation.
     *   @return Resulting point is C as C.v[i] = A.v[i] > B.v[i] ? A.v[i] : B.v[i]
     */
    NEON_CUDA_HOST_DEVICE inline bool operator>(const self_t& B) const;

    /**
     * Returns true if A.x > B.x && A.y > B.y && A.z > B.z && A.w > B.w
     *   @param[in] B: second point for the operation.
     *   @return Resulting point is C as C.v[i] = A.v[i] > B.v[i] ? A.v[i] : B.v[i]
     */
    NEON_CUDA_HOST_DEVICE inline bool operator<(const self_t& B) const;

    /**  Returns true if A.x >= B.x && A.y >= B.y && A.z >= B.z && A.w >= B.w
         *   @param[in] B: second point for the operation.
         *   @return Resulting point is C as C.v[i] = A.v[i] > B.v[i] ? A.v[i] : B.v[i]
         */
    NEON_CUDA_HOST_DEVICE inline bool operator>=(const self_t& B) const;

    /**  Returns true if A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w
         *   @param[in] B: second point for the operation.
         *   @return True if A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w
         */
    NEON_CUDA_HOST_DEVICE inline bool operator<=(const self_t& B) const;

    /**  Returns true if A.x <= B.x && A.y <= B.y && A.z <= B.z  && A.w <= B.w
     *   @param[in] B: second point for the operation.
     *   @return True if A.x <= B.x && A.y <= B.y && A.z <= B.z && A.w <= B.w
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

    NEON_CUDA_HOST_DEVICE inline void operator()(element_t x, element_t y, element_t z, element_t w);

    //---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------
    //---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------
    //---- [Operators CONVERSION] ----------------------------------------------------------------------------------------------

    template <typename target_gp_tt>
    NEON_CUDA_HOST_DEVICE inline target_gp_tt convert() const;

    template <typename K_tt>
    NEON_CUDA_HOST_DEVICE inline Vec_4d<K_tt> newType() const;

    NEON_CUDA_HOST_DEVICE void to_printf(bool newLine = true) const;

    std::string to_string(const std::string& prefix) const;

    std::string to_string(int tab_num = 0) const;

    std::string to_stringForComposedNames() const;

    NEON_CUDA_HOST_DEVICE static element_t dot(const self_t& a, const self_t& b);
};


template <typename T_ta>
std::ostream& operator<<(std::ostream& out, const Real_4d<T_ta>& p);


}  // namespace Neon
