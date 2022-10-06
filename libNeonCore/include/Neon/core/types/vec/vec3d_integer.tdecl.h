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
 * @brief  A class to represent 3 component data types.
 *
 * The class provides
 *      1. methods for computing memory pitch
 *      2. a set of tool to compute cuda grids
 *      3. vector math operations
 *      4. conversion tools
 * */
// $Log$

#pragma once

#include <cmath>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>

// #include <cuda.h>
// #include <cuda_runtime_api.h>

#include "Neon/core/types/BasicTypes.h"
#include "Neon/core/types/Exceptions.h"
#include "Neon/core/types/Macros.h"
#include "Neon/core/types/mode.h"
#include "Neon/core/types/vec/vec3d_generic.h"


namespace Neon {


/**
 * Partial specialization for integer types (int32_t, int64_t, size_t,...)
 */
template <typename IntegerType_ta>
class Vec_3d<IntegerType_ta, true, false>
{
   public:
    using Integer = IntegerType_ta;
    using self_t = Vec_3d<Integer, true, false>;
    using integer_3d = Vec_3d<Integer, true, false>;

    static_assert(std::is_integral<IntegerType_ta>::value, "");
    static_assert(!std::is_floating_point<IntegerType_ta>::value, "");


    enum axis_e
    {
        x_axis = 0,
        y_axis = 1,
        z_axis = 2,
        num_axis = 3
    };

    union
    {
        Integer v[axis_e::num_axis]{0, 0, 0};
        struct
        {
            union
            {
                Integer x;
                Integer r;
                Integer mXpitch;
            };
            union
            {
                Integer y;
                Integer s;
                Integer mYpitch;
            };
            union
            {
                Integer z;
                Integer t;
                Integer mZpitch;
            };
        };
    };

    /**
     * Empty constructor.
     */
    NEON_CUDA_HOST_DEVICE Vec_3d();
    /**
     * Empty destructor.
     */
    ~Vec_3d() = default;

    /**
     * All component of the 3d tuple are set to the same scalar value.
     *   @param[in] other the vector
     */
    NEON_CUDA_HOST_DEVICE inline Vec_3d(const self_t& other);

    /**
     * All component of the 3d tuple are set to the same scalar value.
     *   @param[in] xyz: selected value.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_3d(const Integer& xyz);

    NEON_CUDA_HOST_DEVICE inline Vec_3d(const Integer other[self_t::num_axis]);

    NEON_CUDA_HOST_ONLY inline Vec_3d(std::initializer_list<Integer> other);
    /**
     * Creates a 3d tuple with specific values for each component.
     *   @param[in] px: value for the x component.
     *   @param[in] py: value for the y component.
     *   @param[in] pz: value for the z component.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_3d(Integer px, Integer py, Integer pz);

    NEON_CUDA_HOST_DEVICE inline Vec_3d& operator=(const self_t& other);

    NEON_CUDA_HOST_DEVICE inline void set(Integer px, Integer py, Integer pz);

    NEON_CUDA_HOST_DEVICE inline void set(Integer p[self_t::num_axis]);

    NEON_CUDA_HOST_DEVICE inline void set(const self_t& other);

    NEON_CUDA_HOST_DEVICE inline void set(const Integer& xyz);


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


    //---- [MEMORY SECTION] --------------------------------------------------------------------------------------------
    //---- [MEMORY SECTION] --------------------------------------------------------------------------------------------
    //---- [MEMORY SECTION] --------------------------------------------------------------------------------------------

    /**
     * Returns the memory pitch for the point into a memory buffer representing a volume.
     * @return: pith in terms of element to access the value of the element indexed by this point.
     */
    template <typename OtherIndexType_ta>
    NEON_CUDA_HOST_DEVICE inline size_t mPitch(const Integer_3d<OtherIndexType_ta>& dimGrid /**< dimension of the volume */) const;

    /**
     * Returns the memory pitch for the point into a memory buffer representing a volume.
     * @return: pith in terms of element to access the value of the element indexed by this point.
     */
    template <typename OtherIndexType_ta>
    NEON_CUDA_HOST_DEVICE inline size_t mPitch(const OtherIndexType_ta dimX /**< dimension of the volume in x direction */,
                                               const OtherIndexType_ta dimY /**< dimension of the volume in y direction */) const;

    /**
     * Returns the size required to allocate a buffer for a grid that has the dimension defined by this object.
     */
    template <typename MemotyType_ta>
    NEON_CUDA_HOST_DEVICE inline size_t mSize() const;


    NEON_CUDA_HOST_DEVICE inline self_t mapTo3dIdx(size_t linear1D_idx) const;

    //---- [CUDA SECTION] ----------------------------------------------------------------------------------------------
    //---- [CUDA SECTION] ----------------------------------------------------------------------------------------------
    //---- [CUDA SECTION] ----------------------------------------------------------------------------------------------

    /**
     * Returns the smallest cuda grid covering the compute space defined by this 3d tuple object.
     * @return: a 3d tuple with the dimension of the cuda block grid.
     */
    NEON_CUDA_HOST_DEVICE inline self_t cudaGridDim(const self_t& blockDim) const;

    //---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
    //---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------
    //---- [GEOMETRY SECTION] ----------------------------------------------------------------------------------------------

    /**
     * Returns true if the index is included inside the
     * box delimited by the lower and higher corners.
     */
    bool isInsideBox(const self_t& low, const self_t& hi) const;

    /**
     * Returns a 3d tuple where each componet is the absolute value of the associated input component: {x,y,z}.abs() -> {|x|, |y|, |z|}
     * @return {|x|, |y|, |z|}
     * */
    inline self_t abs() const;

    /**
     * Returns a 3d tuple where each component is the power of 2 of the associated input component: {x,y,z}.getPow2() -> {x^2, y^2, z^2}
     * @return {x^2, y^2, z^2}
     */
    inline self_t pow2() const;

    /**
     * return the norm ov the 3d tuple.
     * @return {x,y,z}.norm() -> |{x,y,z}|
     * */
    inline Integer norm() const;

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


    NEON_CUDA_HOST_DEVICE inline int32_t countZeros() const;

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

    template <typename K_tt>
    NEON_CUDA_HOST_DEVICE inline self_t operator*(const K_tt& alpha) const;
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

    NEON_CUDA_HOST_DEVICE inline bool operator==(const Integer other[self_t::num_axis]) const;

    NEON_CUDA_HOST_DEVICE inline bool operator==(const Integer otherScalar) const;

    NEON_CUDA_HOST_DEVICE inline bool operator!=(const self_t& B) const;

    NEON_CUDA_HOST_DEVICE inline bool operator!=(const Integer other[self_t::num_axis]) const;

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

    //---- [Conversion SECTION] ----------------------------------------------------------------------------------------------
    //---- [Conversion SECTION] ----------------------------------------------------------------------------------------------
    //---- [Conversion SECTION] ----------------------------------------------------------------------------------------------

    template <typename K_tt>
    NEON_CUDA_HOST_DEVICE inline Vec_3d<K_tt> newType() const;

    //---- [to-String SECTION] ----------------------------------------------------------------------------------------------
    //---- [to-String SECTION] ----------------------------------------------------------------------------------------------
    //---- [to-String SECTION] ----------------------------------------------------------------------------------------------

    NEON_CUDA_HOST_DEVICE void to_printf(bool newLine = true) const;

    NEON_CUDA_HOST_DEVICE void to_printfLD(bool newLine = true) const;

    std::string to_string(const std::string& prefix) const;

    std::string to_string(int tab_num = 0) const;

    std::string to_stringForComposedNames() const;

    //---- [ForEach SECTION] ----------------------------------------------------------------------------------------------
    //---- [ForEach SECTION] ----------------------------------------------------------------------------------------------
    //---- [ForEach SECTION] ----------------------------------------------------------------------------------------------

    template <Neon::computeMode_t::computeMode_e computeMode_ta = Neon::computeMode_t::seq>
    static void forEach(const self_t& len, std::function<void(const self_t& idx)> lambda);

    template <Neon::computeMode_t::computeMode_e computeMode_ta = Neon::computeMode_t::seq>
    static void forEach(const self_t& len, std::function<void(Integer idxX, Integer idxY, Integer idxZ)> lambda);

    template <Neon::computeMode_t::computeMode_e computeMode_ta = Neon::computeMode_t::seq, class Lambda = void>
    auto forEach(const Lambda& lambda) const -> std::enable_if_t<std::is_invocable_v<Lambda, self_t> ||
                                                                     std::is_invocable_v<Lambda, Integer, Integer, Integer>,
                                                                 void>;
};


template <typename T_ta>
std::ostream& operator<<(std::ostream& out, const Integer_3d<T_ta>& p);

}  // namespace Neon
