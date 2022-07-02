#pragma once

#include <string>
#include <type_traits>

#include "Neon/core/types/Macros.h"

namespace Neon {

/**
 * This class represent a tree element vector.
 * This is only a generic template class that is never going to be instantiated (see static asserts).
 * Indeed, there is a specialization for all the template values of this class.
 */
template <typename GenericBaseType_ta, bool IsBaseTypeInteger = std::is_integral<GenericBaseType_ta>::value, bool IsBaseTypeReal = std::is_floating_point<GenericBaseType_ta>::value>
class Vec_4d
{
    using element_t = GenericBaseType_ta;
    using self_t = Vec_3d<element_t, IsBaseTypeInteger, IsBaseTypeReal>;

    enum axis_e
    {
        x_axis = 0,
        y_axis = 1,
        z_axis = 2,
        w_axis = 2,
        num_axis = 4
    };

    static_assert(!IsBaseTypeInteger, "");
    static_assert(!IsBaseTypeReal, "");
};


/**
* Generic partial specialization of the Vec_3d, where the userValue is not a number 
*/
template <typename notAnumber_eValue_ta>
class Vec_4d<notAnumber_eValue_ta, false, false>
{
   public:
    using element_t = notAnumber_eValue_ta;
    using self_t = Vec_4d<element_t, false, false>;

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
        element_t v[self_t::num_axis];
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
    NEON_CUDA_HOST_DEVICE Vec_4d()
    {
        static_assert(sizeof(self_t) == sizeof(element_t) * axis_e::num_axis, "");
    };

    /**
     * Default destructor.
     */
    ~Vec_4d() = default;

    /**
     * All components are set to the same value.
     * @param[in] xyzw: selected values for all the point components.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_4d(const element_t& xyzw)
        : x(xyzw), y(xyzw), z(xyzw), w(xyzw)
    {
        set(xyzw);
    }
    /**
    * Creates a 3d tuple with specific values for each component.
    *   @param[in] px: value for the x component.
    *   @param[in] py: value for the y component.
    *   @param[in] pz: value for the z component.
	*   @param[in] pw: value for the w component.
    */
    NEON_CUDA_HOST_DEVICE inline Vec_4d(const element_t& px, const element_t& py, const element_t& pz, const element_t& pw)
    {
        set(px, py, pz, pw);
    }

    /**
     * copy operator.
     * @param xyzw: element to be copied. 
     */
    NEON_CUDA_HOST_DEVICE inline self_t& operator=(const self_t& xyzw)
    {
        set(xyzw);
        return *this;
    }

    NEON_CUDA_HOST_DEVICE inline void set(const self_t& xyzw)
    {
        x = xyzw.x;
        y = xyzw.y;
        z = xyzw.z;
        w = xyzw.w;
    }

    NEON_CUDA_HOST_DEVICE inline void set(const element_t& px, const element_t& py, const element_t& pz, const element_t& pw)
    {
        x = px;
        y = py;
        z = pz;
        w = pw;
    }


    template <typename otherValue_ta>
    NEON_CUDA_HOST_DEVICE inline Vec_4d<otherValue_ta> newType() const
    {
        const self_t&         A = *this;
        Vec_4d<otherValue_ta> C;
        ////
        C.x = static_cast<otherValue_ta>(A.x);
        C.y = static_cast<otherValue_ta>(A.y);
        C.z = static_cast<otherValue_ta>(A.z);
        C.w = static_cast<otherValue_ta>(A.w);
        ////
        return C;
    }
};

}  // namespace Neon
