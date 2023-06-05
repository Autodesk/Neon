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
template <typename GenericBaseType_ta,
          bool IsBaseTypeInteger = std::is_integral<GenericBaseType_ta>::value,
          bool IsBaseTypeReal = std::is_floating_point<GenericBaseType_ta>::value>
class Vec_3d
{
    using Integer = GenericBaseType_ta;
    using self_t = Vec_3d<Integer, IsBaseTypeInteger, IsBaseTypeReal>;

    enum axis_e
    {
        x_axis = 0,
        y_axis = 1,
        z_axis = 2,
        num_axis = 3
    };

    static_assert(!IsBaseTypeInteger, "");
    static_assert(!IsBaseTypeReal, "");
};


/**
 * Generic partial specialization of the Vec_3d, where the userValue is not a number
 */
template <typename notAnumber_eValue_ta>
class Vec_3d<notAnumber_eValue_ta, false, false>
{
   public:
    using Integer = notAnumber_eValue_ta;
    using self_t = Vec_3d<Integer, false, false>;

    enum axis_e
    {
        x_axis = 0,
        y_axis = 1,
        z_axis = 2,
        num_axis = 3
    };

    union
    {
        Integer v[self_t::num_axis];
        struct
        {
            union
            {
                Integer x;
                Integer r;
            };
            union
            {
                Integer y;
                Integer s;
            };
            union
            {
                Integer z;
                Integer t;
            };
        };
    };


    /**
     * Default constructor.
     */
    NEON_CUDA_HOST_DEVICE Vec_3d()
    {
        static_assert(sizeof(self_t) == sizeof(Integer) * axis_e::num_axis, "");
    };

    /**
     * Default destructor.
     */
    ~Vec_3d() = default;

    /**
     * All components are set to the same value.
     * @param[in] xyz: selected values for all the point components.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_3d(const Integer& xyz)
        : x(xyz), y(xyz), z(xyz)
    {
    }
    /**
     * Creates a 3d tuple with specific values for each component.
     *   @param[in] px: value for the x component.
     *   @param[in] py: value for the y component.
     *   @param[in] pz: value for the z component.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_3d(const Integer& px, const Integer& py, const Integer& pz)
    {
        set(px, py, pz);
    }

    /**
     * copy operator.
     * @param xyz: element to be copied.
     */
    NEON_CUDA_HOST_DEVICE inline self_t& operator=(const self_t& xyz)
    {
        set(xyz);
        return *this;
    }

    NEON_CUDA_HOST_DEVICE inline void set(const self_t& xyz)
    {
        x = xyz.x;
        y = xyz.y;
        z = xyz.z;
    }

    NEON_CUDA_HOST_DEVICE inline void set(const Integer& px, const Integer& py, const Integer& pz)
    {
        x = px;
        y = py;
        z = pz;
    }


    template <typename otherValue_ta>
    NEON_CUDA_HOST_DEVICE inline Vec_3d<otherValue_ta> newType() const
    {
        const self_t&         A = *this;
        Vec_3d<otherValue_ta> C;
        ////
        C.x = static_cast<otherValue_ta>(A.x);
        C.y = static_cast<otherValue_ta>(A.y);
        C.z = static_cast<otherValue_ta>(A.z);
        ////
        return C;
    }
};

}  // namespace Neon
