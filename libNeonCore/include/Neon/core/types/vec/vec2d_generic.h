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
class Vec_2d
{
    using eValue_t = GenericBaseType_ta;
    using self_t = Vec_2d<eValue_t, IsBaseTypeInteger, IsBaseTypeReal>;

    enum axis_e
    {
        x_axis = 0,
        y_axis = 1,
        num_axis = 2
    };

    static_assert(!IsBaseTypeInteger, "");
    static_assert(!IsBaseTypeReal, "");
};


/**
 * Generic partial specialization of the Vec_2d, where the userValue is not a number
 */
template <typename notAnumber_eValue_ta>
class Vec_2d<notAnumber_eValue_ta, false, false>
{
   public:
    using eValue_t = notAnumber_eValue_ta;
    using self_t = Vec_2d<eValue_t, false, false>;

    enum axis_e
    {
        x_axis = 0,
        y_axis = 1,
        num_axis = 2
    };

    eValue_t x;
    eValue_t y;
    eValue_t z;


    /**
     * Default constructor.
     */
    NEON_CUDA_HOST_DEVICE Vec_2d()
    {
        static_assert(sizeof(self_t) == sizeof(eValue_t) * axis_e::num_axis, "");
    };

    /**
     * Default destructor.
     */
    ~Vec_2d() = default;

    /**
     * All components are set to the same value.
     * @param[in] xyz: selected values for all the point components.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_2d(const eValue_t& xyz)
        : x(xyz), y(xyz)
    {
        // Nothing to do...
    }
    /**
     * Creates a 3d tuple with specific values for each component.
     *   @param[in] px: value for the x component.
     *   @param[in] py: value for the y component.
     */
    NEON_CUDA_HOST_DEVICE inline Vec_2d(const eValue_t& px, const eValue_t& py)
        : x(px), y(py)
    {
        // Nothing to do...
    }

    /**
     * copy operator.
     * @param xyz: element to be copied.
     */
    NEON_CUDA_HOST_DEVICE inline self_t& operator=(const self_t& xyz)
    {
        this->set(xyz);
    };


    template <typename otherValue_ta>
    NEON_CUDA_HOST_DEVICE inline Vec_2d<otherValue_ta> newType() const
    {
        const self_t&         A = *this;
        Vec_2d<otherValue_ta> C;
        ////
        C.x = static_cast<otherValue_ta>(A.x);
        C.y = static_cast<otherValue_ta>(A.y);
        ////
        return C;
    }
};

}  // namespace Neon
