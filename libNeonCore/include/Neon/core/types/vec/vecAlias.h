#pragma once
#include <Neon/core/types/BasicTypes.h>

namespace Neon {

using int8_1d = int8_t;
using int32_1d = int32_t;
using int64_1d = int64_t;
using uint8_1d = uint8_t;
using uint32_1d = uint32_t;
using uint64_1d = uint64_t;
using size_1d = std::size_t;
using index_1d = index_t;
using index64_1d = index64_t;
using float_1d = float;
using double_1d = double;

#if !defined(NEON_WARP_COMPILATION)
//---- [Real 2D SECTION] ----------------------------------------------------------------------------------------------
//---- [Real 2D SECTION] ----------------------------------------------------------------------------------------------
//---- [Real 2D SECTION] ----------------------------------------------------------------------------------------------

template <typename RealType_ta>
using Real_2d = Vec_2d<RealType_ta, false, true>;

using double_2d = Vec_2d<double>;
using float_2d = Vec_2d<float>;

//---- [Integer 2D SECTION] ----------------------------------------------------------------------------------------------
//---- [Integer 2D SECTION] ----------------------------------------------------------------------------------------------
//---- [Integer 2D SECTION] ----------------------------------------------------------------------------------------------

template <typename Integer_ta>
using Integer_2d = Vec_2d<Integer_ta, true, false>;


using int8_2d = Integer_2d<int8_t>;
using int32_2d = Integer_2d<int32_t>;
using int64_2d = Integer_2d<int64_t>;
using uint8_2d = Integer_2d<uint8_t>;
using uint32_2d = Integer_2d<uint32_t>;
using uint64_2d = Integer_2d<uint64_t>;
using size_2d = Integer_2d<std::size_t>;
using index_2d = Integer_2d<index_t>;
using index64_2d = Integer_2d<index64_1d>;
#endif


//---- [Real 3D SECTION] ----------------------------------------------------------------------------------------------
//---- [Real 3D SECTION] ----------------------------------------------------------------------------------------------
//---- [Real 3D SECTION] ----------------------------------------------------------------------------------------------

template <typename RealType_ta>
using Real_3d = Vec_3d<RealType_ta, false, true>;

using double_3d = Vec_3d<double>;
using float_3d = Vec_3d<float>;

//---- [Integer 3D SECTION] ----------------------------------------------------------------------------------------------
//---- [Integer 3D SECTION] ----------------------------------------------------------------------------------------------
//---- [Integer 3D SECTION] ----------------------------------------------------------------------------------------------

template <typename Integer_ta>
using Integer_3d = Vec_3d<Integer_ta, true, false>;

using int8_3d = Integer_3d<int8_t>;
using int32_3d = Integer_3d<int32_t>;
using int16_3d = Integer_3d<int16_t>;
using int64_3d = Integer_3d<int64_t>;
using uint8_3d = Integer_3d<uint8_t>;
using uint32_3d = Integer_3d<uint32_t>;
using uint64_3d = Integer_3d<uint64_t>;
using size_3d = Integer_3d<std::size_t>;
using index_3d = Integer_3d<index_t>;
using index64_3d = Integer_3d<index64_1d>;
using char_3d = Integer_3d<char>;

#if !defined(NEON_WARP_COMPILATION)

//---- [Real 4D SECTION] ----------------------------------------------------------------------------------------------
//---- [Real 4D SECTION] ----------------------------------------------------------------------------------------------
//---- [Real 4D SECTION] ----------------------------------------------------------------------------------------------

template <typename RealType_ta>
using Real_4d = Vec_4d<RealType_ta, false, true>;

using double_4d = Vec_4d<double>;
using float_4d = Vec_4d<float>;

//---- [Integer 4D SECTION] ----------------------------------------------------------------------------------------------
//---- [Integer 4D SECTION] ----------------------------------------------------------------------------------------------
//---- [Integer 4D SECTION] ----------------------------------------------------------------------------------------------

template <typename Integer_ta>
using Integer_4d = Vec_4d<Integer_ta, true, false>;

using int8_4d = Vec_4d<int8_t>;
using int32_4d = Vec_4d<int32_t>;
using int64_4d = Vec_4d<int64_t>;
using uint8_4d = Vec_4d<uint8_t>;
using uint32_4d = Vec_4d<uint32_t>;
using uint64_4d = Vec_4d<uint64_t>;
using size_4d = Vec_4d<std::size_t>;
using index_4d = Vec_4d<index_t, true, false>;
using char_4d = Vec_4d<char, true, false>;
#endif
}  // End of namespace Neon
