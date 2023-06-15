#include "Neon/domain/details/bGrid/bSpan.h"

namespace Neon::domain::details::bGrid {

template <uint32_t memBlockSizeX_,
          uint32_t memBlockSizeY_,
          uint32_t memBlockSizeZ_,
          uint32_t userBlockSizeX_ = memBlockSizeX_,
          uint32_t userBlockSizeY_ = memBlockSizeY_,
          uint32_t userBlockSizeZ_ = memBlockSizeZ_,
          bool     isMultiResMode_ = false>
struct StaticBlock
{
   public:
    constexpr static uint32_t        memBlockSizeX = memBlockSizeX_;
    constexpr static uint32_t        memBlockSizeY = memBlockSizeY_;
    constexpr static uint32_t        memBlockSizeZ = memBlockSizeZ_;
    constexpr static Neon::uint32_3d memBlockSize3D = Neon::uint32_3d(memBlockSizeX, memBlockSizeY, memBlockSizeZ);

    constexpr static uint32_t        userBlockSizeX = userBlockSizeX_;
    constexpr static uint32_t        userBlockSizeY = userBlockSizeY_;
    constexpr static uint32_t        userBlockSizeZ = userBlockSizeZ_;
    constexpr static Neon::uint32_3d userBlockSize3D = Neon::uint32_3d(userBlockSizeX, userBlockSizeY, userBlockSizeZ);

    constexpr static uint32_t blockRatioX = memBlockSizeX / userBlockSizeX;
    constexpr static uint32_t blockRatioY = memBlockSizeY / userBlockSizeY;
    constexpr static uint32_t blockRatioZ = memBlockSizeZ / userBlockSizeZ;

    constexpr static uint32_t memBlockPitchX = 1;
    constexpr static uint32_t memBlockPitchY = memBlockSizeX;
    constexpr static uint32_t memBlockPitchZ = memBlockSizeX * memBlockSizeY;

    constexpr static bool isMultiResMode = isMultiResMode_;

    constexpr static uint32_t memBlockCountElements = memBlockSizeX * memBlockSizeY * memBlockSizeZ;

    static_assert(memBlockSizeX >= userBlockSizeX);
    static_assert(memBlockSizeY >= userBlockSizeY);
    static_assert(memBlockSizeZ >= userBlockSizeZ);

    static_assert(memBlockSizeX % userBlockSizeX == 0);
    static_assert(memBlockSizeY % userBlockSizeY == 0);
    static_assert(memBlockSizeZ % userBlockSizeZ == 0);
};

}  // namespace Neon::domain::details::bGrid