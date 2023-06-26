#pragma once 

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

    struct BitMask
    {
        using BitMaskWordType = uint32_t;
        auto reset() -> void
        {
            for (BitMaskWordType i = 0; i < nWords; ++i) {
                bits[i] = 0;
            }
        }

        auto setActive(int threadX,
                       int threadY,
                       int threadZ) -> void
        {
            BitMaskWordType mask;
            uint32_t        wordIdx;
            getMaskAndWordI(threadX, threadY, threadZ, mask, wordIdx);
            auto& word = bits[wordIdx];
            word = word | mask;
        }

        inline auto NEON_CUDA_HOST_DEVICE isActive(int threadX,
                                                   int threadY,
                                                   int threadZ) const -> bool
        {
            BitMaskWordType mask;
            uint32_t        wordIdx;
            getMaskAndWordI(threadX, threadY, threadZ, mask, wordIdx);
            auto& word = bits[wordIdx];
            return (word & mask) != 0;
        }

        static inline auto NEON_CUDA_HOST_DEVICE getMaskAndWordI(int                       threadX,
                                                                 int                       threadY,
                                                                 int                       threadZ,
                                                                 NEON_OUT BitMaskWordType& mask,
                                                                 NEON_OUT uint32_t&        wordIdx) -> void
        {
            const uint32_t threadPitch = threadX * memBlockPitchX +
                                         threadY * memBlockPitchY +
                                         threadZ * memBlockPitchZ;

            // threadPitch >> log2_of_bitPerWord
            // the same as: threadPitch / 2^{log2_of_bitPerWord}
            wordIdx = threadPitch >> log2_of_bitPerWord;
            // threadPitch & ((bitMaskWordType(bitMaskStorageBitWidth)) - 1);
            // same as threadPitch % 2^{log2OfbitMaskWordSize}
            const uint32_t offsetInWord = threadPitch & ((BitMaskWordType(bitPerWord)) - 1);
            mask = BitMaskWordType(1) << offsetInWord;
        }

        constexpr static BitMaskWordType nWords = (memBlockCountElements + 31) / 32;
        static constexpr uint32_t        log2_of_bitPerWord = 5;
        static constexpr uint32_t        bitPerWord = 32;

        BitMaskWordType bits[nWords];
    };
};

}  // namespace Neon::domain::details::bGrid