#pragma once
#include "Neon/Neon.h"

namespace Neon::domain::tool::spaceCurves {


enum struct EncoderType
{
    sweep = 0,
    morton = 1,
    hilbert = 2,
};


/**
 * Set of utilities for DataView options.
 */
struct EncoderTypeUtil
{
    /**
     * Number of configurations for the enum
     */
    static const int nConfig{static_cast<int>(3)};

    /**
     * Convert enum value to string
     *
     * @param dataView
     * @return
     */
    static auto toString(EncoderType encoderType) -> std::string;

    /**
     * Returns all valid configuration for DataView
     * @return
     */
    static auto validOptions() -> std::array<EncoderType, DataViewUtil::nConfig>;

    static auto fromInt(int val) -> EncoderType;

    static auto toInt(EncoderType encoderType) -> int;
};


/**
 * operator<<
 *
 * @param os
 * @param m
 * @return
 */
std::ostream& operator<<(std::ostream& os, Neon::DataView const& m);

class Encoder
{
   private:
    static constexpr uint8_t mortonToHilbertTable[] = {
        48,
        33,
        27,
        34,
        47,
        78,
        28,
        77,
        66,
        29,
        51,
        52,
        65,
        30,
        72,
        63,
        76,
        95,
        75,
        24,
        53,
        54,
        82,
        81,
        18,
        3,
        17,
        80,
        61,
        4,
        62,
        15,
        0,
        59,
        71,
        60,
        49,
        50,
        86,
        85,
        84,
        83,
        5,
        90,
        79,
        56,
        6,
        89,
        32,
        23,
        1,
        94,
        11,
        12,
        2,
        93,
        42,
        41,
        13,
        14,
        35,
        88,
        36,
        31,
        92,
        37,
        87,
        38,
        91,
        74,
        8,
        73,
        46,
        45,
        9,
        10,
        7,
        20,
        64,
        19,
        70,
        25,
        39,
        16,
        69,
        26,
        44,
        43,
        22,
        55,
        21,
        68,
        57,
        40,
        58,
        67,
    };

    static constexpr uint8_t hilbertToMortonTable[] = {
        48,
        33,
        35,
        26,
        30,
        79,
        77,
        44,
        78,
        68,
        64,
        50,
        51,
        25,
        29,
        63,
        27,
        87,
        86,
        74,
        72,
        52,
        53,
        89,
        83,
        18,
        16,
        1,
        5,
        60,
        62,
        15,
        0,
        52,
        53,
        57,
        59,
        87,
        86,
        66,
        61,
        95,
        91,
        81,
        80,
        2,
        6,
        76,
        32,
        2,
        6,
        12,
        13,
        95,
        91,
        17,
        93,
        41,
        40,
        36,
        38,
        10,
        11,
        31,
        14,
        79,
        77,
        92,
        88,
        33,
        35,
        82,
        70,
        10,
        11,
        23,
        21,
        41,
        40,
        4,
        19,
        25,
        29,
        47,
        46,
        68,
        64,
        34,
        45,
        60,
        62,
        71,
        67,
        18,
        16,
        49,
    };

    static inline auto transformCurve(uint64_t in, uint64_t bits, const uint8_t* lookupTable)
    {
        uint64_t transform = 0;
        uint64_t out = 0;

        for (int32_t i = 3 * (bits - 1); i >= 0; i -= 3) {
            transform = lookupTable[transform | ((in >> i) & 7)];
            out = (out << 3) | (transform & 7);
            transform &= ~7;
        }

        return out;
    }

    static inline auto mortonToHilbert3D(uint64_t mortonIndex, uint64_t bits)
    {
        return transformCurve(mortonIndex, bits, mortonToHilbertTable);
    }

    static inline auto hilbertToMorton3D(uint64_t hilbertIndex, uint64_t bits)
    {
        return transformCurve(hilbertIndex, bits, hilbertToMortonTable);
    }


    static inline auto splitBy3(unsigned int a)
    {
        uint64_t x = a & 0x1fffff;              // we only care about 21 bits
        x = (x | x << 32) & 0x1f00000000ffff;   // shift left 32 bits, mask out bits 21-31
        x = (x | x << 16) & 0x1f0000ff0000ff;   // shift left 16 bits, mask out bits 11-20, 43-52
        x = (x | x << 8) & 0x100f00f00f00f00f;  // shift left 8 bits, mask out bits 5-10, 21-26, 37-42, 53-58
        x = (x | x << 4) & 0x10c30c30c30c30c3;  // shift left 4 bits, mask out bits 3-4, 11-12, 19-20, 27-28, 35-36, 43-44, 51-52, 59-60
        x = (x | x << 2) & 0x1249249249249249;  // shift left 2 bits, mask out bits 2, 6-7, 10, 14-15, 18, 22-23, 26, 30-31, 34, 38-39, 42, 46-47, 50, 54-55, 58
        return x;
    }

   public:

    static inline auto mortonEncode([[maybe_unused]] Neon::index_3d dim, Neon::index_3d idx)
        -> uint64_t
    {
        auto idxU64 = idx.newType<uint64_t>();
        return splitBy3(idxU64.x) | (splitBy3(idxU64.y) << 1) | (splitBy3(idxU64.z) << 2);
    }

    static inline auto encodeHilbert(Neon::index_3d dim, Neon::index_3d idx)
        -> uint64_t
    {
        uint64_t mortonEncoded = mortonEncode(dim, idx);
        uint64_t bits = std::ceil(std::log2(dim.newType<uint64_t>().rMax()));
        return mortonToHilbert3D(mortonEncoded, bits);
    }

    static inline auto encodeSweep(Neon::index_3d dim, Neon::index_3d idx)
        -> uint64_t
    {
        auto idxU64 = idx.newType<uint64_t>();
        auto dimU64 = dim.newType<uint64_t>();

        uint64_t res = idxU64.x + idxU64.y * dimU64.x + idxU64.z * dimU64.x * dimU64.y;
        return res;
    }

    static inline auto encode(EncoderType type, Neon::index_3d dim, Neon::index_3d idx){
        switch (type) {
            case EncoderType::morton:
                return mortonEncode(dim, idx);
            case EncoderType::hilbert:
                return encodeHilbert(dim, idx);
            case EncoderType::sweep:
                return encodeSweep(dim, idx);
            default:
                NEON_THROW_UNSUPPORTED_OPERATION("Encoder type not supported");
        }
    }
};
}  // namespace Neon::domain::tool::spaceCurves
