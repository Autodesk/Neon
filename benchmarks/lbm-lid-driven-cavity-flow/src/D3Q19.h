#pragma once

#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Neon/set/memory/memSet.h"
#include "Precision.h"

template <typename Precision_>
struct D3Q19
{
   public:
    static constexpr int Q = 19; /** number of directions */
    static constexpr int D = 3;  /** Space dimension */
    using Precision = Precision_;
    using Self = D3Q19<Precision>;
    static constexpr std::array<const Neon::index_3d, Q> stencil{
        Neon::index_3d(-1, 0, 0),
        Neon::index_3d(0, -1, 0),
        Neon::index_3d(0, 0, -1),
        Neon::index_3d(-1, -1, 0),
        Neon::index_3d(-1, 1, 0),
        Neon::index_3d(-1, 0, -1),
        Neon::index_3d(-1, 0, 1),
        Neon::index_3d(0, -1, -1),
        Neon::index_3d(0, -1, 1),
        Neon::index_3d(0, 0, 0),
        Neon::index_3d(1, 0, 0),
        Neon::index_3d(0, 1, 0),
        Neon::index_3d(0, 0, 1),
        Neon::index_3d(1, 1, 0),
        Neon::index_3d(1, -1, 0),
        Neon::index_3d(1, 0, 1),
        Neon::index_3d(1, 0, -1),
        Neon::index_3d(0, 1, 1),
        Neon::index_3d(0, 1, -1)};

    static constexpr int center = 9;       /** Position of direction {0,0,0} */
   private:
    static constexpr int goRangeBegin = 0; /** Symmetry is represented as "go" direction and the "back" their opposite */
    static constexpr int goRangeEnd = 8;
    static constexpr int goBackOffset = 10; /** Offset to compute apply symmetry */

   public:
    explicit D3Q19(const Neon::Backend& backend)
    {
    }

    template <int go>
    static constexpr auto getOpposite()
        -> int
    {
        if constexpr (go == center)
            return center;
        if constexpr (go <= goRangeEnd)
            return go + goBackOffset;
        if constexpr (go <= goRangeEnd + goBackOffset)
            return go - goBackOffset;
    }

    static constexpr std::array<const int, Q> opposite {
            Self::template getOpposite<0>(),
            Self::template getOpposite<1>(),
            Self::template getOpposite<2>(),
            Self::template getOpposite<3>(),
            Self::template getOpposite<4>(),
            Self::template getOpposite<5>(),
            Self::template getOpposite<6>(),
            Self::template getOpposite<7>(),
            Self::template getOpposite<8>(),
            Self::template getOpposite<9>(),
            Self::template getOpposite<10>(),
            Self::template getOpposite<11>(),
            Self::template getOpposite<12>(),
            Self::template getOpposite<13>(),
            Self::template getOpposite<14>(),
            Self::template getOpposite<15>(),
            Self::template getOpposite<16>(),
            Self::template getOpposite<17>(),
            Self::template getOpposite<18>()
    };


    static auto getDirectionAsVector()
        -> std::vector<Neon::index_3d>
    {
        std::vector<Neon::index_3d> vec;
        for (auto const& a : stencil) {
            vec.push_back(a);
        }
        return vec;
    }
};
