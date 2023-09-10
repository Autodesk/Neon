#pragma once

#include "Neon/Neon.h"
#include "Neon/set/Backend.h"
#include "Neon/set/memory/memSet.h"
#include "Precision.h"


/** In each lattice we define two indexing schema
 * - the first one works at the register level. For this one we relay on the same sequence of directions defined by the STLBM code.
 * - the second one works at the memory (RAM) level and it defines how lattice direction are stored in Neon fields.
 *
 * We keep this two aspect separate to experiment on different order of directions in memory as the order will impact the number of halo update transitions.
 *
 */
template <typename Precision_>
struct D3Q27
{
   public:
    D3Q27() = delete;

    static constexpr int Q = 27; /** number of directions */
    static constexpr int D = 3;  /** Space dimension */
    using Precision = Precision_;
    using Self = D3Q27<Precision>;

    static constexpr int RegisterMapping = 1;
    static constexpr int MemoryMapping = 2;

    struct Registers
    {
        using Self = D3Q27<Precision>::Registers;
        static constexpr std::array<const Neon::index_3d, Q> stencil{
            /* 00 */ Neon::index_3d(-1, 0, 0),
            /* 01 */ Neon::index_3d(0, -1, 0),
            /* 02 */ Neon::index_3d(0, 0, -1),
            /* 03 */ Neon::index_3d(-1, -1, 0),
            /* 04 */ Neon::index_3d(-1, 1, 0),
            /* 05 */ Neon::index_3d(-1, 0, -1),
            /* 06 */ Neon::index_3d(-1, 0, 1),
            /* 07 */ Neon::index_3d(0, -1, -1),
            /* 08 */ Neon::index_3d(0, -1, 1),
            /* 09 */ Neon::index_3d(-1, -1, -1),
            /* 00 */ Neon::index_3d(-1, -1, 1),
            /* 11 */ Neon::index_3d(-1, 1, -1),
            /* 12 */ Neon::index_3d(-1, 1, 1),
            /* 13 */ Neon::index_3d(0, 0, 0),
            /* 14 */ Neon::index_3d(1, 0, 0),
            /* 15 */ Neon::index_3d(0, 1, 0),
            /* 16 */ Neon::index_3d(0, 0, 1),
            /* 17 */ Neon::index_3d(1, 1, 0),
            /* 18 */ Neon::index_3d(1, -1, 0),
            /* 19 */ Neon::index_3d(1, 0, 1),
            /* 20 */ Neon::index_3d(1, 0, -1),
            /* 21 */ Neon::index_3d(0, 1, 1),
            /* 22 */ Neon::index_3d(0, 1, -1),
            /* 23 */ Neon::index_3d(1, 1, 1),
            /* 24 */ Neon::index_3d(1, 1, -1),
            /* 25 */ Neon::index_3d(1, -1, 1),
            /* 26 */ Neon::index_3d(1, -1, -1)};

        template <int qIdx, int cIdx>
        static constexpr inline NEON_CUDA_HOST_DEVICE auto
        getComponentOfDirection() -> int
        {
            constexpr std::array<const Neon::index_3d, Q> s{
                /* 00 */ Neon::index_3d(-1, 0, 0),
                /* 01 */ Neon::index_3d(0, -1, 0),
                /* 02 */ Neon::index_3d(0, 0, -1),
                /* 03 */ Neon::index_3d(-1, -1, 0),
                /* 04 */ Neon::index_3d(-1, 1, 0),
                /* 05 */ Neon::index_3d(-1, 0, -1),
                /* 06 */ Neon::index_3d(-1, 0, 1),
                /* 07 */ Neon::index_3d(0, -1, -1),
                /* 08 */ Neon::index_3d(0, -1, 1),
                /* 09 */ Neon::index_3d(-1, -1, -1),
                /* 00 */ Neon::index_3d(-1, -1, 1),
                /* 11 */ Neon::index_3d(-1, 1, -1),
                /* 12 */ Neon::index_3d(-1, 1, 1),
                /* 13 */ Neon::index_3d(0, 0, 0),
                /* 14 */ Neon::index_3d(1, 0, 0),
                /* 15 */ Neon::index_3d(0, 1, 0),
                /* 16 */ Neon::index_3d(0, 0, 1),
                /* 17 */ Neon::index_3d(1, 1, 0),
                /* 18 */ Neon::index_3d(1, -1, 0),
                /* 19 */ Neon::index_3d(1, 0, 1),
                /* 20 */ Neon::index_3d(1, 0, -1),
                /* 21 */ Neon::index_3d(0, 1, 1),
                /* 22 */ Neon::index_3d(0, 1, -1),
                /* 23 */ Neon::index_3d(1, 1, 1),
                /* 24 */ Neon::index_3d(1, 1, -1),
                /* 25 */ Neon::index_3d(1, -1, 1),
                /* 26 */ Neon::index_3d(1, -1, -1)};

            return s[qIdx].v[cIdx];
        }

        static constexpr int center = 13; /** Position of direction {0,0,0} */

        template <int go>
        static constexpr auto getOpposite()
            -> int
        {
            auto opposite3d = stencil[go] * -1;
            for (int i = 0; i < Q; ++i) {
                if (stencil[i] == opposite3d) {
                    return i;
                }
            }
        }

        static constexpr std::array<const int, Q> opposite{
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            13,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        static constexpr std::array<const typename Precision::Compute, Q> t{
            /* 00 */ 2. / 27.,
            /* 01 */ 2. / 27.,
            /* 02 */ 2. / 27.,
            /* 03 */ 1. / 54.,
            /* 04 */ 1. / 54.,
            /* 05 */ 1. / 54.,
            /* 06 */ 1. / 54.,
            /* 07 */ 1. / 54.,
            /* 08 */ 1. / 54.,
            /* 09 */ 1. / 216.,
            /* 00 */ 1. / 216.,
            /* 11 */ 1. / 216.,
            /* 12 */ 1. / 216.,
            /* 13 */ 8. / 27.,
            /* 14 */ 2. / 27.,
            /* 15 */ 2. / 27.,
            /* 16 */ 2. / 27.,
            /* 17 */ 1. / 54.,
            /* 18 */ 1. / 54.,
            /* 19 */ 1. / 54.,
            /* 20 */ 1. / 54.,
            /* 21 */ 1. / 54.,
            /* 22 */ 1. / 54.,
            /* 23 */ 1. / 216.,
            /* 24 */ 1. / 216.,
            /* 25 */ 1. / 216.,
            /* 26 */ 1. / 216.};

        template <int qIdx>
        static inline NEON_CUDA_HOST_DEVICE auto
        getWeightOfDirection() -> const typename Precision::Compute
        {
            return t[qIdx];
        }

        template <int q>
        static constexpr NEON_CUDA_HOST_DEVICE auto getT() -> const typename Precision::Storage
        {
            return t[q];
        }

        template <int q>
        static constexpr NEON_CUDA_HOST_DEVICE auto getDirection() -> const typename Neon::index_3d
        {
            return stencil[q];
        }
        // Identifying first half of the directions
        // For each direction in the list, the opposite is not present.
        // Center is also removed
        static constexpr int                                  firstHalfQLen = (Q - 1) / 2;
        static constexpr std::array<const int, firstHalfQLen> firstHalfQList{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        struct Moment
        {
            std::array<int, 6> v{0, 0, 0, 0, 0, 0};
            inline constexpr Moment(int a0, int a1, int a2, int a3, int a4, int a5)
            {
                v[0] = a0;
                v[1] = a1;
                v[2] = a2;
                v[3] = a3;
                v[4] = a4;
                v[5] = a5;
            }
        };

        static constexpr std::array<const Moment, Q> latticeMoment{
            Moment(1, 0, 0, 0, 0, 0),
            Moment(0, 0, 0, 1, 0, 0),
            Moment(0, 0, 0, 0, 0, 1),
            Moment(1, 1, 0, 1, 0, 0),
            Moment(1, -1, 0, 1, 0, 0),
            Moment(1, 0, 1, 0, 0, 1),
            Moment(1, 0, -1, 0, 0, 1),
            Moment(0, 0, 0, 1, 1, 1),
            Moment(0, 0, 0, 1, -1, 1),
            Moment(1, 1, 1, 1, 1, 1),
            Moment(1, 1, -1, 1, -1, 1),
            Moment(1, -1, 1, 1, -1, 1),
            Moment(1, -1, -1, 1, 1, 1),
            Moment(0, 0, 0, 0, 0, 0),
            Moment(1, 0, 0, 0, 0, 0),
            Moment(0, 0, 0, 1, 0, 0),
            Moment(0, 0, 0, 0, 0, 1),
            Moment(1, 1, 0, 1, 0, 0),
            Moment(1, -1, 0, 1, 0, 0),
            Moment(1, 0, 1, 0, 0, 1),
            Moment(1, 0, -1, 0, 0, 1),
            Moment(0, 0, 0, 1, 1, 1),
            Moment(0, 0, 0, 1, -1, 1),
            Moment(1, 1, 1, 1, 1, 1),
            Moment(1, 1, -1, 1, -1, 1),
            Moment(1, -1, 1, 1, -1, 1),
            Moment(1, -1, -1, 1, 1, 1)};

        template <int qIdx, int mIdx>
        static constexpr inline NEON_CUDA_HOST_DEVICE auto
        getMomentByDirection()
            -> int
        {
            return latticeMoment[qIdx].v[mIdx];
        }
    };

    struct Memory
    {
        using Self = D3Q27<Precision>::Memory;
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
            Neon::index_3d(-1, -1, -1),
            Neon::index_3d(-1, -1, 1),
            Neon::index_3d(-1, 1, -1),
            Neon::index_3d(-1, 1, 1),
            Neon::index_3d(0, 0, 0),
            Neon::index_3d(1, 0, 0),
            Neon::index_3d(0, 1, 0),
            Neon::index_3d(0, 0, 1),
            Neon::index_3d(1, 1, 0),
            Neon::index_3d(1, -1, 0),
            Neon::index_3d(1, 0, 1),
            Neon::index_3d(1, 0, -1),
            Neon::index_3d(0, 1, 1),
            Neon::index_3d(0, 1, -1),
            Neon::index_3d(1, 1, 1),
            Neon::index_3d(1, 1, -1),
            Neon::index_3d(1, -1, 1),
            Neon::index_3d(1, -1, -1)};


        static constexpr int center = 13; /** Position of direction {0,0,0} */

        static constexpr std::array<const int, Q> memoryToRegister{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};

        static constexpr std::array<const int, Q> registerToMemory{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};

        template <int go>
        NEON_CUDA_HOST_DEVICE static constexpr auto mapToRegisters()
            -> int
        {
            return memoryToRegister[go];
        }

        template <int go>
        NEON_CUDA_HOST_DEVICE static constexpr auto mapToMemory()
            -> int
        {
            return registerToMemory[go];
        }

        template <int go>
        NEON_CUDA_HOST_DEVICE static constexpr auto getOpposite()
            -> int
        {
            auto opposite3d = stencil[go] * -1;
            for (int i = 0; i < Q; ++i) {
                if (stencil[i] == opposite3d) {
                    return i;
                }
            }
        }

        static constexpr std::array<const int, Q> opposite{
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            13,
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        template <int go>
        static constexpr auto helpGetValueforT()
            -> typename Precision::Storage
        {
            auto goInRegisterSpace = Self::template mapToRegisters<go>();
            return Registers::t[goInRegisterSpace];
        }

        static constexpr std::array<const typename Precision::Storage, Q> t{
            2. / 27., 2. / 27., 2. / 27., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54.,
            1. / 216., 1. / 216., 1. / 216., 1. / 216.,
            8. / 27.,
            2. / 27., 2. / 27., 2. / 27., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54.,
            1. / 216., 1. / 216., 1. / 216., 1. / 216.};
    };

   public:
    template <int mappingType>
    static auto getDirectionAsVector()
        -> std::vector<Neon::index_3d>
    {
        std::vector<Neon::index_3d> vec;
        if constexpr (mappingType == RegisterMapping) {
            for (auto const& a : Registers::stencil) {
                vec.push_back(a);
            }
        } else if constexpr (mappingType == MemoryMapping) {
            for (auto const& a : Memory::stencil) {
                vec.push_back(a);
            }
        }
        return vec;
    }

    template <int fwdRegIdx_>
    struct RegisterMapper
    {
        constexpr static int fwdRegQ = fwdRegIdx_;
        constexpr static int bkwRegQ = Registers::opposite[fwdRegQ];
        constexpr static int fwdMemQ = Memory::template mapToMemory<fwdRegQ>();
        constexpr static int bkwMemQ = Memory::template mapToMemory<bkwRegQ>();
        constexpr static int centerRegQ = Registers::center;
        constexpr static int centerMemQ = Memory::center;

        constexpr static int fwdMemQX = Memory::stencil[fwdMemQ].x;
        constexpr static int fwdMemQY = Memory::stencil[fwdMemQ].y;
        constexpr static int fwdMemQZ = Memory::stencil[fwdMemQ].z;

        constexpr static int bkwMemQX = Memory::stencil[bkwMemQ].x;
        constexpr static int bkwMemQY = Memory::stencil[bkwMemQ].y;
        constexpr static int bkwMemQZ = Memory::stencil[bkwMemQ].z;
    };
};
