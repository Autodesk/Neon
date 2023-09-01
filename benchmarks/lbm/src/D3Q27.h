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

        template <int qIdx, int cIdx>
        static constexpr inline NEON_CUDA_HOST_DEVICE auto
        getComponentOfDirection() -> int
        {
            return stencil[qIdx].v[cIdx];
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

        static constexpr std::array<const typename Precision::Storage, Q> t{
            2. / 27., 2. / 27., 2. / 27., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54.,
            1. / 216., 1. / 216., 1. / 216., 1. / 216.,
            8. / 27.,
            2. / 27., 2. / 27., 2. / 27., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54., 1. / 54.,
            1. / 216., 1. / 216., 1. / 216., 1. / 216.};

        template <int qIdx>
        static inline NEON_CUDA_HOST_DEVICE auto
        getWeightOfDirection() -> int
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
            int v[6];
            Moment(int a0, int a1, int a2, int a3, int a4, int a5)
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
            {1, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 1},
            {1, 1, 0, 1, 0, 0},
            {1, -1, 0, 1, 0, 0},
            {1, 0, 1, 0, 0, 1},
            {1, 0, -1, 0, 0, 1},
            {0, 0, 0, 1, 1, 1},
            {0, 0, 0, 1, -1, 1},
            {1, 1, 1, 1, 1, 1},
            {1, 1, -1, 1, -1, 1},
            {1, -1, 1, 1, -1, 1},
            {1, -1, -1, 1, 1, 1},
            {0, 0, 0, 0, 0, 0},
            {1, 0, 0, 0, 0, 0},
            {0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 0, 1},
            {1, 1, 0, 1, 0, 0},
            {1, -1, 0, 1, 0, 0},
            {1, 0, 1, 0, 0, 1},
            {1, 0, -1, 0, 0, 1},
            {0, 0, 0, 1, 1, 1},
            {0, 0, 0, 1, -1, 1},
            {1, 1, 1, 1, 1, 1},
            {1, 1, -1, 1, -1, 1},
            {1, -1, 1, 1, -1, 1},
            {1, -1, -1, 1, 1, 1}};

        template <int qIdx, int mIdx>
        static constexpr inline NEON_CUDA_HOST_DEVICE auto
        getMomentByDirection() -> int
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
