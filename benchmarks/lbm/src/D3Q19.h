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
struct D3Q19
{
   public:
    D3Q19() = delete;

    static constexpr int Q = 19; /** number of directions */
    static constexpr int D = 3;  /** Space dimension */
    using Precision = Precision_;
    using Self = D3Q19<Precision>;

    static constexpr int RegisterMapping = 1;
    static constexpr int MemoryMapping = 2;

    struct Registers
    {
        using Self = D3Q19<Precision>::Registers;
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

        static constexpr int center = 9; /** Position of direction {0,0,0} */

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
            10, 11, 12, 13, 14, 15, 16, 17, 18, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};

        static constexpr std::array<const typename Precision::Storage, Q> t{
            1. / 18. /*!  0   */,
            1. / 18. /*!  1   */,
            1. / 18. /*!  2   */,
            1. / 36. /*!  3   */,
            1. / 36. /*!  4   */,
            1. / 36. /*!  5   */,
            1. / 36. /*!  6   */,
            1. / 36. /*!  7   */,
            1. / 36. /*!  8   */,
            1. / 3. /*!   9  */,
            1. / 18. /*!  10   */,
            1. / 18. /*!  11  */,
            1. / 18. /*!  12  */,
            1. / 36. /*!  13  */,
            1. / 36. /*!  14  */,
            1. / 36. /*!  15  */,
            1. / 36. /*!  16  */,
            1. / 36. /*!  17  */,
            1. / 36. /*!  18  */
        };

        template <int q>
        static constexpr auto getT() -> const typename Precision::Storage
        {
            return t[q];
        }

        template <int q>
        static constexpr auto getDirection() -> const typename Neon::index_3d
        {
            return stencil[q];
        }

        static constexpr int                                     fwdRegIdxListLen = (Q - 1) / 2;
        static constexpr std::array<const int, fwdRegIdxListLen> fwdRegIdxList{0, 1, 2, 3, 4, 5, 6, 7, 8};

        template <int tegIdx, typename Compute>
        static inline NEON_CUDA_HOST_DEVICE auto
        getCk_u(std::array<Compute, 3> const& u) -> Compute
        {
            if constexpr (tegIdx == 0 || tegIdx == 9) {
                return u[0];
            }
            if constexpr (tegIdx == 1 || tegIdx == 10) {
                return u[1];
            }
            if constexpr (tegIdx == 2 || tegIdx == 11) {
                return u[2];
            }
            if constexpr (tegIdx == 3 || tegIdx == 12) {
                return u[0] + u[1];
            }
            if constexpr (tegIdx == 4 || tegIdx == 13) {
                return u[0] - u[1];
            }
            if constexpr (tegIdx == 5 || tegIdx == 14) {
                return u[0] + u[2];
            }
            if constexpr (tegIdx == 6 || tegIdx == 15) {

                return u[0] - u[2];
            }
            if constexpr (tegIdx == 7 || tegIdx == 16) {

                return u[1] + u[2];
            }
            if constexpr (tegIdx == 8 || tegIdx == 17) {
                return u[1] - u[2];
            }
        }
    };

    struct Memory
    {
        using Self = D3Q19<Precision>::Memory;

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


        static constexpr int center = 9; /** Position of direction {0,0,0} */

        static constexpr std::array<const int, Q> memoryToRegister{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

        static constexpr std::array<const int, Q> registerToMemory{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};


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
            return opposite[go];
        }

        static constexpr std::array<const int, Q> opposite{
            10, 11, 12, 13, 14, 15, 16, 17, 18, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    };

//    template <int fwMemIdx_>
//    struct MemMapper
//    {
//        constexpr static int fwdMemIdx = fwMemIdx_;
//        constexpr static int fwdX = Memory::stencil[fwdMemIdx].x;
//        constexpr static int fwdY = Memory::stencil[fwdMemIdx].y;
//        constexpr static int fwdZ = Memory::stencil[fwdMemIdx].z;
//
//        constexpr static int bkwMemIdx = Memory::opposite[fwdMemIdx];
//        constexpr static int bkwX = Memory::stencil[bkwMemIdx].x;
//        constexpr static int bkwY = Memory::stencil[bkwMemIdx].y;
//        constexpr static int bkwZ = Memory::stencil[bkwMemIdx].z;
//
//        constexpr static int fwdRegIdx = Memory::template mapToRegisters<fwdMemIdx>();
//        constexpr static int centerRegIdx = Registers::center;
//        constexpr static int centerMemIdx = Memory::center;
//    };

    template <int fwdRegIdx_>
    struct RegisterMapper
    {
        constexpr static int fwdRegIdx = fwdRegIdx_;
        constexpr static int bkwRegIdx = Registers::opposite[fwdRegIdx];
        constexpr static int fwdMemIdx = Memory::template mapToMemory<fwdRegIdx>();
        constexpr static int bkwMemIdx = Memory::template mapToMemory<bkwRegIdx>();
        constexpr static int centerRegIdx = Registers::center;
        constexpr static int centerMemIdx = Memory::center;

        constexpr static int fwdX = Memory::stencil[fwdMemIdx].x;
        constexpr static int fwdY = Memory::stencil[fwdMemIdx].y;
        constexpr static int fwdZ = Memory::stencil[fwdMemIdx].z;

        constexpr static int bkwX = Memory::stencil[bkwMemIdx].x;
        constexpr static int bkwY = Memory::stencil[bkwMemIdx].y;
        constexpr static int bkwZ = Memory::stencil[bkwMemIdx].z;
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
};
