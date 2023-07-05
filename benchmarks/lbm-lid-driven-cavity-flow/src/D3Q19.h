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

        static constexpr std::array<const int, Q> toRegisters{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

        static constexpr std::array<const int, Q> toMemory{
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

        template <int go>
        NEON_CUDA_HOST_DEVICE static constexpr auto mapToRegisters()
            -> int
        {
            return toRegisters[go];
        }

        template <int go>
        NEON_CUDA_HOST_DEVICE static constexpr auto mapFromRegisters()
            -> int
        {
            return toMemory[go];
        }

        template <int go>
        NEON_CUDA_HOST_DEVICE static constexpr auto getOpposite()
            -> int
        {
            return opposite[go];
        }

        static constexpr std::array<const int, Q> opposite{
            10, 11, 12, 13, 14, 15, 16, 17, 18, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};

        template <int go>
        static constexpr auto helpGetValueforT()
            -> typename Precision::Storage
        {
            auto goInRegisterSpace = Self::template mapToRegisters<go>();
            return Registers::t[goInRegisterSpace];
        }

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
            1. / 36. /*!  18  */};

        template <int direction>
        NEON_CUDA_HOST_DEVICE static constexpr auto getT()
            -> typename Precision::Storage
        {
            return t[direction];
        }
        template <int direction>
        NEON_CUDA_HOST_DEVICE static constexpr auto getDirection()
            -> typename Neon::index_3d
        {
            return stencil[direction];
        }
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
