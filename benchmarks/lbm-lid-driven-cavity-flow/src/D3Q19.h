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
            Self::template getOpposite<18>()};

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
            1. / 36. /*!  18  */,
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


        static constexpr int center = 9;       /** Position of direction {0,0,0} */
        static constexpr int goRangeBegin = 0; /** Symmetry is represented as "go" direction and the "back" their opposite */
        static constexpr int goRangeEnd = 8;
        static constexpr int goBackOffset = 10; /** Offset to compute apply symmetry */

        template <int go>
        static constexpr auto mapToRegisters()
            -> int
        {
            auto direction = stencil[go];
            for (int i = 0; i < Q; ++i) {
                if (Registers::stencil[i] == direction) {
                    return i;
                }
            }
        }

        template <int go>
        static constexpr auto mapFromRegisters()
            -> int
        {
            auto direction = Registers::stencil[go];
            for (int i = 0; i < Q; ++i) {
                if (Self::stencil[i] == direction) {
                    return i;
                }
            }
        }

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
            Self::template getOpposite<18>()};

        template <int go>
        static constexpr auto helpGetValueforT()
            -> typename Precision::Storage
        {
            auto goInRegisterSpace = Self::template mapToRegisters<go>();
            return Registers::t[goInRegisterSpace];
        }

        static constexpr std::array<const typename Precision::Storage, Q> t{
            Self::template helpGetValueforT<0>(),
            Self::template helpGetValueforT<1>(),
            Self::template helpGetValueforT<2>(),
            Self::template helpGetValueforT<3>(),
            Self::template helpGetValueforT<4>(),
            Self::template helpGetValueforT<5>(),
            Self::template helpGetValueforT<6>(),
            Self::template helpGetValueforT<7>(),
            Self::template helpGetValueforT<8>(),
            Self::template helpGetValueforT<9>(),
            Self::template helpGetValueforT<10>(),
            Self::template helpGetValueforT<11>(),
            Self::template helpGetValueforT<12>(),
            Self::template helpGetValueforT<13>(),
            Self::template helpGetValueforT<14>(),
            Self::template helpGetValueforT<15>(),
            Self::template helpGetValueforT<16>(),
            Self::template helpGetValueforT<17>(),
            Self::template helpGetValueforT<18>()};
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
