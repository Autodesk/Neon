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

        static constexpr int center = 9; /** Position of direction {0,0,0} */

        template <int myQ, int myXYZ>
        static constexpr auto getVelocityComponent() -> int
        {
            static_assert(myQ < Q);
            static_assert(myXYZ < 3);

#define ADD_COMPONENT(QQ, XXX, YYY, ZZZ) \
    if constexpr ((myQ) == (QQ)) {       \
        if constexpr ((myXYZ) == 0) {    \
            return XXX;                  \
        }                                \
        if constexpr ((myXYZ) == 1) {    \
            return YYY;                  \
        }                                \
        if constexpr ((myXYZ) == 2) {    \
            return ZZZ;                  \
        }                                \
    }

            ADD_COMPONENT(0, -1, 0, 0)
            ADD_COMPONENT(1, 0, -1, 0)
            ADD_COMPONENT(2, 0, 0, -1)
            ADD_COMPONENT(3, -1, -1, 0)
            ADD_COMPONENT(4, -1, 1, 0)
            ADD_COMPONENT(5, -1, 0, -1)
            ADD_COMPONENT(6, -1, 0, 1)
            ADD_COMPONENT(7, 0, -1, -1)
            ADD_COMPONENT(8, 0, -1, 1)
            ADD_COMPONENT(9, 0, 0, 0)
            ADD_COMPONENT(10, 1, 0, 0)
            ADD_COMPONENT(11, 0, 1, 0)
            ADD_COMPONENT(12, 0, 0, 1)
            ADD_COMPONENT(13, 1, 1, 0)
            ADD_COMPONENT(14, 1, -1, 0)
            ADD_COMPONENT(15, 1, 0, 1)
            ADD_COMPONENT(16, 1, 0, -1)
            ADD_COMPONENT(17, 0, 1, 1)
            ADD_COMPONENT(18, 0, 1, -1)

#undef ADD_COMPONENT
        }

        template <int myQ>
        static constexpr auto getOpposite() -> int
        {
            static_assert(myQ < Q);

#define ADD_COMPONENT(QQ, XXX)     \
    if constexpr ((myQ) == (QQ)) { \
        return XXX;                \
    }
            ADD_COMPONENT(0, 10)
            ADD_COMPONENT(1, 11)
            ADD_COMPONENT(2, 12)
            ADD_COMPONENT(3, 13)
            ADD_COMPONENT(4, 14)
            ADD_COMPONENT(5, 15)
            ADD_COMPONENT(6, 16)
            ADD_COMPONENT(7, 17)
            ADD_COMPONENT(8, 18)
            ADD_COMPONENT(9, 9)
            ADD_COMPONENT(10, 0)
            ADD_COMPONENT(11, 1)
            ADD_COMPONENT(12, 2)
            ADD_COMPONENT(13, 3)
            ADD_COMPONENT(14, 4)
            ADD_COMPONENT(15, 5)
            ADD_COMPONENT(16, 6)
            ADD_COMPONENT(17, 7)
            ADD_COMPONENT(18, 8)
#undef ADD_COMPONENT
        }

        template <int myQ>
        static constexpr auto getT() -> typename Precision::Storage
        {
            static_assert(myQ < Q);

#define ADD_COMPONENT(QQ, XXX)     \
    if constexpr ((myQ) == (QQ)) { \
        return XXX;                \
    }

            ADD_COMPONENT(0, 1. / 18.)
            ADD_COMPONENT(1, 1. / 18.)
            ADD_COMPONENT(2, 1. / 18.)
            ADD_COMPONENT(3, 1. / 36.)
            ADD_COMPONENT(4, 1. / 36.)
            ADD_COMPONENT(5, 1. / 36.)
            ADD_COMPONENT(6, 1. / 36.)
            ADD_COMPONENT(7, 1. / 36.)
            ADD_COMPONENT(8, 1. / 36.)
            ADD_COMPONENT(9, 1. / 3.)
            ADD_COMPONENT(10, 1. / 18.)
            ADD_COMPONENT(11, 1. / 18.)
            ADD_COMPONENT(12, 1. / 18.)
            ADD_COMPONENT(13, 1. / 36.)
            ADD_COMPONENT(14, 1. / 36.)
            ADD_COMPONENT(15, 1. / 36.)
            ADD_COMPONENT(16, 1. / 36.)
            ADD_COMPONENT(17, 1. / 36.)
            ADD_COMPONENT(18, 1. / 36.)

#undef ADD_COMPONENT
        }


        template <int q>
        static constexpr auto getVelocity() -> const typename Neon::index_3d
        {
            return Neon::index_3d(getVelocityComponent<q, 0>,
                                  getVelocityComponent<q, 1>,
                                  getVelocityComponent<q, 2>);
        }

        // Identifying first half of the directions
        // For each direction in the list, the opposite is not present.
        // Center is also removed
        static constexpr int                                  firstHalfQLen = (Q - 1) / 2;
        static constexpr std::array<const int, firstHalfQLen> firstHalfQList{0, 1, 2, 3, 4, 5, 6, 7, 8};

        template <int tegIdx, typename Compute>
        static inline NEON_CUDA_HOST_DEVICE auto
        getCk_u(std::array<Compute, 3> const& u)
            -> Compute
        {
            if constexpr (tegIdx == 0 || tegIdx == 10) {
                return -u[0];
            }
            if constexpr (tegIdx == 1 || tegIdx == 11) {
                return -u[1];
            }
            if constexpr (tegIdx == 2 || tegIdx == 12) {
                return -u[2];
            }
            if constexpr (tegIdx == 3 || tegIdx == 13) {
                return -u[0] - u[1];
            }
            if constexpr (tegIdx == 4 || tegIdx == 14) {
                return -u[0] + u[1];
            }
            if constexpr (tegIdx == 5 || tegIdx == 15) {
                return -u[0] - u[2];
            }
            if constexpr (tegIdx == 6 || tegIdx == 16) {

                return -u[0] + u[2];
            }
            if constexpr (tegIdx == 7 || tegIdx == 17) {

                return -u[1] - u[2];
            }
            if constexpr (tegIdx == 8 || tegIdx == 18) {
                return -u[1] + u[2];
            }
        }
    };


    // Memory ------------------------------------------------------------------------------------------------

    struct Memory
    {
        using Self = D3Q19<Precision>::Memory;

        template <int myQ, int myXYZ>
        static constexpr auto getVelocityComponent() -> int
        {
            static_assert(myQ < Q);
            static_assert(myXYZ < 3);

#define ADD_COMPONENT(QQ, XXX, YYY, ZZZ) \
    if constexpr ((myQ) == (QQ)) {       \
        if constexpr ((myXYZ) == 0) {    \
            return XXX;                  \
        }                                \
        if constexpr ((myXYZ) == 1) {    \
            return YYY;                  \
        }                                \
        if constexpr ((myXYZ) == 2) {    \
            return ZZZ;                  \
        }                                \
    }

            ADD_COMPONENT(0, -1, 0, 0)
            ADD_COMPONENT(1, 0, -1, 0)
            ADD_COMPONENT(2, 0, 0, -1)
            ADD_COMPONENT(3, -1, 0, -1)
            ADD_COMPONENT(4, 0, -1, -1)
            ADD_COMPONENT(5, 1, 0, -1)
            ADD_COMPONENT(6, 0, 1, -1)
            ADD_COMPONENT(7, -1, -1, 0)
            ADD_COMPONENT(8, -1, 1, 0)
            ADD_COMPONENT(9, 0, 0, 0)
            ADD_COMPONENT(10, 1, 0, 0)
            ADD_COMPONENT(11, 0, 1, 0)
            ADD_COMPONENT(12, 1, 1, 0)
            ADD_COMPONENT(13, 1, -1, 0)
            ADD_COMPONENT(14, -1, 0, 1)
            ADD_COMPONENT(15, 0, -1, 1)
            ADD_COMPONENT(16, 0, 0, 1)
            ADD_COMPONENT(17, 1, 0, 1)
            ADD_COMPONENT(18, 0, 1, 1)

#undef ADD_COMPONENT
        }


        static constexpr int center = 9; /** Position of direction {0,0,0} */

        template <int myQ>
        static constexpr auto mapToRegisters()
            -> int
        {
            static_assert(myQ < Q);

#define ADD_COMPONENT(QQ, XXX)     \
    if constexpr ((myQ) == (QQ)) { \
        return XXX;                \
    }
            ADD_COMPONENT(0, 0)
            ADD_COMPONENT(1, 1)
            ADD_COMPONENT(2, 2)
            ADD_COMPONENT(3, 5)
            ADD_COMPONENT(4, 7)
            ADD_COMPONENT(5, 16)
            ADD_COMPONENT(6, 18)
            ADD_COMPONENT(7, 3)
            ADD_COMPONENT(8, 4)
            ADD_COMPONENT(9, 9)
            ADD_COMPONENT(10, 10)
            ADD_COMPONENT(11, 11)
            ADD_COMPONENT(12, 13)
            ADD_COMPONENT(13, 14)
            ADD_COMPONENT(14, 6)
            ADD_COMPONENT(15, 8)
            ADD_COMPONENT(16, 12)
            ADD_COMPONENT(17, 15)
            ADD_COMPONENT(18, 17)
#undef ADD_COMPONENT
        }

        template <int myQ>
        static constexpr auto mapToMemory()
            -> int
        {
            static_assert(myQ < Q);

#define ADD_COMPONENT(QQ, XXX)     \
    if constexpr ((myQ) == (QQ)) { \
        return XXX;                \
    }
            ADD_COMPONENT(0, 0)
            ADD_COMPONENT(1, 1)
            ADD_COMPONENT(2, 2)
            ADD_COMPONENT(5, 3)
            ADD_COMPONENT(7, 4)
            ADD_COMPONENT(16, 5)
            ADD_COMPONENT(18, 6)
            ADD_COMPONENT(3, 7)
            ADD_COMPONENT(4, 8)
            ADD_COMPONENT(9, 9)
            ADD_COMPONENT(10, 10)
            ADD_COMPONENT(11, 11)
            ADD_COMPONENT(13, 12)
            ADD_COMPONENT(14, 13)
            ADD_COMPONENT(6, 14)
            ADD_COMPONENT(8, 15)
            ADD_COMPONENT(12, 16)
            ADD_COMPONENT(15, 17)
            ADD_COMPONENT(17, 18)
#undef ADD_COMPONENT
        }

        template <int myQ>
        static constexpr auto getOpposite() -> int
        {
            static_assert(myQ < Q);

#define ADD_COMPONENT(QQ, XXX)     \
    if constexpr ((myQ) == (QQ)) { \
        return XXX;                \
    }
            ADD_COMPONENT(0, 10)
            ADD_COMPONENT(1, 11)
            ADD_COMPONENT(2, 16)
            ADD_COMPONENT(3, 17)
            ADD_COMPONENT(4, 18)
            ADD_COMPONENT(5, 14)
            ADD_COMPONENT(6, 15)
            ADD_COMPONENT(7, 12)
            ADD_COMPONENT(8, 13)
            ADD_COMPONENT(9, 9)
            ADD_COMPONENT(10, 0)
            ADD_COMPONENT(11, 1)
            ADD_COMPONENT(12, 7)
            ADD_COMPONENT(13, 8)
            ADD_COMPONENT(14, 5)
            ADD_COMPONENT(15, 6)
            ADD_COMPONENT(16, 2)
            ADD_COMPONENT(17, 3)
            ADD_COMPONENT(18, 4)
#undef ADD_COMPONENT
        }
    };


    template <int fwdRegIdx_>
    struct RegisterMapper
    {
        constexpr static int fwdRegQ = fwdRegIdx_;
        constexpr static int bkwRegQ = Registers::template getOpposite<fwdRegQ>();
        constexpr static int fwdMemQ = Memory::template mapToMemory<fwdRegQ>();
        constexpr static int bkwMemQ = Memory::template mapToMemory<bkwRegQ>();
        constexpr static int centerRegQ = Registers::center;
        constexpr static int centerMemQ = Memory::center;

        constexpr static int fwdMemQX = Memory::template getVelocityComponent<fwdMemQ, 0>();
        constexpr static int fwdMemQY = Memory::template getVelocityComponent<fwdMemQ, 1>();
        constexpr static int fwdMemQZ = Memory::template getVelocityComponent<fwdMemQ, 2>();

        constexpr static int bkwMemQX = Memory::template getVelocityComponent<bkwMemQ, 0>();
        constexpr static int bkwMemQY = Memory::template getVelocityComponent<bkwMemQ, 1>();
        constexpr static int bkwMemQZ = Memory::template getVelocityComponent<bkwMemQ, 2>();
    };

    template <int fwdMemIdx_>
    struct MemoryMapper
    {
        constexpr static int fwdMemQ = fwdMemIdx_;
        constexpr static int bkwMemQ = Memory::template getOpposite<fwdMemQ>();

        constexpr static int fwdRegQ = Memory::template mapToRegisters<fwdMemQ>();
        ;        constexpr static int bkwRegQ = Registers::template getOpposite<fwdRegQ>();

        constexpr static int centerRegQ = Registers::center;
        constexpr static int centerMemQ = Memory::center;

        constexpr static int fwdMemQX = Memory::template getVelocityComponent<fwdMemQ, 0>();
        constexpr static int fwdMemQY = Memory::template getVelocityComponent<fwdMemQ, 1>();
        constexpr static int fwdMemQZ = Memory::template getVelocityComponent<fwdMemQ, 2>();

        constexpr static int bkwMemQX = Memory::template getVelocityComponent<bkwMemQ, 0>();
        constexpr static int bkwMemQY = Memory::template getVelocityComponent<bkwMemQ, 1>();
        constexpr static int bkwMemQZ = Memory::template getVelocityComponent<bkwMemQ, 2>();
    };

   public:
    template <int mappingType>
    static auto getDirectionAsVector()
        -> std::vector<Neon::index_3d>
    {
        std::vector<Neon::index_3d> vec;
        if constexpr (mappingType == RegisterMapping) {
            Neon::ConstexprFor<0, Q, 1>(
                [&vec](auto q) {
                    Neon::index_3d val(Registers::template getVelocityComponent<q, 0>(),
                                       Registers::template getVelocityComponent<q, 1>(),
                                       Registers::template getVelocityComponent<q, 2>());
                    vec.push_back(val);
                });
        } else if constexpr (mappingType == MemoryMapping) {
            Neon::ConstexprFor<0, Q, 1>(
                [&vec](auto q) {
                    Neon::index_3d val(Memory::template getVelocityComponent<q, 0>(),
                                       Memory::template getVelocityComponent<q, 1>(),
                                       Memory::template getVelocityComponent<q, 2>());
                    vec.push_back(val);
                });
        }
        return vec;
    }
};