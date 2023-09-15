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

        static constexpr int center = 13; /** Position of direction {0,0,0} */
                                          // Identifying first half of the directions
        // For each direction in the list, the opposite is not present.
        // Center is also removed
        static constexpr int                                  firstHalfQLen = (Q - 1) / 2;
        static constexpr std::array<const int, firstHalfQLen> firstHalfQList{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

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
            ADD_COMPONENT(9, -1, -1, -1)
            ADD_COMPONENT(10, -1, -1, 1)
            ADD_COMPONENT(11, -1, 1, -1)
            ADD_COMPONENT(12, -1, 1, 1)
            ADD_COMPONENT(13, 0, 0, 0)
            ADD_COMPONENT(14, 1, 0, 0)
            ADD_COMPONENT(15, 0, 1, 0)
            ADD_COMPONENT(16, 0, 0, 1)
            ADD_COMPONENT(17, 1, 1, 0)
            ADD_COMPONENT(18, 1, -1, 0)
            ADD_COMPONENT(19, 1, 0, 1)
            ADD_COMPONENT(20, 1, 0, -1)
            ADD_COMPONENT(21, 0, 1, 1)
            ADD_COMPONENT(22, 0, 1, -1)
            ADD_COMPONENT(23, 1, 1, 1)
            ADD_COMPONENT(24, 1, 1, -1)
            ADD_COMPONENT(25, 1, -1, 1)
            ADD_COMPONENT(26, 1, -1, -1)

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


            ADD_COMPONENT(0, 14)
            ADD_COMPONENT(1, 15)
            ADD_COMPONENT(2, 16)
            ADD_COMPONENT(3, 17)
            ADD_COMPONENT(4, 18)
            ADD_COMPONENT(5, 19)
            ADD_COMPONENT(6, 20)
            ADD_COMPONENT(7, 21)
            ADD_COMPONENT(8, 22)
            ADD_COMPONENT(9, 23)
            ADD_COMPONENT(10, 24)
            ADD_COMPONENT(11, 25)
            ADD_COMPONENT(12, 26)
            ADD_COMPONENT(13, 13)
            ADD_COMPONENT(14, 0)
            ADD_COMPONENT(15, 1)
            ADD_COMPONENT(16, 2)
            ADD_COMPONENT(17, 3)
            ADD_COMPONENT(18, 4)
            ADD_COMPONENT(19, 5)
            ADD_COMPONENT(20, 6)
            ADD_COMPONENT(21, 7)
            ADD_COMPONENT(22, 8)
            ADD_COMPONENT(23, 9)
            ADD_COMPONENT(24, 10)
            ADD_COMPONENT(25, 11)
            ADD_COMPONENT(26, 12)


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

            ADD_COMPONENT(0, 2. / 27.)
            ADD_COMPONENT(1, 2. / 27.)
            ADD_COMPONENT(2, 2. / 27.)
            ADD_COMPONENT(3, 1. / 54.)
            ADD_COMPONENT(4, 1. / 54.)
            ADD_COMPONENT(5, 1. / 54.)
            ADD_COMPONENT(6, 1. / 54.)
            ADD_COMPONENT(7, 1. / 54.)
            ADD_COMPONENT(8, 1. / 54.)
            ADD_COMPONENT(9, 1. / 216.)
            ADD_COMPONENT(10, 1. / 216.)
            ADD_COMPONENT(11, 1. / 216.)
            ADD_COMPONENT(12, 1. / 216.)
            ADD_COMPONENT(13, 8. / 27.)
            ADD_COMPONENT(14, 2. / 27.)
            ADD_COMPONENT(15, 2. / 27.)
            ADD_COMPONENT(16, 2. / 27.)
            ADD_COMPONENT(17, 1. / 54.)
            ADD_COMPONENT(18, 1. / 54.)
            ADD_COMPONENT(19, 1. / 54.)
            ADD_COMPONENT(20, 1. / 54.)
            ADD_COMPONENT(21, 1. / 54.)
            ADD_COMPONENT(22, 1. / 54.)
            ADD_COMPONENT(23, 1. / 216.)
            ADD_COMPONENT(24, 1. / 216.)
            ADD_COMPONENT(25, 1. / 216.)
            ADD_COMPONENT(26, 1. / 216.)

#undef ADD_COMPONENT
        }

        template <int myQ, int mementumID>
        static constexpr auto getMomentumComponet() -> typename Precision::Storage
        {
            static_assert(myQ < Q);
            static_assert(mementumID < 6);

#define ADD_COMPONENT(QQ, AA, BB, CC, DD, EE, FF) \
    if constexpr ((myQ) == (QQ)) {                \
        if constexpr ((mementumID) == 0) {        \
            return AA;                            \
        }                                         \
        if constexpr ((mementumID) == 1) {        \
            return BB;                            \
        }                                         \
        if constexpr ((mementumID) == 2) {        \
            return CC;                            \
        }                                         \
        if constexpr ((mementumID) == 3) {        \
            return DD;                            \
        }                                         \
        if constexpr ((mementumID) == 4) {        \
            return EE;                            \
        }                                         \
        if constexpr ((mementumID) == 5) {        \
            return FF;                            \
        }                                         \
    }

            ADD_COMPONENT(0, 1, 0, 0, 0, 0, 0)
            ADD_COMPONENT(1, 0, 0, 0, 1, 0, 0)
            ADD_COMPONENT(2, 0, 0, 0, 0, 0, 1)
            ADD_COMPONENT(3, 1, 1, 0, 1, 0, 0)
            ADD_COMPONENT(4, 1, -1, 0, 1, 0, 0)
            ADD_COMPONENT(5, 1, 0, 1, 0, 0, 1)
            ADD_COMPONENT(6, 1, 0, -1, 0, 0, 1)
            ADD_COMPONENT(7, 0, 0, 0, 1, 1, 1)
            ADD_COMPONENT(8, 0, 0, 0, 1, -1, 1)
            ADD_COMPONENT(9, 1, 1, 1, 1, 1, 1)
            ADD_COMPONENT(10, 1, 1, -1, 1, -1, 1)
            ADD_COMPONENT(11, 1, -1, 1, 1, -1, 1)
            ADD_COMPONENT(12, 1, -1, -1, 1, 1, 1)
            ADD_COMPONENT(13, 0, 0, 0, 0, 0, 0)
            ADD_COMPONENT(14, 1, 0, 0, 0, 0, 0)
            ADD_COMPONENT(15, 0, 0, 0, 1, 0, 0)
            ADD_COMPONENT(16, 0, 0, 0, 0, 0, 1)
            ADD_COMPONENT(17, 1, 1, 0, 1, 0, 0)
            ADD_COMPONENT(18, 1, -1, 0, 1, 0, 0)
            ADD_COMPONENT(19, 1, 0, 1, 0, 0, 1)
            ADD_COMPONENT(20, 1, 0, -1, 0, 0, 1)
            ADD_COMPONENT(21, 0, 0, 0, 1, 1, 1)
            ADD_COMPONENT(22, 0, 0, 0, 1, -1, 1)
            ADD_COMPONENT(23, 1, 1, 1, 1, 1, 1)
            ADD_COMPONENT(24, 1, 1, -1, 1, -1, 1)
            ADD_COMPONENT(25, 1, -1, 1, 1, -1, 1)
            ADD_COMPONENT(26, 1, -1, -1, 1, 1, 1)

#undef ADD_COMPONENT
        }


        template <int q>
        static constexpr auto getVelocity() -> const typename Neon::index_3d
        {
            return Neon::index_3d(getVelocityComponent<q, 0>,
                                  getVelocityComponent<q, 1>,
                                  getVelocityComponent<q, 2>);
        }

        //        // Identifying first half of the directions
        //        // For each direction in the list, the opposite is not present.
        //        // Center is also removed
        //        static constexpr int                                  firstHalfQLen = (Q - 1) / 2;
        //        static constexpr std::array<const int, firstHalfQLen> firstHalfQList{0, 1, 2, 3, 4, 5, 6, 7, 8};
    };

    struct Memory
    {
        using Self = D3Q27<Precision>::Memory;

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
            ADD_COMPONENT(9, -1, -1, -1)
            ADD_COMPONENT(10, -1, -1, 1)
            ADD_COMPONENT(11, -1, 1, -1)
            ADD_COMPONENT(12, -1, 1, 1)
            ADD_COMPONENT(13, 0, 0, 0)
            ADD_COMPONENT(14, 1, 0, 0)
            ADD_COMPONENT(15, 0, 1, 0)
            ADD_COMPONENT(16, 0, 0, 1)
            ADD_COMPONENT(17, 1, 1, 0)
            ADD_COMPONENT(18, 1, -1, 0)
            ADD_COMPONENT(19, 1, 0, 1)
            ADD_COMPONENT(20, 1, 0, -1)
            ADD_COMPONENT(21, 0, 1, 1)
            ADD_COMPONENT(22, 0, 1, -1)
            ADD_COMPONENT(23, 1, 1, 1)
            ADD_COMPONENT(24, 1, 1, -1)
            ADD_COMPONENT(25, 1, -1, 1)
            ADD_COMPONENT(26, 1, -1, -1)

#undef ADD_COMPONENT
        }

        static constexpr int center = 13; /** Position of direction {0,0,0} */

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
            ADD_COMPONENT(3, 3)
            ADD_COMPONENT(4, 4)
            ADD_COMPONENT(5, 5)
            ADD_COMPONENT(6, 6)
            ADD_COMPONENT(7, 7)
            ADD_COMPONENT(8, 8)
            ADD_COMPONENT(9, 9)
            ADD_COMPONENT(10, 10)
            ADD_COMPONENT(11, 11)
            ADD_COMPONENT(12, 12)
            ADD_COMPONENT(13, 13)
            ADD_COMPONENT(14, 14)
            ADD_COMPONENT(15, 15)
            ADD_COMPONENT(16, 16)
            ADD_COMPONENT(17, 17)
            ADD_COMPONENT(18, 18)

            ADD_COMPONENT(19, 19)
            ADD_COMPONENT(20, 20)
            ADD_COMPONENT(21, 21)
            ADD_COMPONENT(22, 22)
            ADD_COMPONENT(23, 23)
            ADD_COMPONENT(24, 24)
            ADD_COMPONENT(25, 25)
            ADD_COMPONENT(26, 26)

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
            ADD_COMPONENT(3, 3)
            ADD_COMPONENT(4, 4)
            ADD_COMPONENT(5, 5)
            ADD_COMPONENT(6, 6)
            ADD_COMPONENT(7, 7)
            ADD_COMPONENT(8, 8)
            ADD_COMPONENT(9, 9)
            ADD_COMPONENT(10, 10)
            ADD_COMPONENT(11, 11)
            ADD_COMPONENT(12, 12)
            ADD_COMPONENT(13, 13)
            ADD_COMPONENT(14, 14)
            ADD_COMPONENT(15, 15)
            ADD_COMPONENT(16, 16)
            ADD_COMPONENT(17, 17)
            ADD_COMPONENT(18, 18)

            ADD_COMPONENT(19, 19)
            ADD_COMPONENT(20, 20)
            ADD_COMPONENT(21, 21)
            ADD_COMPONENT(22, 22)
            ADD_COMPONENT(23, 23)
            ADD_COMPONENT(24, 24)
            ADD_COMPONENT(25, 25)
            ADD_COMPONENT(26, 26)
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
            ADD_COMPONENT(0, 14)
            ADD_COMPONENT(1, 15)
            ADD_COMPONENT(2, 16)
            ADD_COMPONENT(3, 17)
            ADD_COMPONENT(4, 18)
            ADD_COMPONENT(5, 19)
            ADD_COMPONENT(6, 20)
            ADD_COMPONENT(7, 21)
            ADD_COMPONENT(8, 22)
            ADD_COMPONENT(9, 23)
            ADD_COMPONENT(10, 24)
            ADD_COMPONENT(11, 25)
            ADD_COMPONENT(12, 26)
            ADD_COMPONENT(13, 13)
            ADD_COMPONENT(14, 0)
            ADD_COMPONENT(15, 1)
            ADD_COMPONENT(16, 2)
            ADD_COMPONENT(17, 3)
            ADD_COMPONENT(18, 4)
            ADD_COMPONENT(19, 5)
            ADD_COMPONENT(20, 6)
            ADD_COMPONENT(21, 7)
            ADD_COMPONENT(22, 8)
            ADD_COMPONENT(23, 9)
            ADD_COMPONENT(24, 10)
            ADD_COMPONENT(25, 11)
            ADD_COMPONENT(26, 12)
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