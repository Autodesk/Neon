#pragma once
#include "CellType.h"
#include "D3Q19.h"
#include "Neon/Neon.h"
#include "Neon/set/Containter.h"


template <typename Precision_, typename Grid_>
struct DeviceD3Q19
{
    using Lattice = D3Q19<Precision_>;
    using Precision = Precision_;
    using Compute = typename Precision::Compute;
    using Storage = typename Precision::Storage;
    using Grid = Grid_;

    using PopField = typename Grid::template Field<Storage, Lattice::Q>;
    using CellTypeField = typename Grid::template Field<CellType, 1>;

    using Idx = typename PopField::Idx;
    using Rho = typename Grid::template Field<Storage, 1>;
    using U = typename Grid::template Field<Storage, 3>;


    static inline NEON_CUDA_HOST_DEVICE auto
    pullStream(Idx const&                          gidx,
               const uint32_t&                     wallBitFlag,
               typename PopField::Partition const& fin,
               NEON_OUT Storage                    popIn[Lattice::Q])
    {

        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto GOMemoryId) {
            if constexpr (GOMemoryId == Lattice::Memory::center) {
                popIn[Lattice::Registers::center] = fin(gidx, Lattice::Memory::center);
            } else {
                constexpr int BKMemoryId = Lattice::Memory::opposite[GOMemoryId];
                constexpr int BKx = Lattice::Memory::stencil[BKMemoryId].x;
                constexpr int BKy = Lattice::Memory::stencil[BKMemoryId].y;
                constexpr int BKz = Lattice::Memory::stencil[BKMemoryId].z;
                constexpr int GORegistersId = Lattice::Memory::template mapToRegisters<GOMemoryId>();

                if (wallBitFlag & (uint32_t(1) << GOMemoryId)) {
                    popIn[GORegistersId] =
                        fin(gidx, BKMemoryId) +
                        fin.template getNghData<BKx, BKy, BKz>(gidx, BKMemoryId)();
                } else {
                    popIn[GORegistersId] =
                        fin.template getNghData<BKx, BKy, BKz>(gidx, GOMemoryId)();
                }
            }
        });
    }

    static inline NEON_CUDA_HOST_DEVICE auto
    macroscopic(const Storage     pop[Lattice::Q],
                NEON_OUT Compute& rho,
                NEON_OUT std::array<Compute, 3>& u)
        -> void
    {

#define POP(IDX) static_cast<Compute>(pop[IDX])
        const Compute X_M1 = POP(0) + POP(3) + POP(4) + POP(5) + POP(6);
        const Compute X_P1 = POP(10) + POP(13) + POP(14) + POP(15) + POP(16);
        const Compute X_0 = POP(9) + POP(1) + POP(2) + POP(7) + POP(8) + POP(11) + POP(12) + POP(17) + POP(18);

        const Compute Y_M1 = POP(1) + POP(3) + POP(7) + POP(8) + POP(14);
        const Compute Y_P1 = POP(4) + POP(11) + POP(13) + POP(17) + POP(18);

        const Compute Z_M1 = POP(2) + POP(5) + POP(7) + POP(16) + POP(18);
        const Compute Z_P1 = POP(6) + POP(8) + POP(12) + POP(15) + POP(17);
#undef POP

        rho = X_M1 + X_P1 + X_0;
        u[0] = (X_P1 - X_M1) / rho;
        u[1] = (Y_P1 - Y_M1) / rho;
        u[2] = (Z_P1 - Z_M1) / rho;
    }


    static inline NEON_CUDA_HOST_DEVICE auto
    collideBgkUnrolled(Idx const&                    i /*!     Compute iterator   */,
                       const Storage                 pop[Lattice::Q],
                       Compute const&                rho /*!   Density            */,
                       std::array<Compute, 3> const& u /*!     Velocity           */,
                       Compute const&                usqr /*!  Usqr               */,
                       Compute const&                omega /*! Omega              */,
                       typename PopField::Partition& fOut /*!  Population         */)

        -> void
    {
        const Compute ck_u03 = u[0] + u[1];
        const Compute ck_u04 = u[0] - u[1];
        const Compute ck_u05 = u[0] + u[2];
        const Compute ck_u06 = u[0] - u[2];
        const Compute ck_u07 = u[1] + u[2];
        const Compute ck_u08 = u[1] - u[2];

        constexpr Compute c1over18 = 1. / 18.;
        constexpr Compute c1over36 = 1. / 36.;
        constexpr Compute c4dot5 = 4.5;
        constexpr Compute c3 = 3.;
        constexpr Compute c1 = 1.;
        constexpr Compute c6 = 6.;

        const Compute eq_00 = rho * c1over18 * (c1 - c6 * u[0] + c4dot5 * u[0] * u[0] - usqr);
        const Compute eq_01 = rho * c1over18 * (c1 - c6 * u[1] + c4dot5 * u[1] * u[1] - usqr);
        const Compute eq_02 = rho * c1over18 * (c1 - c6 * u[2] + c4dot5 * u[2] * u[2] - usqr);
        const Compute eq_03 = rho * c1over36 * (c1 - c6 * ck_u03 + c4dot5 * ck_u03 * ck_u03 - usqr);
        const Compute eq_04 = rho * c1over36 * (c1 - c6 * ck_u04 + c4dot5 * ck_u04 * ck_u04 - usqr);
        const Compute eq_05 = rho * c1over36 * (c1 - c6 * ck_u05 + c4dot5 * ck_u05 * ck_u05 - usqr);
        const Compute eq_06 = rho * c1over36 * (c1 - c6 * ck_u06 + c4dot5 * ck_u06 * ck_u06 - usqr);
        const Compute eq_07 = rho * c1over36 * (c1 - c6 * ck_u07 + c4dot5 * ck_u07 * ck_u07 - usqr);
        const Compute eq_08 = rho * c1over36 * (c1 - c6 * ck_u08 + c4dot5 * ck_u08 * ck_u08 - usqr);

        const Compute eqopp_00 = eq_00 + rho * c1over18 * c6 * u[0];
        const Compute eqopp_01 = eq_01 + rho * c1over18 * c6 * u[1];
        const Compute eqopp_02 = eq_02 + rho * c1over18 * c6 * u[2];
        const Compute eqopp_03 = eq_03 + rho * c1over36 * c6 * ck_u03;
        const Compute eqopp_04 = eq_04 + rho * c1over36 * c6 * ck_u04;
        const Compute eqopp_05 = eq_05 + rho * c1over36 * c6 * ck_u05;
        const Compute eqopp_06 = eq_06 + rho * c1over36 * c6 * ck_u06;
        const Compute eqopp_07 = eq_07 + rho * c1over36 * c6 * ck_u07;
        const Compute eqopp_08 = eq_08 + rho * c1over36 * c6 * ck_u08;

        const Compute pop_out_00 = (c1 - omega) * static_cast<Compute>(pop[0]) + omega * eq_00;
        const Compute pop_out_01 = (c1 - omega) * static_cast<Compute>(pop[1]) + omega * eq_01;
        const Compute pop_out_02 = (c1 - omega) * static_cast<Compute>(pop[2]) + omega * eq_02;
        const Compute pop_out_03 = (c1 - omega) * static_cast<Compute>(pop[3]) + omega * eq_03;
        const Compute pop_out_04 = (c1 - omega) * static_cast<Compute>(pop[4]) + omega * eq_04;
        const Compute pop_out_05 = (c1 - omega) * static_cast<Compute>(pop[5]) + omega * eq_05;
        const Compute pop_out_06 = (c1 - omega) * static_cast<Compute>(pop[6]) + omega * eq_06;
        const Compute pop_out_07 = (c1 - omega) * static_cast<Compute>(pop[7]) + omega * eq_07;
        const Compute pop_out_08 = (c1 - omega) * static_cast<Compute>(pop[8]) + omega * eq_08;

        const Compute pop_out_opp_00 = (c1 - omega) * static_cast<Compute>(pop[10]) + omega * eqopp_00;
        const Compute pop_out_opp_01 = (c1 - omega) * static_cast<Compute>(pop[11]) + omega * eqopp_01;
        const Compute pop_out_opp_02 = (c1 - omega) * static_cast<Compute>(pop[12]) + omega * eqopp_02;
        const Compute pop_out_opp_03 = (c1 - omega) * static_cast<Compute>(pop[13]) + omega * eqopp_03;
        const Compute pop_out_opp_04 = (c1 - omega) * static_cast<Compute>(pop[14]) + omega * eqopp_04;
        const Compute pop_out_opp_05 = (c1 - omega) * static_cast<Compute>(pop[15]) + omega * eqopp_05;
        const Compute pop_out_opp_06 = (c1 - omega) * static_cast<Compute>(pop[16]) + omega * eqopp_06;
        const Compute pop_out_opp_07 = (c1 - omega) * static_cast<Compute>(pop[17]) + omega * eqopp_07;
        const Compute pop_out_opp_08 = (c1 - omega) * static_cast<Compute>(pop[18]) + omega * eqopp_08;


#define COMPUTE_GO_AND_BACK(GOid, BKid)                                                                          \
    {                                                                                                            \
        fOut(i, Lattice::Memory::template mapFromRegisters<GOid>()) = static_cast<Storage>(pop_out_0##GOid);     \
        fOut(i, Lattice::Memory::template mapFromRegisters<BKid>()) = static_cast<Storage>(pop_out_opp_0##GOid); \
    }

        COMPUTE_GO_AND_BACK(0, 10)
        COMPUTE_GO_AND_BACK(1, 11)
        COMPUTE_GO_AND_BACK(2, 12)
        COMPUTE_GO_AND_BACK(3, 13)
        COMPUTE_GO_AND_BACK(4, 14)
        COMPUTE_GO_AND_BACK(5, 15)
        COMPUTE_GO_AND_BACK(6, 16)
        COMPUTE_GO_AND_BACK(7, 17)
        COMPUTE_GO_AND_BACK(8, 18)

#undef COMPUTE_GO_AND_BACK

        {
            const Compute eq_09 = rho * (c1 / c3) * (c1 - usqr);
            const Compute pop_out_09 = (c1 - omega) * static_cast<Compute>(pop[Lattice::Registers::center]) +
                                       omega * eq_09;
            fOut(i, Lattice::Memory::center) = static_cast<Storage>(pop_out_09);
        }
    }
};

#undef CAST_TO_COMPUTE