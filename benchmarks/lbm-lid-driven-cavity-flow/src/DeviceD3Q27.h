#pragma once
#include "CellType.h"
#include "D3Q27.h"
#include "Neon/Neon.h"
#include "Neon/set/Containter.h"


template <typename Precision_, typename Grid_>
struct DeviceD3Q27
{
    using Lattice = D3Q27<Precision_>;
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
        const Compute X_M1 = POP(0) + POP(3) + POP(4) + POP(5) + POP(6) + POP(9) + POP(10) + POP(11) + POP(12);
        const Compute X_P1 = POP(14) + POP(17) + POP(18) + POP(19) + POP(20) + POP(23) + POP(24) + POP(25) + POP(26);
        const Compute X_0 = POP(1) + POP(2) + POP(7) + POP(8) + POP(13) + POP(15) + POP(16) + POP(21) + POP(22);

        const Compute Y_M1 = POP(1) + POP(3) + POP(7) + POP(8) + POP(9) + POP(10) + POP(18) + POP(25) + POP(26);
        const Compute Y_P1 = POP(15) + POP(17) + POP(21) + POP(22) + POP(23) + POP(24) + POP(4) + POP(11) + POP(12);

        const Compute Z_M1 = POP(2) + POP(5) + POP(7) + POP(9) + POP(11) + POP(20) + POP(22) + POP(24) + POP(26);
        const Compute Z_P1 = POP(16) + POP(19) + POP(21) + POP(23) + POP(25) + POP(6) + POP(8) + POP(10) + POP(12);
#undef POP

        rho = X_M1 + X_P1 + X_0;
        u[0] = (X_P1 - X_M1) / rho;
        u[0] = (Y_P1 - Y_M1) / rho;
        u[0] = (Z_P1 - Z_M1) / rho;
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
        const Compute cku1 = u[0] + u[1];
        const Compute cku2 = -u[0] + u[1];
        const Compute cku3 = u[0] + u[2];
        const Compute cku4 = -u[0] + u[2];
        const Compute cku5 = u[1] + u[2];
        const Compute cku6 = -u[1] + u[2];
        const Compute cku7 = u[0] + u[1] + u[2];
        const Compute cku8 = -u[0] + u[1] + u[2];
        const Compute cku9 = u[0] - u[1] + u[2];
        const Compute cku0 = u[0] + u[1] - u[2];

        std::array<Compute, Lattice::Q> feqRM;

        constexpr int F000 = 13;
        constexpr int FM00 = 0;
        constexpr int F0M0 = 1;
        constexpr int F00M = 2;
        constexpr int FMM0 = 3;
        constexpr int FMP0 = 4;
        constexpr int FM0M = 5;
        constexpr int FM0P = 6;
        constexpr int F0MM = 7;
        constexpr int F0MP = 8;
        constexpr int FMMM = 9;
        constexpr int FMMP = 10;
        constexpr int FMPM = 11;
        constexpr int FMPP = 12;
        constexpr int FP00 = 14;
        constexpr int F0P0 = 15;
        constexpr int F00P = 16;
        constexpr int FPP0 = 17;
        constexpr int FPM0 = 18;
        constexpr int FP0P = 19;
        constexpr int FP0M = 20;
        constexpr int F0PP = 21;
        constexpr int F0PM = 22;
        constexpr int FPPP = 23;
        constexpr int FPPM = 24;
        constexpr int FPMP = 25;
        constexpr int FPMM = 26;

        constexpr Compute c1over18 = 1. / 18.;
        constexpr Compute c1over36 = 1. / 36.;
        constexpr Compute c4dot5 = 4.5;
        constexpr Compute c3 = 3.;
        constexpr Compute c1 = 1.;
        constexpr Compute c6 = 6.;

        feqRM[F000] = rho * Lattice::Registers::t[F000] * (c1- usqr);

        feqRM[FM00] = rho * Lattice::Registers::t[FM00] * (c1- c3* u[0] + c4dot5* u[0] * u[0] - usqr);
        feqRM[FP00] = rho * Lattice::Registers::t[FP00] * (c6 * u[0]) + feqRM[FM00];

        feqRM[F0M0] = rho * Lattice::Registers::t[F0M0] * (c1- c3* u[1] + c4dot5* u[1] * u[1] - usqr);
        feqRM[F0P0] = rho * Lattice::Registers::t[F0P0] * (c6 * u[1]) + feqRM[F0M0];

        feqRM[F00M] = rho * Lattice::Registers::t[F00M] * (c1- c3* u[2] + c4dot5* u[2] * u[2] - usqr);
        feqRM[F00P] = rho * Lattice::Registers::t[F00P] * (c6 * u[2]) + feqRM[F00M];

        feqRM[FMM0] = rho * Lattice::Registers::t[FMM0] * (c1- c3* cku1 + c4dot5* cku1 * cku1 - usqr);
        feqRM[FPP0] = rho * Lattice::Registers::t[FPP0] * (c6 * cku1) + feqRM[FMM0];
        feqRM[FPM0] = rho * Lattice::Registers::t[FPM0] * (c1- c3* cku2 + c4dot5* cku2 * cku2 - usqr);
        feqRM[FMP0] = rho * Lattice::Registers::t[FMP0] * (c6 * cku2) + feqRM[FPM0];

        feqRM[FM0M] = rho * Lattice::Registers::t[FM0M] * (c1- c3* cku3 + c4dot5* cku3 * cku3 - usqr);
        feqRM[FP0P] = rho * Lattice::Registers::t[FP0P] * (c6 * cku3) + feqRM[FM0M];
        feqRM[FP0M] = rho * Lattice::Registers::t[FP0M] * (c1- c3* cku4 + c4dot5* cku4 * cku4 - usqr);
        feqRM[FM0P] = rho * Lattice::Registers::t[FM0P] * (c6 * cku4) + feqRM[FP0M];

        feqRM[F0MM] = rho * Lattice::Registers::t[F0MM] * (c1- c3* cku5 + c4dot5* cku5 * cku5 - usqr);
        feqRM[F0PP] = rho * Lattice::Registers::t[F0PP] * (c6 * cku5) + feqRM[F0MM];
        feqRM[F0PM] = rho * Lattice::Registers::t[F0PM] * (c1- c3* cku6 + c4dot5* cku6 * cku6 - usqr);
        feqRM[F0MP] = rho * Lattice::Registers::t[F0MP] * (c6 * cku6) + feqRM[F0PM];

        feqRM[FMMM] = rho * Lattice::Registers::t[FMMM] * (c1- c3* cku7 + c4dot5* cku7 * cku7 - usqr);
        feqRM[FPPP] = rho * Lattice::Registers::t[FPPP] * (c6 * cku7) + feqRM[FMMM];
        feqRM[FPMM] = rho * Lattice::Registers::t[FPMM] * (c1- c3* cku8 + c4dot5* cku8 * cku8 - usqr);
        feqRM[FMPP] = rho * Lattice::Registers::t[FMPP] * (c6 * cku8) + feqRM[FPMM];
        feqRM[FMPM] = rho * Lattice::Registers::t[FMPM] * (c1- c3* cku9 + c4dot5* cku9 * cku9 - usqr);
        feqRM[FPMP] = rho * Lattice::Registers::t[FPMP] * (c6 * cku9) + feqRM[FMPM];
        feqRM[FMMP] = rho * Lattice::Registers::t[FMMP] * (c1- c3* cku0 + c4dot5* cku0 * cku0 - usqr);
        feqRM[FPPM] = rho * Lattice::Registers::t[FPPM] * (c6 * cku0) + feqRM[FMMP];

        // BGK Collision based on the second-order equilibrium
        std::array<Compute, Lattice::Q>  foutRM;

        foutRM[F000] = (c1- omega) * static_cast<Compute>(pop[F000]) + omega * feqRM[F000];

        foutRM[FP00] = (c1- omega) * static_cast<Compute>(pop[FP00]) + omega * feqRM[FP00];
        foutRM[FM00] = (c1- omega) * static_cast<Compute>(pop[FM00]) + omega * feqRM[FM00];

        foutRM[F0P0] = (c1- omega) * static_cast<Compute>(pop[F0P0]) + omega * feqRM[F0P0];
        foutRM[F0M0] = (c1- omega) * static_cast<Compute>(pop[F0M0]) + omega * feqRM[F0M0];

        foutRM[F00P] = (c1- omega) * static_cast<Compute>(pop[F00P]) + omega * feqRM[F00P];
        foutRM[F00M] = (c1- omega) * static_cast<Compute>(pop[F00M]) + omega * feqRM[F00M];

        foutRM[FPP0] = (c1- omega) * static_cast<Compute>(pop[FPP0]) + omega * feqRM[FPP0];
        foutRM[FMP0] = (c1- omega) * static_cast<Compute>(pop[FMP0]) + omega * feqRM[FMP0];
        foutRM[FPM0] = (c1- omega) * static_cast<Compute>(pop[FPM0]) + omega * feqRM[FPM0];
        foutRM[FMM0] = (c1- omega) * static_cast<Compute>(pop[FMM0]) + omega * feqRM[FMM0];

        foutRM[FP0P] = (c1- omega) * static_cast<Compute>(pop[FP0P]) + omega * feqRM[FP0P];
        foutRM[FM0P] = (c1- omega) * static_cast<Compute>(pop[FM0P]) + omega * feqRM[FM0P];
        foutRM[FP0M] = (c1- omega) * static_cast<Compute>(pop[FP0M]) + omega * feqRM[FP0M];
        foutRM[FM0M] = (c1- omega) * static_cast<Compute>(pop[FM0M]) + omega * feqRM[FM0M];

        foutRM[F0PP] = (c1- omega) * static_cast<Compute>(pop[F0PP]) + omega * feqRM[F0PP];
        foutRM[F0MP] = (c1- omega) * static_cast<Compute>(pop[F0MP]) + omega * feqRM[F0MP];
        foutRM[F0PM] = (c1- omega) * static_cast<Compute>(pop[F0PM]) + omega * feqRM[F0PM];
        foutRM[F0MM] = (c1- omega) * static_cast<Compute>(pop[F0MM]) + omega * feqRM[F0MM];

        foutRM[FPPP] = (c1- omega) * static_cast<Compute>(pop[FPPP]) + omega * feqRM[FPPP];
        foutRM[FMPP] = (c1- omega) * static_cast<Compute>(pop[FMPP]) + omega * feqRM[FMPP];
        foutRM[FPMP] = (c1- omega) * static_cast<Compute>(pop[FPMP]) + omega * feqRM[FPMP];
        foutRM[FPPM] = (c1- omega) * static_cast<Compute>(pop[FPPM]) + omega * feqRM[FPPM];
        foutRM[FMMP] = (c1- omega) * static_cast<Compute>(pop[FMMP]) + omega * feqRM[FMMP];
        foutRM[FMPM] = (c1- omega) * static_cast<Compute>(pop[FMPM]) + omega * feqRM[FMPM];
        foutRM[FPMM] = (c1- omega) * static_cast<Compute>(pop[FPMM]) + omega * feqRM[FPMM];
        foutRM[FMMM] = (c1- omega) * static_cast<Compute>(pop[FMMM]) + omega * feqRM[FMMM];

        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto GOMemoryId) {
            fOut(i, GOMemoryId) = static_cast<Storage>(foutRM[Lattice::Memory::template mapToRegisters<GOMemoryId>()]);
        });
    }
};

