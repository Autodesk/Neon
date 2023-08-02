#pragma once
#include "CellType.h"
#include "D3Q19.h"
#include "Neon/Neon.h"
#include "Neon/set/Containter.h"

namespace pull {
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
        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto fwMemIdx) {
            using M = typename Lattice::template MappersIdxSetWithFwdMem<fwMemIdx>;

            if constexpr (fwMemIdx == Lattice::Memory::center) {
                popIn[M::centerRegIdx] = fin(gidx, M::centerMemIdx);
            } else {
                if (CellType::isWall<M::bkMemIdx>()) {
                    popIn[M::fwdRegIdx] = fin(gidx, M::bkMemIdx) +
                                          fin.template getNghData<M::bkX, M::bkY, M::bkZ>(gidx, M::bkMemIdx)();
                } else {
                    popIn[M::fwdRegIdx] = fin.template getNghData<M::bkX, M::bkY, M::bkZ>(gidx, fwMemIdx)();
                }
            }
        });
    }
};

#undef CAST_TO_COMPUTE
}  // namespace pull

namespace push {
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
    pushStream(Idx const&                             gidx,
               const uint32_t&                        wallNghBitFlag,
               NEON_OUT Storage                       pOut[Lattice::Q],
               NEON_OUT typename PopField::Partition& fOut)
    {
        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
            using M = typename Lattice::template RegisterMapper<q>;

            if constexpr (M::fwdMemIdx == M::centerMemIdx) {
                fOut(gidx, M::fwdMemIdx) = pOut[M::fwdRegIdx];
            } else {
                if (CellType::isWall<M::fwdRegIdx>(wallNghBitFlag)) {
                    // fout(i, opp[k]) =
                    //      pop_out +
                    //      f(nb, k);
                    fOut(gidx, M::bkwMemIdx) =
                        pOut[M::fwdRegIdx] +
                        fOut.template getNghData<M::fwdX, M::fwdY, M::fwdZ>(gidx, M::fwdMemIdx)();
                } else {
                    // fout(nb,                                 k)         = pop_out;
                    fOut.template writeNghData<M::fwdX, M::fwdY, M::fwdZ>(gidx, M::fwdMemIdx, pOut[M::fwdRegIdx]);
                }
            }
        });
    }
};
}  // namespace push


namespace common {
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
    collideBgkUnrolled(Compute const&                rho /*!   Density            */,
                       std::array<Compute, 3> const& u /*!     Velocity           */,
                       Compute const&                usqr /*!  Usqr               */,
                       Compute const&                omega /*! Omega              */,
                       NEON_IO Storage               pop[Lattice::Q])

        -> void
    {

        // constexpr Compute c1over18 = 1. / 18.;
        constexpr Compute c1over36 = 1. / 36.;
        constexpr Compute c4dot5 = 4.5;
        constexpr Compute c3 = 3.;
        constexpr Compute c1 = 1.;
        constexpr Compute c6 = 6.;

        // constexpr int regCenter = Lattice::Registers::center;
        // constexpr int regFir = Lattice::Registers::center;

        Neon::ConstexprFor<0, Lattice::Registers::fwdRegIdxListLen, 1>(
            [&](auto fwdRegIdxListIdx) {
                using M = typename Lattice::template RegisterMapper<fwdRegIdxListIdx>;
                using T = typename Lattice::Registers;

                Compute eqFw;
                Compute eqBk;

                const Compute ck_u = T::template getCk_u<M::fwdRegIdx, Compute>(u);
                // double eq = rho * t[k] *
                //             (1. +
                //             3. * ck_u +
                //             4.5 * ck_u * ck_u -
                //             usqr);
                eqFw = rho * T::t[M::fwdRegIdx] *
                       (c1 +
                        c3 * ck_u +
                        c4dot5 * ck_u * ck_u -
                        usqr);

                // double eqopp = eq - 6.* rho * t[k] * ck_u;
                eqBk = eqFw -
                       c6 * rho * c1over36 * T::t[M::fwdRegIdx] * ck_u;

                // pop_out        = (1. - omega) * fin(i, k)                              + omega * eq;
                pop[M::fwdRegIdx] = (c1 - omega) * static_cast<Compute>(pop[M::fwdRegIdx]) + omega * eqFw;
                // pop_out_opp    = (1. - omega) * fin(i, opp[k])                         + omega * eqopp;
                pop[M::bkwRegIdx] = (c1 - omega) * static_cast<Compute>(pop[M::bkwRegIdx]) + omega * eqBk;
            });
        {  // Center;
            using T = typename Lattice::Registers;
            using M = typename Lattice::template RegisterMapper<Lattice::Registers::center>;
            //                  eq = rho * t[k]              * (1. - usqr);
            const Compute eqCenter = rho * T::t[M::fwdRegIdx] * (c1 - usqr);
            //                   fout(i, k) = (1. - omega) * fin(i, k)                              + omega * eq;
            pop[Lattice::Registers::center] = (c1 - omega) * static_cast<Compute>(pop[M::fwdRegIdx]) + omega * eqCenter;
        }
    }
    static inline NEON_CUDA_HOST_DEVICE auto
    localLoad(Idx const&                                  gidx,
              NEON_IN typename PopField::Partition const& fOut,
              Storage NEON_RESTRICT                       pOut[Lattice::Q])
    {
        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
            using M = typename Lattice::template RegisterMapper<q>;
            pOut[M::fwdRegIdx] = fOut(gidx, M::fwdMemIdx);
        });
    }

    static inline NEON_CUDA_HOST_DEVICE auto
    localStore(Idx const&                            gidx,
               Storage NEON_RESTRICT                 pOut[Lattice::Q],
               NEON_IN typename PopField::Partition& fOut)
    {
        Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
            using M = typename Lattice::template RegisterMapper<q>;
            fOut(gidx, M::fwdMemIdx) = pOut[M::fwdRegIdx];
        });
    }
};
}  // namespace common