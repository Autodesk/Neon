#pragma once
#include "CellType.h"
#include "D3Q19.h"
#include "Neon/Neon.h"
#include "Neon/set/Containter.h"

template <typename Precision_, typename Grid_, typename Lattice_>
struct DeviceD3QXX
{
    using Lattice = Lattice_;
    using Precision = Precision_;
    using Compute = typename Precision::Compute;
    using Storage = typename Precision::Storage;
    using Grid = Grid_;

    using PopField = typename Grid::template Field<Storage, Lattice::Q>;
    using CellTypeField = typename Grid::template Field<CellType, 1>;

    using Idx = typename PopField::Idx;
    using Rho = typename Grid::template Field<Storage, 1>;
    using U = typename Grid::template Field<Storage, 3>;

    struct Pull
    {
        static inline NEON_CUDA_HOST_DEVICE auto
        pullStream(Idx const&                          gidx,
                   const uint32_t&                     wallBitFlag,
                   typename PopField::Partition const& fin,
                   NEON_OUT Storage                    popIn[Lattice::Q])
        {
            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                using M = typename Lattice::template RegisterMapper<q>;

                if constexpr (M::fwdMemQ == M::centerMemQ) {
                    popIn[M::centerRegQ] = fin(gidx, M::centerMemQ);
                } else {
                    if (CellType::isWall<M::bkMemIdx>()) {
                        popIn[M::fwdRegQ] = fin(gidx, M::bkMemIdx) +
                                            fin.template getNghData<M::bkwMemQX, M::bkwMemQY, M::bkwMemQZ>(gidx, M::bkwMemIdx)();
                    } else {
                        popIn[M::fwdRegQ] = fin.template getNghData<M::bkwMemQX, M::bkwMemQY, M::bkwMemQZ>(gidx, M::fwdMemIdx)();
                    }
                }
            });
        }
    };

    struct Push
    {
        static inline NEON_CUDA_HOST_DEVICE auto
        pushStream(Idx const&                             gidx,
                   const uint32_t&                        wallNghBitFlag,
                   NEON_OUT Storage                       pOut[Lattice::Q],
                   NEON_OUT typename PopField::Partition& fOut)
        {
            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                using M = typename Lattice::template RegisterMapper<q>;

                if constexpr (M::fwdMemQ == M::centerMemQ) {
                    fOut(gidx, M::centerMemQ) = pOut[M::centerRegQ];
                } else {
                    if (CellType::isWall<M::fwdRegQ>(wallNghBitFlag)) {
                        const auto pop_out = pOut[M::fwdRegQ];
                        const auto f_nb_k = fOut.template getNghData<M::fwdMemQX, M::fwdMemQY, M::fwdMemQZ>(gidx, M::fwdMemQ)();

                        // fout(i, opp[k]) =
                        fOut(gidx, M::bkwMemQ) =
                            // pop_out +
                            pop_out +
                            // f(nb, k);
                            f_nb_k;
                    } else {
                        // fout(nb,
                        fOut.template writeNghData<M::fwdMemQX, M::fwdMemQY, M::fwdMemQZ>(gidx,
                                                                                          // k)
                                                                                          M::fwdMemQ,
                                                                                          //    = pop_out;
                                                                                          pOut[M::fwdRegQ]);
                    }
                }
            });
        }
    };


    struct Common
    {
        static inline NEON_CUDA_HOST_DEVICE auto
        macroscopic(const Storage     pop[Lattice::Q],
                    NEON_OUT Compute& rho,
                    NEON_OUT std::array<Compute, 3>& u)
            -> void
        {
            if constexpr (Lattice::Q == 19) {
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
                return;
            }
            if constexpr (Lattice::Q == 27) {
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
                u[1] = (Y_P1 - Y_M1) / rho;
                u[2] = (Z_P1 - Z_M1) / rho;
                return;
            }
            printf("Error: macroscopic function does not support the selected lattice.\n");
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
            constexpr Compute c4dot5 = 4.5;
            constexpr Compute c3 = 3.;
            constexpr Compute c1 = 1.;
            constexpr Compute c6 = 6.;

            // constexpr int regCenter = Lattice::Registers::center;
            // constexpr int regFir = Lattice::Registers::center;

            Neon::ConstexprFor<0, Lattice::Registers::firstHalfQLen, 1>(
                [&](auto q) {
                    using M = typename Lattice::template RegisterMapper<q>;
                    using T = typename Lattice::Registers;

                    Compute eqFw;
                    Compute eqBk;

                    const Compute ck_u = u[0] * Lattice::Registers::template getComponentOfDirection<q, 0>() +
                                         u[1] * Lattice::Registers::template getComponentOfDirection<q, 1>() +
                                         u[2] * Lattice::Registers::template getComponentOfDirection<q, 2>();

                    // double eq = rho * t[k] *
                    //             (1. +
                    //             3. * ck_u +
                    //             4.5 * ck_u * ck_u -
                    //             usqr);
                    eqFw = rho * T::t[M::fwdRegQ] *
                           (c1 +
                            c3 * ck_u +
                            c4dot5 * ck_u * ck_u -
                            usqr);

                    // double eqopp = eq - 6.* rho * t[k] * ck_u;
                    eqBk = eqFw -
                           c6 * rho * T::t[M::fwdRegQ] * ck_u;

                    // pop_out      = (1. - omega) * fin(i, k)                             + omega * eq;
                    pop[M::fwdRegQ] = (c1 - omega) * static_cast<Compute>(pop[M::fwdRegQ]) + omega * eqFw;
                    // pop_out_opp  = (1. - omega) * fin(i, opp[k])                        + omega * eqopp;
                    pop[M::bkwRegQ] = (c1 - omega) * static_cast<Compute>(pop[M::bkwRegQ]) + omega * eqBk;
                });
            {  // Center;
                using T = typename Lattice::Registers;
                using M = typename Lattice::template RegisterMapper<Lattice::Registers::center>;
                //                  eq = rho * t[k]                * (1. - usqr);
                const Compute eqCenter = rho * T::t[M::centerRegQ] * (c1 - usqr);
                //      fout(i, k) = (1. - omega) * fin(i, k)                                + omega * eq;
                pop[M::centerRegQ] = (c1 - omega) * static_cast<Compute>(pop[M::centerRegQ]) + omega * eqCenter;
            }
        }

        static inline NEON_CUDA_HOST_DEVICE auto
        localLoad(Idx const&                                  gidx,
                  NEON_IN typename PopField::Partition const& fOut,
                  Storage NEON_RESTRICT                       pOut[Lattice::Q])
        {
            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                using M = typename Lattice::template RegisterMapper<q>;
                pOut[M::fwdRegQ] = fOut(gidx, M::fwdMemQ);
            });
        }

        static inline NEON_CUDA_HOST_DEVICE auto
        localStore(Idx const&                            gidx,
                   Storage NEON_RESTRICT                 pOut[Lattice::Q],
                   NEON_IN typename PopField::Partition& fOut)
        {
            Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                using M = typename Lattice::template RegisterMapper<q>;
                fOut(gidx, M::fwdMemQ) = pOut[M::fwdRegQ];
            });
        }

        static inline NEON_CUDA_HOST_DEVICE auto
        collideKBCUnrolled(Compute const&                rho /*!   Density            */,
                           std::array<Compute, 3> const& u /*!     Velocity           */,
                           Compute const&                usqr /*!  Usqr               */,
                           Compute const&                omega /*! Omega              */,
                           Compute const&                invBeta /*! invBeta              */,
                           NEON_IO Storage               pop[Lattice::Q])

            -> void
        {
            if constexpr (Lattice::Q == 27) {
                constexpr Compute tiny = Compute(1e-7);

                Compute       Pi[6] = {0, 0, 0, 0, 0, 0};
                Compute       e0 = 0;
                Compute       e1 = 0;
                Compute       deltaS[Lattice::Q];
                Compute       fneq[Lattice::Q];
                Compute       feq[Lattice::Q];
                const Compute beta = omega * 0.5;

                auto fdecompose_shear = [&](const int q) -> Compute {
                    const Compute Nxz = Pi[0] - Pi[5];
                    const Compute Nyz = Pi[3] - Pi[5];
                    if (q == 9) {
                        return (2.0 * Nxz - Nyz) / 6.0;
                    } else if (q == 18) {
                        return (2.0 * Nxz - Nyz) / 6.0;
                    } else if (q == 3) {
                        return (-Nxz + 2.0 * Nyz) / 6.0;
                    } else if (q == 6) {
                        return (-Nxz + 2.0 * Nyz) / 6.0;
                    } else if (q == 1) {
                        return (-Nxz - Nyz) / 6.0;
                    } else if (q == 2) {
                        return (-Nxz - Nyz) / 6.0;
                    } else if (q == 12 || q == 24) {
                        return Pi[1] / 4.0;
                    } else if (q == 21 || q == 15) {
                        return -Pi[1] / 4.0;
                    } else if (q == 10 || q == 20) {
                        return Pi[2] / 4.0;
                    } else if (q == 19 || q == 11) {
                        return -Pi[2] / 4.0;
                    } else if (q == 8 || q == 4) {
                        return Pi[4] / 4.0;
                    } else if (q == 7 || q == 5) {
                        return -Pi[4] / 4.0;
                    } else {
                        return Compute(0);
                    }
                };

                // equilibrium
                Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                    const Compute cu = Compute(3) *
                                       (u[0] * Lattice::Registers::template getComponentOfDirection<q, 0>() +
                                        u[1] * Lattice::Registers::template getComponentOfDirection<q, 1>() +
                                        u[2] * Lattice::Registers::template getComponentOfDirection<q, 2>());

                    feq[q] = rho * Lattice::Registers::template getWeightOfDirection<q, 0>() * (1. + cu + 0.5 * cu * cu - usqr);

                    fneq[q] = pop[q] - feq[q];
                });

                // momentum_flux
                Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                    Neon::ConstexprFor<0, 6, 1>([&](auto i) {
                        Pi[i] += fneq[q] * Lattice::Registers::template getMomentByDirection<q, i>();
                    });
                });

                // fdecompose_shear
                Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                    deltaS[q] = rho * fdecompose_shear(q);

                    Compute deltaH = fneq[q] - deltaS[q];

                    e0 += (deltaS[q] * deltaH / feq[q]);
                    e1 += (deltaH * deltaH / feq[q]);
                });

                // gamma
                Compute gamma = invBeta - (2.0 - invBeta) * e0 / (tiny + e1);


                // fout
                Neon::ConstexprFor<0, Lattice::Q, 1>([&](auto q) {
                    Compute deltaH = fneq[q] - deltaS[q];
                    pop[q] = pop[q] - beta * (2.0 * deltaS[q] + gamma * deltaH);
                });
            }
        }
    };
};