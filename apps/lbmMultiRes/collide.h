#pragma once

#include "lattice.h"
#include "util.h"


template <typename T, int Q>
inline Neon::set::Container collideBGKUnrolled(Neon::domain::mGrid&                        grid,
                                               T                                           omega0,
                                               int                                         level,
                                               int                                         numLevels,
                                               const Neon::domain::mGrid::Field<CellType>& cellType,
                                               const Neon::domain::mGrid::Field<T>&        fin,
                                               Neon::domain::mGrid::Field<T>&              fout)
{
    return grid.getContainer(
        "collideKBC_" + std::to_string(level), level,
        [&, level, omega0, numLevels](Neon::set::Loader& loader) {
            const auto& type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
            const auto& in = fin.load(loader, level, Neon::MultiResCompute::MAP);
            auto        out = fout.load(loader, level, Neon::MultiResCompute::MAP);
            const T     omega = computeOmega(omega0, level, numLevels);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                if (type(cell, 0) == CellType::bulk) {

                    if (!in.hasChildren(cell)) {
                        //fin
                        T ins[Q];
                        for (int i = 0; i < Q; ++i) {
                            ins[i] = in(cell, i);
                        }


                        const T X_M1 = ins[0] + ins[3] + ins[4] + ins[5] + ins[6];
                        const T X_P1 = ins[10] + ins[13] + ins[14] + ins[15] + ins[16];
                        const T X_0 = ins[9] + ins[1] + ins[2] + ins[7] + ins[8] + ins[11] + ins[12] + ins[17] + ins[18];
                        const T Y_M1 = ins[1] + ins[3] + ins[7] + ins[8] + ins[14];
                        const T Y_P1 = ins[4] + ins[11] + ins[13] + ins[17] + ins[18];
                        const T Z_M1 = ins[2] + ins[5] + ins[7] + ins[16] + ins[18];
                        const T Z_P1 = ins[6] + ins[8] + ins[12] + ins[15] + ins[17];

                        //density
                        const T rho = X_M1 + X_P1 + X_0;

                        //velocity
                        Neon::Vec_3d<T> vel;

                        vel.v[0] = (X_P1 - X_M1) / rho;
                        vel.v[1] = (Y_P1 - Y_M1) / rho;
                        vel.v[2] = (Z_P1 - Z_M1) / rho;


                        const T usqr = (1.5) * (vel.v[0] * vel.v[0] + vel.v[1] * vel.v[1] + vel.v[2] * vel.v[2]);

                        //collide
                        const T ck_u03 = vel.v[0] + vel.v[1];
                        const T ck_u04 = vel.v[0] - vel.v[1];
                        const T ck_u05 = vel.v[0] + vel.v[2];
                        const T ck_u06 = vel.v[0] - vel.v[2];
                        const T ck_u07 = vel.v[1] + vel.v[2];
                        const T ck_u08 = vel.v[1] - vel.v[2];

                        const T eq_00 = rho * (1. / 18.) * (1. - 3. * vel.v[0] + 4.5 * vel.v[0] * vel.v[0] - usqr);
                        const T eq_01 = rho * (1. / 18.) * (1. - 3. * vel.v[1] + 4.5 * vel.v[1] * vel.v[1] - usqr);
                        const T eq_02 = rho * (1. / 18.) * (1. - 3. * vel.v[2] + 4.5 * vel.v[2] * vel.v[2] - usqr);
                        const T eq_03 = rho * (1. / 36.) * (1. - 3. * ck_u03 + 4.5 * ck_u03 * ck_u03 - usqr);
                        const T eq_04 = rho * (1. / 36.) * (1. - 3. * ck_u04 + 4.5 * ck_u04 * ck_u04 - usqr);
                        const T eq_05 = rho * (1. / 36.) * (1. - 3. * ck_u05 + 4.5 * ck_u05 * ck_u05 - usqr);
                        const T eq_06 = rho * (1. / 36.) * (1. - 3. * ck_u06 + 4.5 * ck_u06 * ck_u06 - usqr);
                        const T eq_07 = rho * (1. / 36.) * (1. - 3. * ck_u07 + 4.5 * ck_u07 * ck_u07 - usqr);
                        const T eq_08 = rho * (1. / 36.) * (1. - 3. * ck_u08 + 4.5 * ck_u08 * ck_u08 - usqr);

                        const T eqopp_00 = eq_00 + rho * (1. / 18.) * 6. * vel.v[0];
                        const T eqopp_01 = eq_01 + rho * (1. / 18.) * 6. * vel.v[1];
                        const T eqopp_02 = eq_02 + rho * (1. / 18.) * 6. * vel.v[2];
                        const T eqopp_03 = eq_03 + rho * (1. / 36.) * 6. * ck_u03;
                        const T eqopp_04 = eq_04 + rho * (1. / 36.) * 6. * ck_u04;
                        const T eqopp_05 = eq_05 + rho * (1. / 36.) * 6. * ck_u05;
                        const T eqopp_06 = eq_06 + rho * (1. / 36.) * 6. * ck_u06;
                        const T eqopp_07 = eq_07 + rho * (1. / 36.) * 6. * ck_u07;
                        const T eqopp_08 = eq_08 + rho * (1. / 36.) * 6. * ck_u08;

                        const T pop_out_00 = (1. - omega) * ins[0] + omega * eq_00;
                        const T pop_out_01 = (1. - omega) * ins[1] + omega * eq_01;
                        const T pop_out_02 = (1. - omega) * ins[2] + omega * eq_02;
                        const T pop_out_03 = (1. - omega) * ins[3] + omega * eq_03;
                        const T pop_out_04 = (1. - omega) * ins[4] + omega * eq_04;
                        const T pop_out_05 = (1. - omega) * ins[5] + omega * eq_05;
                        const T pop_out_06 = (1. - omega) * ins[6] + omega * eq_06;
                        const T pop_out_07 = (1. - omega) * ins[7] + omega * eq_07;
                        const T pop_out_08 = (1. - omega) * ins[8] + omega * eq_08;

                        const T pop_out_opp_00 = (1. - omega) * ins[10] + omega * eqopp_00;
                        const T pop_out_opp_01 = (1. - omega) * ins[11] + omega * eqopp_01;
                        const T pop_out_opp_02 = (1. - omega) * ins[12] + omega * eqopp_02;
                        const T pop_out_opp_03 = (1. - omega) * ins[13] + omega * eqopp_03;
                        const T pop_out_opp_04 = (1. - omega) * ins[14] + omega * eqopp_04;
                        const T pop_out_opp_05 = (1. - omega) * ins[15] + omega * eqopp_05;
                        const T pop_out_opp_06 = (1. - omega) * ins[16] + omega * eqopp_06;
                        const T pop_out_opp_07 = (1. - omega) * ins[17] + omega * eqopp_07;
                        const T pop_out_opp_08 = (1. - omega) * ins[18] + omega * eqopp_08;


#define COMPUTE_GO_AND_BACK(GOid, BKid) \
    out(cell, GOid) = pop_out_0##GOid;  \
    out(cell, BKid) = pop_out_opp_0##GOid;

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


                        const T eq_09 = rho * (1. / 3.) * (1. - usqr);
                        const T pop_out_09 = (1. - omega) * ins[9] + omega * eq_09;
                        out(cell, 9) = pop_out_09;


                        /*{
                            constexpr T tiny = 0.0000000000000001;

                            //////
                            T rho_gt = 0;
                            for (int i = 0; i < Q; ++i) {
                                rho_gt += ins[i];
                            }
                            if (std::abs(rho_gt - rho) > tiny) {
                                printf("\n diff rho_gt= %.17g, rho= %.17g",
                                       rho_gt, rho);
#ifdef __CUDA_ARCH__
                                __trap();
#endif
                            }

                            //////
                            const Neon::Vec_3d<T> vel_gt = velocity<T, Q>(ins, rho_gt);
                            if (std::abs(vel_gt.x - vel.x) > tiny ||
                                std::abs(vel_gt.y - vel.y) > tiny ||
                                std::abs(vel_gt.z - vel.z) > tiny) {
                                printf("\n diff vel_gt = %.17g, %.17g, %.17g, vel= %.17g, %.17g, %.17g",
                                       vel_gt.x,
                                       vel_gt.y,
                                       vel_gt.z,
                                       vel.x,
                                       vel.y,
                                       vel.z);
#ifdef __CUDA_ARCH__
                                __trap();
#endif
                            }

                            /////

                            const T usqr_gt = (3.0 / 2.0) * (vel_gt.x * vel_gt.x + vel_gt.y * vel_gt.y + vel_gt.z * vel_gt.z);
                            for (int q = 0; q < Q; ++q) {
                                T cu = 0;
                                for (int d = 0; d < 3; ++d) {
                                    cu += latticeVelocity[q][d] * vel_gt.v[d];
                                }
                                cu *= 3.0;

                                //equilibrium
                                T feq = rho_gt * latticeWeights[q] * (1. + cu + 0.5 * cu * cu - usqr);

                                //collide
                                T out_gt = (1 - omega) * ins[q] + omega * feq;
                                if (std::abs(out_gt - out(cell, q)) > tiny) {
                                    printf("\n diff q= %d gt= %.17g, pop= %.17g",
                                           q, out_gt, out(cell, q));
#ifdef __CUDA_ARCH__
                                    __trap();
#endif
                                }
                            }
                            //////
                        }*/
                    }
                }
            };
        });
}

template <typename T, int Q>
inline Neon::set::Container collideKBC(Neon::domain::mGrid&                        grid,
                                       T                                           omega0,
                                       int                                         level,
                                       int                                         numLevels,
                                       const Neon::domain::mGrid::Field<CellType>& cellType,
                                       const Neon::domain::mGrid::Field<T>&        fin,
                                       Neon::domain::mGrid::Field<T>&              fout)
{
    return grid.getContainer(
        "collideKBC_" + std::to_string(level), level,
        [&, level, omega0, numLevels](Neon::set::Loader& loader) {
            const auto& type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
            const auto& in = fin.load(loader, level, Neon::MultiResCompute::MAP);
            auto        out = fout.load(loader, level, Neon::MultiResCompute::MAP);
            const T     omega = computeOmega(omega0, level, numLevels);
            const T     beta = omega * 0.5;
            const T     invBeta = 1.0 / beta;

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                if (type(cell, 0) == CellType::bulk) {

                    constexpr T tiny = 1e-7;

                    if (!in.hasChildren(cell)) {

                        //fin
                        T ins[Q];
                        for (int i = 0; i < Q; ++i) {
                            ins[i] = in(cell, i);
                        }

                        //density
                        T rho = 0;
                        for (int i = 0; i < Q; ++i) {
                            rho += ins[i];
                        }

                        //velocity
                        const Neon::Vec_3d<T> vel = velocity<T, Q>(ins, rho);

                        T Pi[6] = {0, 0, 0, 0, 0, 0};
                        T e0 = 0;
                        T e1 = 0;
                        T deltaS[Q];
                        T fneq[Q];
                        T feq[Q];


                        auto fdecompose_shear = [&](const int q) {
                            const T Nxz = Pi[0] - Pi[5];
                            const T Nyz = Pi[3] - Pi[5];
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
                                return T(0);
                            }
                        };


                        //equilibrium
                        const T usqr = (3.0 / 2.0) * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
                        for (int q = 0; q < Q; ++q) {
                            T cu = 0;
                            for (int d = 0; d < 3; ++d) {
                                cu += latticeVelocity[q][d] * vel.v[d];
                            }
                            cu *= 3.0;

                            feq[q] = rho * latticeWeights[q] * (1. + cu + 0.5 * cu * cu - usqr);

                            fneq[q] = ins[q] - feq[q];
                        }

                        //momentum_flux
                        for (int q = 0; q < Q; ++q) {
                            for (int i = 0; i < 6; ++i) {
                                Pi[i] += fneq[q] * latticeMoment[q][i];
                            }
                        }


                        //fdecompose_shear
                        for (int q = 0; q < Q; ++q) {
                            deltaS[q] = rho * fdecompose_shear(q);

                            T deltaH = fneq[q] - deltaS[q];

                            e0 += (deltaS[q] * deltaH / feq[q]);
                            e1 += (deltaH * deltaH / feq[q]);
                        }

                        //gamma
                        T gamma = invBeta - (2.0 - invBeta) * e0 / (tiny + e1);


                        //fout
                        for (int q = 0; q < Q; ++q) {
                            T deltaH = fneq[q] - deltaS[q];
                            out(cell, q) = ins[q] - beta * (2.0 * deltaS[q] + gamma * deltaH);
                        }
                    }
                }
            };
        });
}

template <typename T, int Q>
Neon::set::Container collideBGK(Neon::domain::mGrid&                        grid,
                                T                                           omega0,
                                int                                         level,
                                int                                         numLevels,
                                const Neon::domain::mGrid::Field<CellType>& cellType,
                                const Neon::domain::mGrid::Field<T>&        fin,
                                Neon::domain::mGrid::Field<T>&              fout)
{
    return grid.getContainer(
        "collideKBG_" + std::to_string(level), level,
        [&, level, omega0, numLevels](Neon::set::Loader& loader) {
            const auto& type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
            const auto& in = fin.load(loader, level, Neon::MultiResCompute::MAP);
            auto        out = fout.load(loader, level, Neon::MultiResCompute::MAP);
            const T     omega = computeOmega(omega0, level, numLevels);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                if (type(cell, 0) == CellType::bulk) {

                    if (!in.hasChildren(cell)) {


                        //fin
                        T ins[Q];
                        for (int i = 0; i < Q; ++i) {
                            ins[i] = in(cell, i);
                        }

                        //density
                        T rho = 0;
                        for (int i = 0; i < Q; ++i) {
                            rho += ins[i];
                        }

                        //velocity
                        const Neon::Vec_3d<T> vel = velocity<T, Q>(ins, rho);


                        const T usqr = (3.0 / 2.0) * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
                        for (int q = 0; q < Q; ++q) {
                            T cu = 0;
                            for (int d = 0; d < 3; ++d) {
                                cu += latticeVelocity[q][d] * vel.v[d];
                            }
                            cu *= 3.0;

                            //equilibrium
                            T feq = rho * latticeWeights[q] * (1. + cu + 0.5 * cu * cu - usqr);

                            //collide
                            //out(cell, q) = ins[q] - omega * (ins[q] - feq);
                            out(cell, q) = (1 - omega) * ins[q] + omega * feq;
                        }
                    }
                }
            };
        });
}