#pragma once

#include "lattice.h"
#include "util.h"

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
Neon::set::Container collideKBG(Neon::domain::mGrid&                        grid,
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
                        for (int i = 0; i < Q; ++i) {
                            T cu = 0;
                            for (int d = 0; d < 3; ++d) {
                                cu += latticeVelocity[i][d] * vel.v[d];
                            }
                            cu *= 3.0;

                            //equilibrium
                            T feq = rho * latticeWeights[i] * (1. + cu + 0.5 * cu * cu - usqr);

                            //collide
                            out(cell, i) = ins[i] - omega * (ins[i] - feq);
                        }
                    }
                }
            };
        });
}