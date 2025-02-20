#pragma once
#include "collide.h"

template <typename T, typename FieldT, typename FieldType>
inline NEON_CUDA_HOST_DEVICE void stream(const typename Neon::domain::mGrid::Idx& cell,
                                         FieldT&                                  out,
                                         const FieldT&                            explosionIn,
                                         const FieldType&                         type,
                                         const int8_t                             q,
                                         const T                                  cellVal)
{
    //since we are on the finest level, we only need to do streaming and explosion (no coalescence)
    //streaming is done as push

    const Neon::int8_3d dir = getDir(q);

    //if the neighbor cell has children, then this 'cell' is interfacing with L-1 (fine) along q direction
    const auto nghCell = out.helpGetNghIdx(cell, dir);
    if (!out.hasChildren(nghCell)) {
        if (out.isActive(nghCell)) {
            auto nghType = type(nghCell);
            if (nghType == CellType::bulk) {
                out(nghCell, q) = cellVal;
            } else {
                const int8_t opposte_q = latticeOppositeID[q];
                out(cell, opposte_q) = cellVal + out(nghCell, q);
            }
        } else if (!(dir.x == 0 && dir.y == 0 && dir.z == 0)) {
            //only if we are not on the coarsest level and
            //only if we can not do normal streaming, then we may have a coarser neighbor from which
            //we can read this pop

            //get the uncle direction/offset i.e., the neighbor of the cell's parent
            //this direction/offset is wrt to the cell's parent
            Neon::int8_3d uncleDir = uncleOffset(cell.mInDataBlockIdx, dir);

            const int8_t opposte_q = latticeOppositeID[q];
            const auto   uncle = explosionIn.uncleVal(cell, uncleDir, opposte_q, T(0));
            if (uncle.mIsValid) {
                out(cell, opposte_q) = uncle.mData;
            }
        }
    }
}


template <typename T, int Q>
inline Neon::set::Container collideBGKUnrolledFusedAll(Neon::domain::mGrid&                        grid,
                                                       T                                           omega0,
                                                       int                                         level,
                                                       int                                         numLevels,
                                                       const Neon::domain::mGrid::Field<CellType>& cellType,
                                                       const Neon::domain::mGrid::Field<T>&        fin,
                                                       Neon::domain::mGrid::Field<T>&              fout,
                                                       bool                                        storeOut)  //store in fout
{
    if (level != 0) {
        Neon::NeonException exp("collideBGKUnrolledFusedAll");
        exp << "Only works with on the finest level. Input level =" << level;
        NEON_THROW(exp);
    }

    return grid.newContainer(
        "CHSOE" + std::to_string(level), level,
        [&, level, omega0, numLevels, storeOut](Neon::set::Loader& loader) {
            const auto type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
            const auto in = fin.load(loader, level, Neon::MultiResCompute::MAP);
            auto       out = fout.load(loader, level, Neon::MultiResCompute::MAP);
            const T    omega = computeOmega(omega0, level, numLevels);

            if (numLevels > 1) {
                //load the next level as a map to indicate that we will (remote) write to it
                fout.load(loader, level + 1, Neon::MultiResCompute::MAP);
            }

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                if (type(cell, 0) == CellType::bulk) {

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


                    const T usqr = T(1.5) * (vel.v[0] * vel.v[0] + vel.v[1] * vel.v[1] + vel.v[2] * vel.v[2]);

                    //collide
                    const T ck_u03 = vel.v[0] + vel.v[1];
                    const T ck_u04 = vel.v[0] - vel.v[1];
                    const T ck_u05 = vel.v[0] + vel.v[2];
                    const T ck_u06 = vel.v[0] - vel.v[2];
                    const T ck_u07 = vel.v[1] + vel.v[2];
                    const T ck_u08 = vel.v[1] - vel.v[2];

                    constexpr T c1over18 = 1. / 18.;
                    constexpr T c1over36 = 1. / 36.;
                    constexpr T c1over3 = 1. / 3.;
                    constexpr T c4dot5 = 4.5;
                    constexpr T c3 = 3.;
                    constexpr T c1 = 1.;
                    constexpr T c6 = 6.;

                    const T eq_00 = rho * c1over18 * (c1 - c3 * vel.v[0] + c4dot5 * vel.v[0] * vel.v[0] - usqr);
                    const T eq_01 = rho * c1over18 * (c1 - c3 * vel.v[1] + c4dot5 * vel.v[1] * vel.v[1] - usqr);
                    const T eq_02 = rho * c1over18 * (c1 - c3 * vel.v[2] + c4dot5 * vel.v[2] * vel.v[2] - usqr);
                    const T eq_03 = rho * c1over36 * (c1 - c3 * ck_u03 + c4dot5 * ck_u03 * ck_u03 - usqr);
                    const T eq_04 = rho * c1over36 * (c1 - c3 * ck_u04 + c4dot5 * ck_u04 * ck_u04 - usqr);
                    const T eq_05 = rho * c1over36 * (c1 - c3 * ck_u05 + c4dot5 * ck_u05 * ck_u05 - usqr);
                    const T eq_06 = rho * c1over36 * (c1 - c3 * ck_u06 + c4dot5 * ck_u06 * ck_u06 - usqr);
                    const T eq_07 = rho * c1over36 * (c1 - c3 * ck_u07 + c4dot5 * ck_u07 * ck_u07 - usqr);
                    const T eq_08 = rho * c1over36 * (c1 - c3 * ck_u08 + c4dot5 * ck_u08 * ck_u08 - usqr);

                    const T eqopp_00 = eq_00 + rho * c1over18 * c6 * vel.v[0];
                    const T eqopp_01 = eq_01 + rho * c1over18 * c6 * vel.v[1];
                    const T eqopp_02 = eq_02 + rho * c1over18 * c6 * vel.v[2];
                    const T eqopp_03 = eq_03 + rho * c1over36 * c6 * ck_u03;
                    const T eqopp_04 = eq_04 + rho * c1over36 * c6 * ck_u04;
                    const T eqopp_05 = eq_05 + rho * c1over36 * c6 * ck_u05;
                    const T eqopp_06 = eq_06 + rho * c1over36 * c6 * ck_u06;
                    const T eqopp_07 = eq_07 + rho * c1over36 * c6 * ck_u07;
                    const T eqopp_08 = eq_08 + rho * c1over36 * c6 * ck_u08;

                    const T pop_out_00 = (c1 - omega) * ins[0] + omega * eq_00;
                    const T pop_out_01 = (c1 - omega) * ins[1] + omega * eq_01;
                    const T pop_out_02 = (c1 - omega) * ins[2] + omega * eq_02;
                    const T pop_out_03 = (c1 - omega) * ins[3] + omega * eq_03;
                    const T pop_out_04 = (c1 - omega) * ins[4] + omega * eq_04;
                    const T pop_out_05 = (c1 - omega) * ins[5] + omega * eq_05;
                    const T pop_out_06 = (c1 - omega) * ins[6] + omega * eq_06;
                    const T pop_out_07 = (c1 - omega) * ins[7] + omega * eq_07;
                    const T pop_out_08 = (c1 - omega) * ins[8] + omega * eq_08;

                    const T pop_out_opp_00 = (c1 - omega) * ins[10] + omega * eqopp_00;
                    const T pop_out_opp_01 = (c1 - omega) * ins[11] + omega * eqopp_01;
                    const T pop_out_opp_02 = (c1 - omega) * ins[12] + omega * eqopp_02;
                    const T pop_out_opp_03 = (c1 - omega) * ins[13] + omega * eqopp_03;
                    const T pop_out_opp_04 = (c1 - omega) * ins[14] + omega * eqopp_04;
                    const T pop_out_opp_05 = (c1 - omega) * ins[15] + omega * eqopp_05;
                    const T pop_out_opp_06 = (c1 - omega) * ins[16] + omega * eqopp_06;
                    const T pop_out_opp_07 = (c1 - omega) * ins[17] + omega * eqopp_07;
                    const T pop_out_opp_08 = (c1 - omega) * ins[18] + omega * eqopp_08;

                    const T eq_09 = rho * c1over3 * (c1 - usqr);
                    const T pop_out_09 = (c1 - omega) * ins[9] + omega * eq_09;


                    //store operation
                    store<T>(cell, (storeOut) ? out : in, 0, pop_out_00);
                    store<T>(cell, (storeOut) ? out : in, 1, pop_out_01);
                    store<T>(cell, (storeOut) ? out : in, 2, pop_out_02);
                    store<T>(cell, (storeOut) ? out : in, 3, pop_out_03);
                    store<T>(cell, (storeOut) ? out : in, 4, pop_out_04);
                    store<T>(cell, (storeOut) ? out : in, 5, pop_out_05);
                    store<T>(cell, (storeOut) ? out : in, 6, pop_out_06);
                    store<T>(cell, (storeOut) ? out : in, 7, pop_out_07);
                    store<T>(cell, (storeOut) ? out : in, 8, pop_out_08);
                    store<T>(cell, (storeOut) ? out : in, 9, pop_out_09);

                    store<T>(cell, (storeOut) ? out : in, 10, pop_out_opp_00);
                    store<T>(cell, (storeOut) ? out : in, 11, pop_out_opp_01);
                    store<T>(cell, (storeOut) ? out : in, 12, pop_out_opp_02);
                    store<T>(cell, (storeOut) ? out : in, 13, pop_out_opp_03);
                    store<T>(cell, (storeOut) ? out : in, 14, pop_out_opp_04);
                    store<T>(cell, (storeOut) ? out : in, 15, pop_out_opp_05);
                    store<T>(cell, (storeOut) ? out : in, 16, pop_out_opp_06);
                    store<T>(cell, (storeOut) ? out : in, 17, pop_out_opp_07);
                    store<T>(cell, (storeOut) ? out : in, 18, pop_out_opp_08);

                    //streaming (push)
                    stream<T>(cell, out, (storeOut) ? out : in, type, 0, pop_out_00);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 1, pop_out_01);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 2, pop_out_02);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 3, pop_out_03);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 4, pop_out_04);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 5, pop_out_05);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 6, pop_out_06);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 7, pop_out_07);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 8, pop_out_08);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 9, pop_out_09);

                    stream<T>(cell, out, (storeOut) ? out : in, type, 10, pop_out_opp_00);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 11, pop_out_opp_01);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 12, pop_out_opp_02);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 13, pop_out_opp_03);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 14, pop_out_opp_04);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 15, pop_out_opp_05);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 16, pop_out_opp_06);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 17, pop_out_opp_07);
                    stream<T>(cell, out, (storeOut) ? out : in, type, 18, pop_out_opp_08);
                }
            };
        });
}


template <typename T, int Q>
inline Neon::set::Container collideKBCFusedAll(Neon::domain::mGrid&                        grid,
                                                       T                                           omega0,
                                                       int                                         level,
                                                       int                                         numLevels,
                                                       const Neon::domain::mGrid::Field<CellType>& cellType,
                                                       const Neon::domain::mGrid::Field<T>&        fin,
                                                       Neon::domain::mGrid::Field<T>&              fout,
                                                       bool                                        storeOut)
{

    if (level != 0) {
        Neon::NeonException exp("collideKBCFusedAll");
        exp << "Only works with on the finest level. Input level =" << level;
        NEON_THROW(exp);
    }

    return grid.newContainer(
        "CHSOE" + std::to_string(level), level,
        [&, level, omega0, numLevels, storeOut](Neon::set::Loader& loader) {
            const auto type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
            const auto in = fin.load(loader, level, Neon::MultiResCompute::MAP);
            auto       out = fout.load(loader, level, Neon::MultiResCompute::MAP);
            const T    omega = computeOmega(omega0, level, numLevels);
            const T    beta = omega * 0.5;
            const T    invBeta = 1.0 / beta;

            if (numLevels > 1) {
                //reload the next level as a map to indicate that we will (remote) write to it
                fout.load(loader, level + 1, Neon::MultiResCompute::MAP);
            }

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::mGrid::Idx& cell) mutable {
                if (type(cell, 0) == CellType::bulk) {

                    constexpr T tiny = 1e-7;

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


                    auto fdecompose_shear = [&](const int q) -> T {
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
                    for (int8_t q = 0; q < Q; ++q) {
                        T cu = 0;
                        for (int d = 0; d < 3; ++d) {
                            cu += latticeVelocity[q][d] * vel.v[d];
                        }
                        cu *= 3.0;

                        feq[q] = rho * latticeWeights[q] * (1. + cu + 0.5 * cu * cu - usqr);

                        fneq[q] = ins[q] - feq[q];
                    }

                    //momentum_flux
                    for (int8_t q = 0; q < Q; ++q) {
                        for (int i = 0; i < 6; ++i) {
                            Pi[i] += fneq[q] * latticeMoment[q][i];
                        }
                    }


                    //fdecompose_shear
                    for (int8_t q = 0; q < Q; ++q) {
                        deltaS[q] = rho * fdecompose_shear(q);

                        T deltaH = fneq[q] - deltaS[q];

                        e0 += (deltaS[q] * deltaH / feq[q]);
                        e1 += (deltaH * deltaH / feq[q]);
                    }

                    //gamma
                    T gamma = invBeta - (2.0 - invBeta) * e0 / (tiny + e1);


                    //fout
                    for (int8_t q = 0; q < Q; ++q) {
                        const T deltaH = fneq[q] - deltaS[q];

                        const T res = ins[q] - beta * (2.0 * deltaS[q] + gamma * deltaH);


                        //store operation
                        store<T>(cell, (storeOut) ? out : in, q, res);

                        //streaming (push)
                        stream<T>(cell, out, (storeOut) ? out : in, type, q, res);
                    }
                }
            };
        });
}
