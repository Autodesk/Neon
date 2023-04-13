#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"

#include "lattice.h"

NEON_CUDA_HOST_DEVICE Neon::int8_3d getDir(const int8_t q)
{
    return Neon::int8_3d(latticeVelocity[q][0], latticeVelocity[q][1], latticeVelocity[q][2]);
}


template <typename T>
NEON_CUDA_HOST_DEVICE inline Neon::int8_3d uncleOffset(const T& cell, const Neon::int8_3d& q)
{
    //given a local index within a cell and a population direction (q)
    //find the uncle's (the parent neighbor) offset from which the desired population (q) should be read
    //this offset is wrt the cell containing the localID (i.e., the parent of localID)

    auto off = [](const int8_t i, const int8_t j) {
        //0, -1 --> -1
        //1, -1 --> 0
        //0, 0 --> 0
        //0, 1 --> 0
        //1, 1 --> 1
        const int8_t s = i + j;
        return (s <= 0) ? s : s - 1;
    };

    Neon::int8_3d offset(off(cell.x, q.x), off(cell.y, q.y), off(cell.z, q.z));
    return offset;
}


template <typename T>
NEON_CUDA_HOST_DEVICE T computeOmega(T omega0, int level, int numLevels)
{
    int ilevel = numLevels - level - 1;
    // scalbln(1.0, x) = 2^x
    return 2 * omega0 / (scalbln(1.0, ilevel + 1) + (1. - scalbln(1.0, ilevel)) * omega0);
}

template <typename T, int Q>
NEON_CUDA_HOST_DEVICE Neon::Vec_3d<T> velocity(const T* fin,
                                               const T  rho)
{
    Neon::Vec_3d<T> vel(0, 0, 0);

    for (int i = 0; i < Q; ++i) {
        const T f = fin[i];
        for (int d = 0; d < 3; ++d) {
            vel.v[d] += f * latticeVelocity[i][d];
        }
    }

    for (int d = 0; d < 3; ++d) {
        vel.v[d] /= rho;
    }
    return vel;
}


template <typename T, int Q>
Neon::set::Container collideKBC(Neon::domain::mGrid&                        grid,
                                T                                           omega0,
                                int                                         level,
                                int                                         numLevels,
                                const Neon::domain::mGrid::Field<CellType>& cellType,
                                const Neon::domain::mGrid::Field<T>&        fin,
                                Neon::domain::mGrid::Field<T>&              fout)
{
    return grid.getContainer(
        "collide_" + std::to_string(level), level,
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
                        T deltaS[27];
                        T fneq[27];
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
        "collide_" + std::to_string(level), level,
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

template <typename T, int Q>
Neon::set::Container stream(Neon::domain::mGrid&                        grid,
                            int                                         level,
                            const Neon::domain::mGrid::Field<CellType>& cellType,
                            const Neon::domain::mGrid::Field<T>&        postCollision,
                            Neon::domain::mGrid::Field<T>&              postStreaming)
{
    //regular Streaming of the normal voxels at level L which are not interfaced with L+1 and L-1 levels.
    //This is "pull" stream

    return grid.getContainer(
        "stream_" + std::to_string(level), level,
        [&, level](Neon::set::Loader& loader) {
            const auto& type = cellType.load(loader, level, Neon::MultiResCompute::STENCIL);
            const auto& fpost_col = postCollision.load(loader, level, Neon::MultiResCompute::STENCIL);
            auto        fpost_stm = postStreaming.load(loader, level, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                if (type(cell, 0) == CellType::bulk) {
                    //If this cell has children i.e., it is been refined, than we should not work on it
                    //because this cell is only there to allow query and not to operate on
                    if (!fpost_stm.hasChildren(cell)) {

                        for (int8_t q = 0; q < Q; ++q) {
                            const Neon::int8_3d dir = -getDir(q);

                            //if the neighbor cell has children, then this 'cell' is interfacing with L-1 (fine) along q direction
                            if (!fpost_stm.hasChildren(cell, dir)) {
                                auto nghType = type.nghVal(cell, dir, 0, CellType::undefined);

                                if (nghType.isValid) {
                                    if (nghType.value == CellType::bulk) {
                                        fpost_stm(cell, q) = fpost_col.nghVal(cell, dir, q, T(0)).value;
                                    } else {
                                        const int8_t opposte_q = latticeOppositeID[q];
                                        fpost_stm(cell, q) = fpost_col(cell, opposte_q) + fpost_col.nghVal(cell, dir, opposte_q, T(0)).value;
                                    }
                                }
                            }
                        }
                    }
                }
            };
        });
}

template <typename T, int Q>
Neon::set::Container explosionPull(Neon::domain::mGrid&                 grid,
                                   int                                  level,
                                   const Neon::domain::mGrid::Field<T>& postCollision,
                                   Neon::domain::mGrid::Field<T>&       postStreaming)
{
    // Initiated by the fine level (hence "pull"), this function performs a coarse (level+1) to
    // fine (level) communication or "explosion" by simply distributing copies of coarse grid onto the fine grid.
    // In other words, this function updates the "halo" cells of the fine level by making copies of the coarse cell
    // values.


    return grid.getContainer(
        "Explosion_" + std::to_string(level), level,
        [&, level](Neon::set::Loader& loader) {
            const auto& fpost_col = postCollision.load(loader, level, Neon::MultiResCompute::STENCIL_UP);
            auto        fpost_stm = postStreaming.load(loader, level, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                //If this cell has children i.e., it is been refined, that we should not work on it
                //because this cell is only there to allow query and not to operate on
                if (!fpost_stm.hasChildren(cell)) {
                    for (int8_t q = 1; q < Q; ++q) {

                        const Neon::int8_3d dir = -getDir(q);

                        //if the neighbor cell has children, then this 'cell' is interfacing with L-1 (fine) along q direction
                        //we want to only work on cells that interface with L+1 (coarse) cell along q
                        if (!fpost_stm.hasChildren(cell, dir)) {

                            //try to query the cell along this direction (opposite of the population direction) as we do
                            //in 'normal' streaming
                            auto neighborCell = fpost_col.getNghCell(cell, dir);
                            if (!neighborCell.isActive()) {
                                //only if we can not do normal streaming, then we may have a coarser neighbor from which
                                //we can read this pop

                                //get the uncle direction/offset i.e., the neighbor of the cell's parent
                                //this direction/offset is wrt to the cell's parent
                                Neon::int8_3d uncleDir = uncleOffset(cell.mLocation, dir);

                                auto uncleLoc = fpost_col.getUncle(cell, uncleDir);

                                auto uncle = fpost_col.uncleVal(cell, uncleDir, q, T(0));
                                if (uncle.isValid) {
                                    fpost_stm(cell, q) = uncle.value;
                                }
                            }
                        }
                    }
                }
            };
        });
}


template <typename T, int Q>
Neon::set::Container coalescencePull(Neon::domain::mGrid&                 grid,
                                     int                                  level,
                                     const Neon::domain::mGrid::Field<T>& postCollision,
                                     Neon::domain::mGrid::Field<T>&       postStreaming)
{
    // Initiated by the coarse level (hence "pull"), this function simply read the missing population
    // across the interface between coarse<->fine boundary by reading the population prepare during the store()

    return grid.getContainer(
        "Coalescence" + std::to_string(level), level,
        [&, level](Neon::set::Loader& loader) {
            const auto& fpost_col = postCollision.load(loader, level, Neon::MultiResCompute::STENCIL);
            auto&       fpost_stm = postStreaming.load(loader, level, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                //If this cell has children i.e., it is been refined, than we should not work on it
                //because this cell is only there to allow query and not to operate on
                if (!fpost_stm.hasChildren(cell)) {

                    for (int q = 1; q < Q; ++q) {
                        const Neon::int8_3d dir = -getDir(q);
                        //if we have a neighbor at the same level that has been refined, then cell is on
                        //the interface and this is where we should do the coalescence
                        if (fpost_stm.hasChildren(cell, dir)) {
                            auto neighbor = fpost_col.nghVal(cell, dir, q, T(0));
                            if (neighbor.isValid) {
                                fpost_stm(cell, q) = neighbor.value;
                            }
                        }
                    }
                }
            };
        });
}


template <typename T, int Q>
Neon::set::Container store(Neon::domain::mGrid&           grid,
                           int                            level,
                           Neon::domain::mGrid::Field<T>& postCollision)
{
    //Initiated by the coarse level (level), this function prepares and stores the fine (level - 1)
    // information for further pulling initiated by the coarse (this) level invoked by coalescence_pull
    //
    //Where a coarse cell stores its information? at itself i.e., pull
    //Where a coarse cell reads the needed info? from its children and neighbor cell's children (level -1)
    //This function only operates on a coarse cell that has children.
    //For such cell, we check its neighbor cells at the same level. If any of these neighbor has NO
    //children, then we need to prepare something for them to be read during coalescence. What
    //we prepare is some sort of averaged the data from the children (the cell's children and/or
    //its neighbor's children)

    return grid.getContainer(
        "store_" + std::to_string(level), level,
        [&, level](Neon::set::Loader& loader) {
            auto& fpost_col = postCollision.load(loader, level, Neon::MultiResCompute::STENCIL_DOWN);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                //if the cell is refined, we might need to store something in it for its neighbor
                if (fpost_col.hasChildren(cell)) {

                    const int refFactor = fpost_col.getRefFactor(level);

                    bool should_accumelate = ((int(fpost_col(cell, 0)) % refFactor) != 0);

                    //fpost_col(cell, 0) += 1;
                    fpost_col(cell, 0) = (int(fpost_col(cell, 0)) + 1) % refFactor;


                    //for each direction aka for each neighbor
                    //we skip the center here
                    for (int8_t q = 1; q < Q; ++q) {
                        const Neon::int8_3d q_dir = getDir(q);

                        //check if the neighbor in this direction has children
                        auto neighborCell = fpost_col.getNghCell(cell, q_dir);
                        if (neighborCell.isActive()) {

                            if (!fpost_col.hasChildren(neighborCell)) {
                                //now, we know that there is actually something we need to store for this neighbor
                                //in cell along q (q_dir) direction
                                int num = 0;
                                T   sum = 0;


                                //for every neighbor cell including the center cell (i.e., cell)
                                for (int8_t p = 0; p < Q; ++p) {
                                    const Neon::int8_3d p_dir = getDir(p);

                                    const auto p_cell = fpost_col.getNghCell(cell, p_dir);
                                    //relative direction of q w.r.t p
                                    //i.e., in which direction we should move starting from p to land on q
                                    const Neon::int8_3d r_dir = q_dir - p_dir;

                                    //if this neighbor is refined
                                    if (fpost_col.hasChildren(cell, p_dir)) {

                                        //for each children of p
                                        for (int8_t i = 0; i < refFactor; ++i) {
                                            for (int8_t j = 0; j < refFactor; ++j) {
                                                for (int8_t k = 0; k < refFactor; ++k) {
                                                    const Neon::int8_3d c(i, j, k);

                                                    //cq is coarse neighbor (i.e., uncle) that we need to go in order to read q
                                                    //for c (this is what we do for explosion but here we do this just for the check)
                                                    const Neon::int8_3d cq = uncleOffset(c, q_dir);
                                                    if (cq == r_dir) {
                                                        auto childVal = fpost_col.childVal(p_cell, c, q, 0);
                                                        if (childVal.isValid) {
                                                            num++;
                                                            sum += childVal.value;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                if (should_accumelate) {
                                    fpost_col(cell, q) += sum / static_cast<T>(num * refFactor);
                                } else {
                                    fpost_col(cell, q) = sum / static_cast<T>(num * refFactor);
                                }
                            }
                        }
                    }
                }
            };
        });
}


template <typename T, int Q>
void stream(Neon::domain::mGrid&                        grid,
            int                                         level,
            const int                                   numLevels,
            const Neon::domain::mGrid::Field<CellType>& cellType,
            const Neon::domain::mGrid::Field<T>&        postCollision,  //fout
            Neon::domain::mGrid::Field<T>&              postStreaming,  //fin
            std::vector<Neon::set::Container>&          containers)
{
    containers.push_back(stream<T, Q>(grid, level, cellType, postCollision, postStreaming));

    /*
    * Streaming for interface voxels that have
    *  (i) coarser or (ii) finer neighbors at level+1 and level-1 and hence require
    *  (i) "explosion" or (ii) coalescence
    */
    if (level != numLevels - 1) {
        /* Explosion: pull missing populations from coarser neighbors by copying coarse (level+1) to fine (level) 
        * neighbors, initiated by the fine level ("Pull").
        */
        containers.push_back(explosionPull<T, Q>(grid, level, postCollision, postStreaming));
    }

    if (level != 0) {
        /* Coalescence: pull missing populations from finer neighbors by "smart" averaging fine (level-1) 
        * to coarse (level) communication, initiated by the coarse level ("Pull").
        */
        containers.push_back(coalescencePull<T, Q>(grid, level, postCollision, postStreaming));
    }
}

template <typename T, int Q>
void nonUniformTimestepRecursive(Neon::domain::mGrid&                        grid,
                                 const T                                     omega0,
                                 const int                                   level,
                                 const int                                   numLevels,
                                 const Neon::domain::mGrid::Field<CellType>& cellType,
                                 Neon::domain::mGrid::Field<T>&              fin,
                                 Neon::domain::mGrid::Field<T>&              fout,
                                 std::vector<Neon::set::Container>&          containers)
{
    // 1) collision for all voxels at level L=level
    containers.push_back(collideKBG<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));

    // 2) Storing fine (level - 1) data for later "coalescence" pulled by the coarse (level)
    if (level != numLevels - 1) {
        containers.push_back(store<T, Q>(grid, level + 1, fout));
    }


    // 3) recurse down
    if (level != 0) {
        nonUniformTimestepRecursive<T, Q>(grid, omega0, level - 1, numLevels, cellType, fin, fout, containers);
    }

    // 4) Streaming step that also performs the necessary "explosion" and "coalescence" steps.
    stream<T, Q>(grid, level, numLevels, cellType, fout, fin, containers);

    // 5) stop
    if (level == numLevels - 1) {
        return;
    }

    // 6) collision for all voxels at level L = level
    containers.push_back(collideKBG<T, Q>(grid, omega0, level, numLevels, cellType, fin, fout));

    // 7) Storing fine(level) data for later "coalescence" pulled by the coarse(level)
    if (level != numLevels - 1) {
        containers.push_back(store<T, Q>(grid, level + 1, fout));
    }

    // 8) recurse down
    if (level != 0) {
        nonUniformTimestepRecursive<T, Q>(grid, omega0, level - 1, numLevels, cellType, fin, fout, containers);
    }

    // 9) Streaming step
    stream<T, Q>(grid, level, numLevels, cellType, fout, fin, containers);
}


inline float sdfCube(Neon::index_3d id, Neon::index_3d dim, Neon::float_3d b = {1.0, 1.0, 1.0})
{
    auto mapToCube = [&](Neon::index_3d id) {
        //map p to an axis-aligned cube from -1 to 1
        Neon::float_3d half_dim = dim.newType<float>() * 0.5;
        Neon::float_3d ret = (id.newType<float>() - half_dim) / half_dim;
        return ret;
    };
    Neon::float_3d p = mapToCube(id);

    Neon::float_3d d(std::abs(p.x) - b.x, std::abs(p.y) - b.y, std::abs(p.z) - b.z);

    Neon::float_3d d_max(std::max(d.x, 0.f), std::max(d.y, 0.f), std::max(d.z, 0.f));
    float          len = std::sqrt(d_max.x * d_max.x + d_max.y * d_max.y + d_max.z * d_max.z);
    float          val = std::min(std::max(d.x, std::max(d.y, d.z)), 0.f) + len;
    return val;
}

template <typename T, int Q>
void postProcess(Neon::domain::mGrid&                        grid,
                 const int                                   numLevels,
                 const Neon::domain::mGrid::Field<T>&        fpop,
                 const Neon::domain::mGrid::Field<CellType>& cellType,
                 const int                                   iteration,
                 Neon::domain::mGrid::Field<T>&              vel,
                 Neon::domain::mGrid::Field<T>&              rho,
                 bool                                        generateValidateFile = false)
{
    grid.getBackend().syncAll();

    for (int level = 0; level < numLevels; ++level) {
        auto container =
            grid.getContainer(
                "postProcess_" + std::to_string(level), level,
                [&, level](Neon::set::Loader& loader) {
                    const auto& pop = fpop.load(loader, level, Neon::MultiResCompute::STENCIL);
                    const auto& type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&       u = vel.load(loader, level, Neon::MultiResCompute::MAP);
                    auto&       rh = rho.load(loader, level, Neon::MultiResCompute::MAP);


                    return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                        if (!pop.hasChildren(cell)) {
                            if (type(cell, 0) == CellType::bulk) {

                                //fin
                                T ins[Q];
                                for (int i = 0; i < Q; ++i) {
                                    ins[i] = pop(cell, i);
                                }

                                //density
                                T r = 0;
                                for (int i = 0; i < Q; ++i) {
                                    r += ins[i];
                                }
                                rh(cell, 0) = r;

                                //velocity
                                const Neon::Vec_3d<T> vel = velocity<T, Q>(ins, r);

                                u(cell, 0) = vel.v[0];
                                u(cell, 1) = vel.v[1];
                                u(cell, 2) = vel.v[2];
                            }
                            if (type(cell, 0) == CellType::movingWall) {
                                rh(cell, 0) = 1.0;

                                for (int d = 0; d < 3; ++d) {
                                    int i = (d == 0) ? 3 : ((d == 1) ? 1 : 9);
                                    u(cell, d) = pop(cell, i) / (6.0 * 1.0 / 18.0);
                                }
                            }
                        }
                    };
                });

        container.run(0);
    }

    grid.getBackend().syncAll();


    vel.updateIO();
    //rho.updateIO();


    int                precision = 4;
    std::ostringstream suffix;
    suffix << std::setw(precision) << std::setfill('0') << iteration;

    vel.ioToVtk("Velocity_" + suffix.str());
    //rho.ioToVtk("Density_" + suffix.str());

    if (generateValidateFile) {
        const Neon::index_3d grid_dim = grid.getDimension();
        std::ofstream        file;
        file.open("NeonMultiResLBM_" + suffix.str() + ".dat");

        for (int level = 0; level < numLevels; ++level) {
            vel.forEachActiveCell(
                level, [&](const Neon::index_3d& id, const int& card, T& val) {
                    if (id.x == grid_dim.x / 2 && id.z == grid_dim.z / 2) {
                        if (card == 0) {
                            file << static_cast<double>(id.v[1]) / static_cast<double>(grid_dim.y) << " " << val << "\n";
                        }
                    }
                },
                Neon::computeMode_t::seq);
        }
        file.close();
    }
}

int main(int argc, char** argv)
{
    Neon::init();

    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using T = double;

        //Neon grid
        auto             runtime = Neon::Runtime::stream;
        std::vector<int> gpu_ids{0};
        Neon::Backend    backend(gpu_ids, runtime);

        constexpr int Q = 27;
        constexpr int depth = 3;

        const Neon::domain::mGridDescriptor descriptor(depth);

        const Neon::index_3d grid_dim(160, 160, 160);
        float                levelSDF[depth + 1];
        levelSDF[0] = 0;
        levelSDF[1] = -31.0 / 160.0;
        levelSDF[2] = -64 / 160.0;
        levelSDF[3] = -1.0;

        // const Neon::index_3d grid_dim(48, 48, 48);
        // float                levelSDF[depth + 1];
        // levelSDF[0] = 0;
        // levelSDF[1] = -8 / 24.0;
        // levelSDF[2] = -16 / 24.0;
        // levelSDF[3] = -1.0;

        Neon::domain::mGrid grid(
            backend, grid_dim,
            {[&](const Neon::index_3d id) -> bool {
                 return sdfCube(id, grid_dim - 1) <= levelSDF[0] &&
                        sdfCube(id, grid_dim - 1) > levelSDF[1];
             },
             [&](const Neon::index_3d& id) -> bool {
                 return sdfCube(id, grid_dim - 1) <= levelSDF[1] &&
                        sdfCube(id, grid_dim - 1) > levelSDF[2];
             },
             [&](const Neon::index_3d& id) -> bool {
                 return sdfCube(id, grid_dim - 1) <= levelSDF[2] &&
                        sdfCube(id, grid_dim - 1) > levelSDF[3];
             }},
            Neon::domain::Stencil::s19_t(false), descriptor);

        //LBM problem
        const int             max_iter = 400000;
        const T               ulb = 0.04;
        const T               Re = 100;
        const T               clength = T(grid.getDimension(descriptor.getDepth() - 1).x);
        const T               visclb = ulb * clength / Re;
        const T               omega = 1.0 / (3. * visclb + 0.5);
        const Neon::double_3d ulid(ulb, 0., 0.);

        //alloc fields
        auto fin = grid.newField<T>("fin", Q, 0);
        auto fout = grid.newField<T>("fout", Q, 0);
        auto cellType = grid.newField<CellType>("CellType", 1, CellType::bulk);

        auto vel = grid.newField<T>("vel", 3, 0);
        auto rho = grid.newField<T>("rho", 1, 0);

        int* d_num_elements = nullptr;
        if (runtime == Neon::Runtime::stream) {
            cudaMalloc((void**)&d_num_elements, sizeof(int));
            cudaMemset(d_num_elements, 0, sizeof(int));
        }

        //init fields
        for (int level = 0; level < descriptor.getDepth(); ++level) {

            auto container =
                grid.getContainer(
                    "Init_" + std::to_string(level), level,
                    [&fin, &fout, &cellType, &vel, &rho, level, grid_dim, ulid, Q, d_num_elements](Neon::set::Loader& loader) {
                        auto& in = fin.load(loader, level, Neon::MultiResCompute::MAP);
                        auto& out = fout.load(loader, level, Neon::MultiResCompute::MAP);
                        auto& type = cellType.load(loader, level, Neon::MultiResCompute::MAP);
                        auto& u = vel.load(loader, level, Neon::MultiResCompute::MAP);
                        auto& rh = rho.load(loader, level, Neon::MultiResCompute::MAP);

                        return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                            //velocity and density
                            u(cell, 0) = 0;
                            u(cell, 1) = 0;
                            u(cell, 2) = 0;
                            rh(cell, 0) = 0;
                            type(cell, 0) = CellType::bulk;

#ifdef __CUDA_ARCH__
                            atomicAdd(d_num_elements, 1);
#else
                            (void*)d_num_elements;
#endif

                            if (!in.hasChildren(cell)) {
                                const Neon::index_3d idx = in.mapToGlobal(cell);

                                //pop
                                for (int q = 0; q < Q; ++q) {
                                    T pop_init_val = latticeWeights[q];

                                    if (level == 0) {
                                        if (idx.x == 0 || idx.x == grid_dim.x - 1 ||
                                            idx.y == 0 || idx.y == grid_dim.y - 1 ||
                                            idx.z == 0 || idx.z == grid_dim.z - 1) {
                                            type(cell, 0) = CellType::bounceBack;

                                            if (idx.y == grid_dim.y - 1) {
                                                type(cell, 0) = CellType::movingWall;
                                                pop_init_val = 0;
                                                for (int d = 0; d < 3; ++d) {
                                                    pop_init_val += latticeVelocity[q][d] * ulid.v[d];
                                                }
                                                pop_init_val *= -6. * latticeWeights[q];
                                            } else {
                                                pop_init_val = 0;
                                            }
                                        }
                                    }

                                    out(cell, q) = pop_init_val;
                                    in(cell, q) = pop_init_val;
                                }
                            } else {
                                in(cell, 0) = 0;
                                out(cell, 0) = 0;
                            }
                        };
                    });

            container.run(0);
        }

        grid.getBackend().syncAll();


        int h_num_elements = 0;
        if (runtime == Neon::Runtime::stream) {
            cudaMemcpy(&h_num_elements, d_num_elements, sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_num_elements);
        }

        //skeleton
        std::vector<Neon::set::Container> containers;
        nonUniformTimestepRecursive<T, Q>(grid,
                                          omega,
                                          descriptor.getDepth() - 1,
                                          descriptor.getDepth(),
                                          cellType,
                                          fin, fout, containers);

        Neon::skeleton::Skeleton skl(grid.getBackend());
        skl.sequence(containers, "MultiResLBM");
        // skl.ioToDot("MultiResLBM", "", true);

        //execution
        auto start = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < max_iter; ++t) {
            printf("\n Iteration = %d", t);
            skl.run();
            if (t % 100 == 0) {
                postProcess<T, Q>(grid, descriptor.getDepth(), fout, cellType, t, vel, rho, false);
            }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        double mlups = static_cast<double>(max_iter) / duration.count();
        mlups *= static_cast<double>(h_num_elements);

        //double mlups = static_cast<double>(h_num_elements * max_iter) / duration.count();
        double eff_mlups = static_cast<double>(max_iter) / duration.count();
        eff_mlups *= static_cast<double>(grid_dim.x * grid_dim.y * grid_dim.z);

        std::cout << "MLUPS=     " << mlups << " num_elements= " << h_num_elements << std::endl;
        std::cout << "Eff MLUPS= " << eff_mlups << " eff num_elements= " << grid_dim.x * grid_dim.y * grid_dim.z << std::endl;

        postProcess<T, Q>(grid, descriptor.getDepth(), fout, cellType, max_iter, vel, rho, true);
    }
}