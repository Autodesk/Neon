#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"

template <unsigned int DIM, unsigned int Q>
Neon::domain::Stencil create_stencil();

template <>
Neon::domain::Stencil create_stencil<2, 9>()
{
    std::vector<Neon::index_3d> stencil;
    stencil.reserve(9);
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            stencil.emplace_back(Neon::index_3d(x, y, 0));
        }
    }
    return Neon::domain::Stencil(stencil);
}

template <>
Neon::domain::Stencil create_stencil<3, 19>()
{
    // filterCenterOut = false;
    return Neon::domain::Stencil::s19_t(false);
}

NEON_CUDA_DEVICE_ONLY static constexpr char latticeVelocity2D[9][2] = {
    {0, 0},
    {0, -1},
    {0, 1},
    {-1, 0},
    {-1, -1},
    {-1, 1},
    {1, 0},
    {1, -1},
    {1, 1}};

NEON_CUDA_DEVICE_ONLY static constexpr char latticeVelocity3D[27][3] = {
    {0, 0, 0},
    {0, -1, 0},
    {0, 1, 0},
    {-1, 0, 0},
    {-1, -1, 0},
    {-1, 1, 0},
    {1, 0, 0},
    {1, -1, 0},
    {1, 1, 0},

    {0, 0, -1},
    {0, -1, -1},
    {0, 1, -1},
    {-1, 0, -1},
    {-1, -1, -1},
    {-1, 1, -1},
    {1, 0, -1},
    {1, -1, -1},
    {1, 1, -1},

    {0, 0, 1},
    {0, -1, 1},
    {0, 1, 1},
    {-1, 0, 1},
    {-1, -1, 1},
    {-1, 1, 1},
    {1, 0, 1},
    {1, -1, 1},
    {1, 1, 1}

};


template <int DIM>
NEON_CUDA_HOST_DEVICE Neon::int8_3d getDir(const int8_t q)
{
    if constexpr (DIM == 2) {
        return Neon::int8_3d(latticeVelocity2D[q][0], latticeVelocity2D[q][1], 0);
    }
    if constexpr (DIM == 3) {
        return Neon::int8_3d(latticeVelocity3D[q][0], latticeVelocity3D[q][1], latticeVelocity3D[q][2]);
    }
}

template <int DIM, int Q>
struct latticeWeight
{
    NEON_CUDA_HOST_DEVICE __inline__ constexpr latticeWeight()
        : t()
    {
        if constexpr (DIM == 2) {

            for (int i = 0; i < Q; ++i) {
                if (latticeVelocity2D[i][0] * latticeVelocity2D[i][0] +
                        latticeVelocity2D[i][1] * latticeVelocity2D[i][1] <
                    1.1f) {
                    t[i] = 1.0f / 9.0f;
                } else {
                    t[i] = 1.0f / 36.0f;
                }
            }
            t[0] = 4.0f / 9.0f;
        }

        if constexpr (DIM == 3) {
            for (int i = 0; i < Q; ++i) {
                if (latticeVelocity2D[i][0] * latticeVelocity2D[i][0] +
                        latticeVelocity2D[i][1] * latticeVelocity2D[i][1] +
                        latticeVelocity2D[i][2] * latticeVelocity2D[i][2] <
                    1.1f) {
                    t[i] = 2.0f / 36.0f;
                } else {
                    t[i] = 1.0f / 36.0f;
                }
            }
            t[0] = 1.0f / 3.0f;
        }
    }
    float t[Q];
};


template <typename Field>
inline void exportVTI(const int t, Field& field)
{
    printf("\n Exporting Frame =%d", t);
    int                precision = 4;
    std::ostringstream oss;
    oss << std::setw(precision) << std::setfill('0') << t;
    std::string prefix = "lbm" + std::to_string(field.getCardinality()) + "D_";
    std::string fname = prefix + oss.str();
    field.ioToVtk(fname, "field");
}

template <typename T>
NEON_CUDA_HOST_DEVICE inline Neon::int8_3d unlceOffset(const T& cell, const Neon::int8_3d& q)
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
NEON_CUDA_HOST_DEVICE T computeOmega(T omega0, int level, int num_levels)
{
    int ilevel = num_levels - level - 1;
    // scalbln(1.0, x) = 2^x
    return 2 * omega0 / (scalbln(1.0, ilevel + 1) + (1. - scalbln(1.0, ilevel)) * omega0);
}

template <typename T, int DIM, int Q>
NEON_CUDA_HOST_DEVICE Neon::Vec_3d<T> velocity(const T* fin,
                                               const T  rho)
{
    Neon::Vec_3d<T> vel(0, 0, 0);
    if constexpr (DIM == 2) {
        for (int i = 0; i < Q; ++i) {
            const T f = fin[i];
            for (int d = 0; d < DIM; ++d) {
                vel.v[d] += f * latticeVelocity2D[i][d];
            }
        }
    }

    if constexpr (DIM == 3) {
        for (int i = 0; i < Q; ++i) {
            const T f = fin[i];
            for (int d = 0; d < DIM; ++d) {
                vel.v[d] += f * latticeVelocity3D[i][d];
            }
        }
    }

    for (int d = 0; d < DIM; ++d) {
        vel.v[d] /= rho;
    }
    return vel;
}

template <typename T, int DIM, int Q>
Neon::set::Container collide(Neon::domain::mGrid&                 grid,
                             T                                    omega0,
                             int                                  level,
                             int                                  max_level,
                             const Neon::domain::mGrid::Field<T>& fin,
                             Neon::domain::mGrid::Field<T>&       fout)
{
    return grid.getContainer(
        "Collide" + std::to_string(level), level,
        [=](Neon::set::Loader& loader) {
            const auto& in = fin.load(loader, level, Neon::MultiResCompute::MAP);
            auto        out = fout.load(loader, level, Neon::MultiResCompute::MAP);
            const T     omega = computeOmega(omega0, level, max_level);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                if (!in.hasChildren(cell)) {

                    constexpr auto t = latticeWeight<DIM, Q>();

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
                    const Neon::Vec_3d<T> vel = velocity<T, DIM, Q>(ins, rho);


                    const T usqr = (3.0 / 2.0) * (vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);
                    for (int i = 0; i < Q; ++i) {
                        T cu = 0;
                        for (int d = 0; d < DIM; ++d) {
                            cu += latticeVelocity2D[i][d] * vel.v[d];
                        }
                        //equilibrium
                        T feq = rho * t.t[i] * (1. + cu + 0.5 * cu * cu - usqr);

                        //collide
                        out(cell, i) = ins[i] - omega * (ins[i] - feq);
                    }
                }
            };
        });
}

template <typename T, int DIM, int Q>
Neon::set::Container stream(Neon::domain::mGrid&                 grid,
                            int                                  level,
                            const Neon::domain::mGrid::Field<T>& fpop_postcollision,
                            Neon::domain::mGrid::Field<T>&       fpop_poststreaming)
{
    //regular Streaming of the normal voxels at level L which are not interfaced with L+1 and L-1 levels.
    //This is "pull" stream

    return grid.getContainer(
        "Stream" + std::to_string(level), level,
        [=](Neon::set::Loader& loader) {
            const auto& fpost_col = fpop_postcollision.load(loader, level, Neon::MultiResCompute::STENCIL);
            auto        fpost_stm = fpop_poststreaming.load(loader, level, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                //If this cell has children i.e., it is been refined, than we should not work on it
                //because this cell is only there to allow query and not to operate on
                if (!fpost_stm.hasChildren(cell)) {

                    for (int8_t q = 0; q < Q; ++q) {
                        const Neon::int8_3d dir = -getDir<DIM>(q);

                        //if the neighbor cell has children, then this 'cell' is interfacing with L-1 (fine) along q direction
                        if (!fpost_stm.hasChildren(cell, dir)) {
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

template <typename T, int DIM, int Q>
Neon::set::Container explosionPull(Neon::domain::mGrid&                 grid,
                                   int                                  level,
                                   const Neon::domain::mGrid::Field<T>& fpop_postcollision,
                                   Neon::domain::mGrid::Field<T>&       fpop_poststreaming)
{
    // Initiated by the fine level (hence "pull"), this function performs a coarse (level+1) to
    // fine (level) communication or "explosion" by simply distributing copies of coarse grid onto the fine grid.
    // In other words, this function updates the "halo" cells of the fine level by making copies of the coarse cell
    // values.


    return grid.getContainer(
        "explosionPull" + std::to_string(level), level,
        [=](Neon::set::Loader& loader) {
            const auto& fpost_col = fpop_postcollision.load(loader, level, Neon::MultiResCompute::STENCIL_UP);
            auto        fpost_stm = fpop_poststreaming.load(loader, level, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                //If this cell has children i.e., it is been refined, that we should not work on it
                //because this cell is only there to allow query and not to operate on
                if (!fpost_stm.hasChildren(cell)) {
                    for (int8_t q = 0; q < Q; ++q) {

                        const Neon::int8_3d dir = -getDir<DIM>(q);

                        //if the neighbor cell has children, then this 'cell' is interfacing with L-1 (fine) along q direction
                        //we want to only work on cells that interface with L+1 (coarse) cell along q
                        if (!fpost_stm.hasChildren(cell, dir)) {

                            //try to query the cell along this direction (opposite of the population direction) as we do
                            //in 'normal' streaming
                            auto neighborCell = fpost_col.setNghCell(cell, dir);
                            if (!neighborCell.isActive()) {
                                //only if we can not do normal streaming, then we may have a coarser neighbor from which
                                //we can read this pop

                                //get the uncle direction/offset i.e., the neighbor of the cell's parent
                                //this direction/offset is wrt to the cell's parent
                                Neon::int8_3d uncleDir = unlceOffset(cell.mLocation, dir);

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


template <typename T, int DIM, int Q>
Neon::set::Container coalescencePull(Neon::domain::mGrid&           grid,
                                     int                            level,
                                     Neon::domain::mGrid::Field<T>& fpop_poststreaming)
{
    // Initiated by the coarse level (hence "pull"), this function simply read the missing population
    // across the interface between coarse<->fine boundary by reading the population prepare during the store()

    return grid.getContainer(
        "coalescencePull" + std::to_string(level), level,
        [=](Neon::set::Loader& loader) {
            auto fpost_stm = fpop_poststreaming.load(loader, level, Neon::MultiResCompute::STENCIL);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                //If this cell has children i.e., it is been refined, than we should not work on it
                //because this cell is only there to allow query and not to operate on
                if (!fpost_stm.hasChildren(cell)) {

                    for (int q = 0; q < Q; ++q) {
                        const Neon::int8_3d dir = -getDir<DIM>(q);
                        //if we have a neighbor at the same level that has been refined, then cell is on
                        //the interface and this is where we should do the coalescence
                        if (fpost_stm.hasChildren(cell, dir)) {
                            auto neighbor = fpost_stm.nghVal(cell, dir, q, T(0));
                            if (neighbor.isValid) {
                                fpost_stm(cell, q) = neighbor.value;
                            }
                        }
                    }
                }
            };
        });
}


template <typename T, int DIM, int Q>
Neon::set::Container store(Neon::domain::mGrid&           grid,
                           int                            level,
                           Neon::domain::mGrid::Field<T>& fpop_poststreaming)
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
        "store" + std::to_string(level), level,
        [=](Neon::set::Loader& loader) {
            auto fpost_stm = fpop_poststreaming.load(loader, level, Neon::MultiResCompute::STENCIL_DOWN);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                //if the cell is refined, we might need to store something in it for its neighbor
                if (fpost_stm.hasChildren(cell)) {

                    const int refFactor = fpost_stm.getRefFactor(level);

                    bool should_accumelate =  ((int(fpost_stm(cell, 0)) % refFactor) != 0);

                    fpost_stm(cell, 0) += 1;


                    //for each direction aka for each neighbor
                    //we skip the center here
                    for (int8_t q = 1; q < Q; ++q) {
                        const Neon::int8_3d q_dir = getDir<DIM>(q);

                        //check if the neighbor in this direction has children
                        if (!fpost_stm.hasChildren(cell, q_dir)) {
                            //now, we know that there is actually something we need to store for this neighbor
                            //in cell along q (q_dir) direction
                            int num = 0;
                            T   sum = 0;


                            //for every neighbor cell including the center cell (i.e., cell)
                            for (int8_t p = 0; p < Q; ++p) {
                                const Neon::int8_3d p_dir = getDir<DIM>(p);

                                //relative direction of q w.r.t p
                                //i.e., in which direction we should move starting from p to land on q
                                const Neon::int8_3d r_dir = q_dir - p_dir;

                                //if this neighbor is refined
                                if (fpost_stm.hasChildren(cell, p_dir)) {

                                    //for each children of p
                                    for (int8_t i = 0; i < refFactor; ++i) {
                                        for (int8_t j = 0; j < refFactor; ++j) {
                                            for (int8_t k = 0; k < refFactor; ++k) {
                                                const Neon::int8_3d c(i, j, k);

                                                //cq is coarse neighbor (i.e., uncle) that we need to go in order to read q
                                                //for c (this is what we do for explosion but here we do this just for the check)
                                                const Neon::int8_3d cq = unlceOffset(c, q_dir);
                                                if (cq == r_dir) {
                                                    num++;
                                                    sum += fpost_stm.childVal(cell, c, q, 0).value;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if (should_accumelate) {
                                fpost_stm(cell, q) += sum / static_cast<T>(num * refFactor);
                            } else {
                                fpost_stm(cell, q) = sum / static_cast<T>(num * refFactor);
                            }
                        }
                    }
                }
            };
        });
}


template <typename T, int DIM, int Q>
void stream(Neon::domain::mGrid&                 grid,
            int                                  level,
            const int                            max_level,
            const Neon::domain::mGrid::Field<T>& fpop_postcollision,
            Neon::domain::mGrid::Field<T>&       fpop_poststreaming,
            std::vector<Neon::set::Container>&   containers)
{
    containers.push_back(stream<T, DIM, Q>(grid, level, fpop_postcollision, fpop_poststreaming));

    /*
    * Streaming for interface voxels that have
    *  (i) coarser or (ii) finer neighbors at level+1 and level-1 and hence require
    *  (i) "explosion" or (ii) coalescence
    */
    if (level != max_level - 1) {
        /* Explosion: pull missing populations from coarser neighbors by copying coarse (level+1) to fine (level) 
        * neighbors, initiated by the fine level ("Pull").
        */
        containers.push_back(explosionPull<T, DIM, Q>(grid, level, fpop_postcollision, fpop_poststreaming));
    }

    if (level != 0) {
        /* Coalescence: pull missing populations from finer neighbors by "smart" averaging fine (level-1) 
        * to coarse (level) communication, initiated by the coarse level ("Pull").
        */
        containers.push_back(coalescencePull<T, DIM, Q>(grid, level, fpop_poststreaming));
    }
}

template <typename T, int DIM, int Q>
void nonUniformTimestepRecursive(Neon::domain::mGrid&               grid,
                                 const T                            omega0,
                                 const int                          level,
                                 const int                          max_level,
                                 Neon::domain::mGrid::Field<T>&     fin,
                                 Neon::domain::mGrid::Field<T>&     fout,
                                 std::vector<Neon::set::Container>& containers)
{
    // 1) collision for all voxels at level L=level
    containers.push_back(collide<T, DIM, Q>(grid, omega0, level, max_level, fin, fout));

    // 2) Storing fine (level - 1) data for later "coalescence" pulled by the coarse (level)
    if (level != max_level) {
        store<T, DIM, Q>(grid, level + 1, fout);
    }


    // 3) recurse down
    if (level != 0) {
        nonUniformTimestepRecursive<T, DIM, Q>(grid, omega0, level - 1, max_level, fin, fout, containers);
    }

    // 4) Streaming step that also performs the necessary "explosion" and "coalescence" steps.
    stream<T, DIM, Q>(grid, level, max_level, fout, fin, containers);

    // 5) stop
    if (level == max_level - 1) {
        return;
    }

    // 6) collision for all voxels at level L = level
    containers.push_back(collide<T, DIM, Q>(grid, omega0, level, max_level, fin, fout));

    // 7) Storing fine(level) data for later "coalescence" pulled by the coarse(level)
    if (level != max_level) {
        store<T, DIM, Q>(grid, level + 1, fout);
    }

    // 8) recurse down
    if (level != 0) {
        nonUniformTimestepRecursive<T, DIM, Q>(grid, omega0, level - 1, max_level, fin, fout, containers);
    }

    // 9) Streaming step
    stream<T, DIM, Q>(grid, level, max_level, fout, fin, containers);
}


int main(int argc, char** argv)
{
    Neon::init();

    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using T = double;

        //Neon grid
        auto             runtime = Neon::Runtime::openmp;
        std::vector<int> gpu_ids{0};
        Neon::Backend    backend(gpu_ids, runtime);

        constexpr int DIM = 2;
        constexpr int Q = (DIM == 2) ? 9 : 19;

        const int dim_x = 12;
        const int dim_y = 12;
        const int dim_z = (DIM < 3) ? 4 : 4;

        const Neon::index_3d grid_dim(dim_x, dim_y, dim_z);

        const Neon::domain::mGridDescriptor descriptor({1, 1, 1});


        Neon::domain::mGrid grid(
            backend, grid_dim,
            {[&](const Neon::index_3d id) -> bool {
                 return id.x > 7;
             },
             [&](const Neon::index_3d& id) -> bool {
                 return id.x > 3;
             },
             [&](const Neon::index_3d&) -> bool {
                 return true;
             }},
            create_stencil<DIM, Q>(), descriptor);

        //grid.topologyToVTK("lbm.vtk", false);


        //LBM problem
        const int max_iter = 300;
        const T   ulb = 0.01;
        const T   Re = 20;
        const T   clength = grid_dim.x;
        const T   visclb = ulb * clength / Re;
        const T   smagorinskyConstant = 0.02;
        const T   omega = 1.0 / (3. * visclb + 0.5);

        auto fin = grid.newField<T>("fin", Q, 0);
        auto fout = grid.newField<T>("fout", Q, 0);

        //TODO init fin and fout
        for (int l = 0; l < descriptor.getDepth(); ++l) {
            fin.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, T& val) {
                    val = 0;
                });
            fout.forEachActiveCell(
                l,
                [&](const Neon::int32_3d, const int, T& val) {
                    val = 0;
                });
        }

        fin.updateCompute();
        fout.updateCompute();

        std::vector<Neon::set::Container> containers;

        nonUniformTimestepRecursive<T, DIM, Q>(grid,
                                               omega,
                                               descriptor.getDepth() - 1,
                                               descriptor.getDepth(),
                                               fin, fout, containers);

        Neon::skeleton::Skeleton skl(grid.getBackend());
        skl.sequence(containers, "MultiResLBM");
        //skl.ioToDot("MultiRes");

        skl.run();

        grid.getBackend().syncAll();
        fin.updateIO();
        fout.updateIO();
    }
}