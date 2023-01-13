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
            };
        });
}

template <typename T, int DIM, int Q>
Neon::set::Container stream(Neon::domain::mGrid&                 grid,
                            int                                  level,
                            const Neon::domain::mGrid::Field<T>& fpop_postcollision,
                            Neon::domain::mGrid::Field<T>&       fpop_poststreaming)
{
    //TODO
    //regular Streaming of the normal voxels at level L which are not interfaced with L+1 and L-1 levels.

    return grid.getContainer(
        "Stream" + std::to_string(level), level,
        [=](Neon::set::Loader& loader) {
            const auto& fpost_col = fpop_postcollision.load(loader, level, Neon::MultiResCompute::STENCIL);
            auto        fpost_stm = fpop_poststreaming.load(loader, level, Neon::MultiResCompute::MAP);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                for (int i = 0; i < Q; ++i) {

                    //TODO filter out cells that interface with L+1 or L-1
                    //TODO figure out the right direction to push/pull the data in the i-direction
                    fpost_stm(cell, i) = fpost_col(cell, i);
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
    //TODO
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
                for (int i = 0; i < Q; ++i) {
                }
            };
        });
}


template <typename T, int DIM, int Q>
Neon::set::Container coalescencePull(Neon::domain::mGrid&           grid,
                                     int                            level,
                                     Neon::domain::mGrid::Field<T>& fpop_poststreaming)
{
    //TODO
    // Initiated by the coarse level (hence "pull"), this function performs fine (level-1) to coarse
    // (level) communication or "coalescence" by simply averaging the fine data stored in self.fpop_halo

    return grid.getContainer(
        "explosionPull" + std::to_string(level), level,
        [=](Neon::set::Loader& loader) {
            auto fpost_stm = fpop_poststreaming.load(loader, level, Neon::MultiResCompute::STENCIL_DOWN);

            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {
                for (int i = 0; i < Q; ++i) {
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

    // 2) Storing fine(level) data for later "coalescence" pulled by the coarse(level)

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
        auto             runtime = Neon::Runtime::stream;
        std::vector<int> gpu_ids{0};
        Neon::Backend    backend(gpu_ids, runtime);

        constexpr int DIM = 2;
        constexpr int Q = (DIM == 2) ? 9 : 19;

        const int dim_x = 4;
        const int dim_y = 4;
        const int dim_z = (DIM < 3) ? 1 : 4;

        const Neon::index_3d grid_dim(dim_x, dim_y, dim_z);

        const Neon::domain::mGridDescriptor descriptor({1, 1, 1});


        Neon::domain::mGrid grid(
            backend, grid_dim,
            {[&](const Neon::index_3d id) -> bool {
                 return true;
             },
             [&](const Neon::index_3d&) -> bool {
                 return true;
             },
             [&](const Neon::index_3d&) -> bool {
                 return true;
             }},
            create_stencil<DIM, Q>(), descriptor);


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