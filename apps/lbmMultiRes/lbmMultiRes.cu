#include "Neon/Neon.h"
#include "Neon/domain/mGrid.h"
#include "Neon/skeleton/Skeleton.h"

template <unsigned int DIM, unsigned int COMP>
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

/*Neon::set::Container setOmega(Neon::domain::mGrid& grid, int level)
{
    return grid.getContainer(
        "SetOmega" + std::to_string(level), level,
        [=](Neon::set::Loader& loader) {
            //auto& local = field.load(loader, level, Neon::MultiResCompute::MAP);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {};
        });
}*/

template <typename T>
NEON_CUDA_HOST_DEVICE T computeOmega(T omega0, int level, int num_levels)
{
    int ilevel = num_levels - level - 1;
    // scalbln(1.0, x) = 2^x
    return 2 * omega0 / (scalbln(1.0, ilevel + 1) + (1. - scalbln(1.0, ilevel)) * omega0);
}

template <typename T>
Neon::set::Container velocity(Neon::domain::mGrid& grid, int level, Neon::domain::mGrid::Field<T>& fin)
{
    return grid.getContainer(
        "SetOmega" + std::to_string(level), level,
        [=](Neon::set::Loader& loader) {
            auto& flocal = fin.load(loader, level, Neon::MultiResCompute::MAP);
            return [=] NEON_CUDA_HOST_DEVICE(const typename Neon::domain::bGrid::Cell& cell) mutable {};
        });
}

template <typename T>
void nonUniformTimestepRecursive(Neon::domain::mGrid&               grid,
                                 T                                  omega0,
                                 int                                level,
                                 int                                max_level,
                                 Neon::domain::mGrid::Field<T>&     fin,
                                 Neon::domain::mGrid::Field<T>&     fout,
                                 std::vector<Neon::set::Container>& containers)
{
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
        constexpr int COMP = (DIM == 2) ? 9 : 19;

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
            create_stencil<DIM, COMP>(), descriptor);


        //LBM problem
        const int max_iter = 300;
        const T   ulb = 0.01;
        const T   Re = 20;
        const T   clength = grid_dim.x;
        const T   visclb = ulb * clength / Re;
        const T   smagorinskyConstant = 0.02;
        const T   omega = 1.0 / (3. * visclb + 0.5);

        auto fin = grid.newField<T>("fin", COMP, 0);
        auto fout = grid.newField<T>("fout", COMP, 0);

        //TODO init fin and fout

        fin.updateCompute();
        fout.updateCompute();

        std::vector<Neon::set::Container> containers;

        nonUniformTimestepRecursive(grid, omega, 0, descriptor.getDepth(), fin, fout, containers);

        Neon::skeleton::Skeleton skl(grid.getBackend());
        skl.sequence(containers, "MultiResLBM");
        //skl.ioToDot("MultiRes");

        skl.run();

        grid.getBackend().syncAll();
        fin.updateIO();
        fout.updateIO();
    }
}