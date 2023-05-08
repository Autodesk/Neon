// References
// https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
// https://tebs-game-of-life.com/rainbow/rainbow.html

#include <assert.h>
#include <cstdlib>
#include <iomanip>
#include <vector>

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "Neon/skeleton/Skeleton.h"

template <typename FieldT>
inline void exportVTI(FieldT& voxel_1, FieldT& voxel_2, int frame_id)
{
    auto io = [&](int f, FieldT& voxel) {
        printf("\n Exporting Frame =%d", f);
        int precision = 4;
        voxel.updateHostData(0);
        std::ostringstream oss;
        oss << std::setw(precision) << std::setfill('0') << f;
        std::string fname = "gameOfLife_" + oss.str();
        voxel.ioToVtk(fname, "GoL");
    };
    io(2 * frame_id, voxel_2);
    io(2 * frame_id + 1, voxel_1);
}

Neon::domain::Stencil createStencil()
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

template <typename FieldT>
inline Neon::set::Container GoLContainer(const FieldT&         in_cells,
                                         FieldT&               out_cells,
                                         typename FieldT::Type length)
{
    using T = typename FieldT::Type;
    return in_cells.getGrid().getContainer(
        "GoLContainer", [&, length](Neon::set::Loader& L) {
            const auto& ins = L.load(in_cells, Neon::Compute::STENCIL);
            auto&       out = L.load(out_cells);

            return [=] NEON_CUDA_HOST_DEVICE(
                       const typename FieldT::Cell& idx) mutable {
                typename FieldT::ngh_idx ngh(0, 0, 0);
                const T                  default_value = 0;
                int                      alive = 0;
                T                        value = 0;
                T                        status = ins.nghVal(idx, ngh, 0, default_value).value;

                //+x
                ngh.x = 1;
                ngh.y = 0;
                ngh.z = 0;
                value = ins.nghVal(idx, ngh, 0, default_value).value;
                alive += (value > 0.0 ? 1 : 0);
                ngh.y = 1;
                value = ins.nghVal(idx, ngh, 0, default_value).value;
                alive += (value > 0.0 ? 1 : 0);

                //-x
                ngh.x = -1;
                ngh.y = 0;
                ngh.z = 0;
                value = ins.nghVal(idx, ngh, 0, default_value).value;
                alive += (value > 0.0 ? 1 : 0);
                ngh.y = -1;
                value = ins.nghVal(idx, ngh, 0, default_value).value;
                alive += (value > 0.0 ? 1 : 0);

                //+y
                ngh.x = 0;
                ngh.y = 1;
                ngh.z = 0;
                value = ins.nghVal(idx, ngh, 0, default_value).value;
                alive += (value > 0.0 ? 1 : 0);
                ngh.x = -1;
                value = ins.nghVal(idx, ngh, 0, default_value).value;
                alive += (value > 0.0 ? 1 : 0);

                //-y
                ngh.x = 0;
                ngh.y = -1;
                ngh.z = 0;
                value = ins.nghVal(idx, ngh, 0, default_value).value;
                alive += (value > 0.0 ? 1 : 0);
                ngh.x = 1;
                value = ins.nghVal(idx, ngh, 0, default_value).value;
                alive += (value > 0.0 ? 1 : 0);

                auto id_global = ins.mapToGlobal(idx);
                out(idx, 0) = ((T)id_global.x / length) * (T)((alive == 3 || (alive == 2 && status) ? 1 : 0));
            };
        });
}

int main(int argc, char** argv)
{
    Neon::init();
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        std::vector<int> gpu_ids{0};
        auto             runtime = Neon::Runtime::stream;
        Neon::Backend    backend(gpu_ids, runtime);


        const Neon::index_3d grid_dim(256, 256, 1);
        const size_t         num_frames = 500;

        using T = float;
        using Grid = Neon::dGrid;
        Grid grid(
            backend, grid_dim,
            [](const Neon::index_3d& idx) -> bool { return true; },
            createStencil());

        int   cardinality = 1;
        float inactiveValue = 0.0f;
        auto  voxel_1 = grid.template newField<T>("GoL1", cardinality, inactiveValue);
        auto  voxel_2 = grid.template newField<T>("GoL2", cardinality, inactiveValue);

        std::srand(19937);
        voxel_1.forEachActiveCell(
            [](const Neon::index_3d& idx, const int&, T& val) {
                if (idx.z == 0) {
                    val = rand() % 2;
                }
            });
        voxel_2.forEachActiveCell(
            [](const Neon::index_3d&, const int&, T& val) { val = 0; });


        voxel_1.updateDeviceData(0);
        voxel_2.updateDeviceData(0);


        std::vector<Neon::set::Container> containers;
        containers.push_back(GoLContainer(voxel_1, voxel_2, T(grid_dim.x)));
        containers.push_back(GoLContainer(voxel_2, voxel_1, T(grid_dim.x)));

        Neon::skeleton::Skeleton skeleton(backend);
        skeleton.sequence(containers, "GoLSkeleton");


        for (int f = 0; f < num_frames / 2; ++f) {
            skeleton.run();

            backend.syncAll();
            exportVTI(voxel_1, voxel_2, f);
        }
    }

    return 0;
}