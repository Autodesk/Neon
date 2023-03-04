#include "gtest/gtest.h"

#include "Neon/Neon.h"

#include "Neon/domain/dGrid.h"
#include "Neon/domain/StaggeredGrid.h"

#include "containers.h"

/**
 * A simple tutorial demonstrating the use of staggered grid in Neon.
 */
int main()
{
    // Selecting the hardware for the computation
    Neon::Backend backend = [] {
        Neon::init();
        // Our XPU will be a CPU device.
        auto runtime = Neon::Runtime::openmp;
        // We are overbooking XPU 0 two times
        std::vector<int> gpu_ids{0, 0};
        Neon::Backend    backend(gpu_ids, runtime);
        return backend;
    }();

    // Define an alias for our staggered grid
    using UniformGrid = Neon::domain::dGrid;
    using StaggeredGrid = Neon::domain::details::experimental::staggeredGrid::StaggeredGrid<UniformGrid>;
    using FP = double;
    using UserContainers = tools::Containers<StaggeredGrid, FP>;

    // Initializing a staggered grid based on dGrid
    StaggeredGrid grid = [&] {
        // Setting the dimension for the voxel grid
        Neon::int32_3d voxDims{2, 3, 9};

        // For our tutorial, we don't need to add new stencil.
        // By default, the staggered grid will add the stencil
        // to move from voxel to node and vice versa
        std::vector<Neon::domain::Stencil> noExtraStencils;

        StaggeredGrid newGrid(
            backend,
            voxDims,
            [](const Neon::index_3d&) -> bool {
                return true;
            },
            noExtraStencils);

        return newGrid;
    }();

    auto density = [&grid] {
        // Defining a voxel field to represent the density of each voxel
        // We then set its initial values to zero in the host (CPU),
        // and finally we transfer the data to the XPU
        auto density = grid.template newVoxelField<FP, 1>("Density", 1, 0);
        density.forEachActiveCell([](const Neon::index_3d& /*idx*/,
                                     const int& /*cardinality*/,
                                     FP& value) {
            value = 0;
        });
        density.updateCompute(Neon::Backend::mainStreamIdx);
        return density;
    }();

    auto temperature = [&grid] {
        // Defining a node field to represent the temperature at each node
        // We then set its initial values to one, but to make it more interesting,
        // we do the initialization directly on the device calling a Containers.
        // Note that in this case we don't need to transfer data from host to XPUs
        auto temperature = grid.template newNodeField<FP, 1>("Temperature", 1, 0);
        temperature.forEachActiveCell([](const Neon::index_3d& /*idx*/,
                                         const int& /*cardinality*/,
                                         FP& value) {
            value = 0;
        });
        UserContainers::resetValue(temperature, FP(1.0)).run(Neon::Backend::mainStreamIdx);
        return temperature;
    }();

    // We define a simple function to export to vtk both
    // temperature and density field during each step of the tutorial.
    auto exportingToVti = [&](const std::string& tutorialStepId) {
        {  // Moving memory from XPUs to CPU
            temperature.updateIO(Neon::Backend::mainStreamIdx);
            density.updateIO(Neon::Backend::mainStreamIdx);
            backend.sync(Neon::Backend::mainStreamIdx);
        }
        {  // Exporting the results
            const std::string appName("staggered-grid");
            temperature.ioToVtk(appName + "-temperature-" + tutorialStepId, "temperature", false, Neon::IoFileType::BINARY);
            density.ioToVtk(appName + "-density-" + tutorialStepId, "density");
        }
    };

    // We are exporting to vtk the values of the fields after the initialization.
    // We expect all temperature nodes to be set to one,
    // and all density voxels to be set to zero.
    exportingToVti("0000");

    {  // As the previous container changes the values of the voxel field,
       // a halo update must be called before the next Container that uses
       // the temperature as input for a stencil operation.
       //
       // Because we wrote this tutorial only using the Domain level, we have to
       // execute the halo update manually.
       //
       // However, when using the Neon skeleton level halo update are
       // automatically managed.
        Neon::set::HuOptions huOptions(Neon::set::TransferMode::get,
                                       true,
                                       Neon::Backend::mainStreamIdx);
        temperature.haloUpdate(huOptions);
    }

    {  // Accessing voxels from nodes
       // For each node we loop around active voxel, and we sum their values.
       // At the end of this operation we expect all voxel to store a value of 8,
       // as there are 8 nodes for each voxel, each one set to one.
        UserContainers::sumNodesOnVoxels(density, temperature)
            .run(Neon::Backend::mainStreamIdx);

        exportingToVti("0001");
    }

    {  // The previous comment on halo updates is applicable here.
        Neon::set::HuOptions huOptions(Neon::set::TransferMode::get,
                                       true,
                                       Neon::Backend::mainStreamIdx);
        density.haloUpdate(huOptions);
    }
    {  // Accessing nodes from voxels
       // For each voxel we loop around all nodes, we sum their values and divide the result by 8.
       // At the end of the container we expect to have the following values:
       // -- 1 for any corner node
       // -- 2 for any node on an edge of our domain
       // -- 3 for any node on a face
       // -- 4 for internal nodes
        UserContainers::sumVoxelsOnNodesAndDivideBy8(temperature,
                                                     density)
            .run(Neon::Backend::mainStreamIdx);

        exportingToVti("0002");
    }
    return 0;
}
