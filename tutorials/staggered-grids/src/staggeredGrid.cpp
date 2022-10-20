#include "gtest/gtest.h"

#include "Neon/Neon.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/internal/experimental/staggeredGrid/StaggeredGrid.h"

#include "containers.h"

int main()
{
    // Step 0 -> initialize Neon runtime
    Neon::init();

    // Step 1 -> Neon backend: choosing the hardware for the computation
    Neon::Backend backend = [] {
        // auto runtime = Neon::Runtime::openmp;
        auto runtime = Neon::Runtime::openmp;
        // We are overbooking GPU 0 three times
        std::vector<int> gpu_ids{0,0};
        Neon::Backend    backend(gpu_ids, runtime);

        // Printing some information
        NEON_INFO(backend.toString());
        return backend;
    }();

    // Define an alias for our staggered grid
    using UniformGrid = Neon::domain::dGrid;
    using StaggeredGrid = Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<UniformGrid>;
    using FP = double;

    StaggeredGrid grid = [&] {
        // Setting the dimension for the voxel grid
        Neon::int32_3d                     voxDims{2, 3, 9};

        // For our example we don't need to add new stencil.
        // By default the staggered grid will add the stencil
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

    // Defining a voxel field to represent the density of each voxel
    auto density = grid.template newVoxelField<FP, 1>("Density", 1, 0);
    density.forEachActiveCell([](const Neon::index_3d& /*idx*/,
                                 const int& /*cardinality*/,
                                 FP& value) {
        value = 1;
    });

    // Defining a node field to represent the temperature at each node
    auto temperature = grid.template newNodeField<FP, 1>("Temperature", 1, 0);
    temperature.forEachActiveCell([](const Neon::index_3d& /*idx*/,
                                 const int& /*cardinality*/,
                                 FP& value) {
        value = 0;
    });

    // Exporting the initial values of the fields
    // For the temperature field we choose to use binary format for vti file.
    const std::string appName("staggered-grid");
    temperature.ioToVtk(appName + "-temperature_binary_0000", "temperature", false, Neon::IoFileType::BINARY);
    density.ioToVtk(appName + "-density_0000", "density");

    {  // Moving memory from CPU to the GPUs
        temperature.updateCompute(Neon::Backend::mainStreamIdx);
        density.updateCompute(Neon::Backend::mainStreamIdx);
    }


    { // Accessing voxels from nodes
        {
            // We use halo update as we are working only at the domain level.
            // When leveraging the skeleton abstractions, hlo update are automatically
            // handled by Neon.
            Neon::set::HuOptions huOptions(Neon::set::TransferMode::get,
                                           true,
                                           Neon::Backend::mainStreamIdx);
            density.haloUpdate(huOptions);
        }

        // For each nodes we loop around active voxel and we sum their values.
       Containers<StaggeredGrid, FP>::sumVoxelsOnNodes(temperature, density).run(Neon::Backend::mainStreamIdx);

        temperature.updateIO(Neon::Backend::mainStreamIdx);
        density.updateIO(Neon::Backend::mainStreamIdx);
        backend.sync(Neon::Backend::mainStreamIdx);

        // Exporting the results
        temperature.ioToVtk(appName + "-temperature_binary_0001", "temperature", false, Neon::IoFileType::BINARY);
        density.ioToVtk(appName + "-density_0001", "density");
    }

    { // Accessing nodes from voxels
        Neon::set::HuOptions huOptions(Neon::set::TransferMode::get,
                                       true,
                                       Neon::Backend::mainStreamIdx);

        // We reset all node value to 1.
        Containers<StaggeredGrid, FP>::resetValue(temperature, FP(1.0)).run(Neon::Backend::mainStreamIdx);
        temperature.haloUpdate(huOptions);

        // For each voxel we loop around all nodes and we sum their values.
        // Note that all nodes of a voxel are always active.
        Containers<StaggeredGrid, FP>::sumNodesOnVoxels(density, temperature).run(Neon::Backend::mainStreamIdx);

        temperature.updateIO(Neon::Backend::mainStreamIdx);
        density.updateIO(Neon::Backend::mainStreamIdx);
        backend.sync(Neon::Backend::mainStreamIdx);

        // Exporting the results
        temperature.ioToVtk(appName + "-temperature_binary_0002", "temperature", false, Neon::IoFileType::BINARY);
        density.ioToVtk(appName + "-density_0002", "density");
    }
    return 0;
}
