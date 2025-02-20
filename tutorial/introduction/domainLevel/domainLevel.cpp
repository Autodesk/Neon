#include <sstream>

#include "Neon/Neon.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"

#include "expandLevelSet.h"
#include "grad.h"

auto sdfCenteredSphere(Neon::int32_3d idx /**< queried location*/,
                       Neon::index_3d dim,
                       double         voxelEdge,
                       double         r)
    -> double
{
    auto sphereCenter = (dim.newType<double>() * voxelEdge) / 2.0;
    auto queryPoint = idx.newType<double>() * voxelEdge;
    auto diff = queryPoint - sphereCenter;
    // distance from sphere origin
    double d = std::pow(diff.x, 2) +
               std::pow(diff.y, 2) +
               std::pow(diff.z, 2);
    return std::sqrt(d) - r;
}

int main(int, char**)
{
    // Step 1 -> Neon backend: choosing the hardware for the computation
    Neon::Backend backend = [] {
        Neon::init();
        // auto runtime = Neon::Runtime::openmp;
        auto runtime = Neon::Runtime::stream;
        // We are overbooking XPU 0 three times
        std::vector<int> xpuIds{0, 0, 0};
        Neon::Backend    backend(xpuIds, runtime);
        // Printing some information
        NEON_INFO(backend.toString());
        return backend;
    }();

    // Step 2 -> Neon grid: setting up a 100^3 dense cartesian domain
    const int32_t  n = 100;
    Neon::index_3d dim(n, n, n);     // Size of the domain
    const double   voxelEdge = 1.0;  // Size of a voxel edge

    using Grid = Neon::domain::eGrid;  // Selecting one of the grid provided by Neon
    Grid grid = [&] {
        Neon::domain::Stencil gradStencil([] {
            // We use a center difference scheme to compute the grad
            // The order of the points is important,
            // as we'll leverage the specific order when computing the grad.
            // First positive direction on x, y and z,
            // then negative direction on x, y, z respectively.
            return std::vector<Neon::index_3d>{
                {1, 0, 0},
                {0, 1, 0},
                {0, 0, 1},
                {-1, 0, 0},
                {0, -1, 0},
                {0, 0, -1}};
        }());

        // Actual Neon grid allocation
        Grid grid(
            backend,  // <- Passing the target hardware for the computation
            dim,      // <- Dimension of the regular grid used for the discretizasion.
            [&](const Neon::index_3d&) -> bool {
                // We are looking for a dense domain,
                // so we are interested in all the points in the grid.
                return true;
            },             // <-  defining the active cells.
            gradStencil);  // <- Stencil that will be used during computations on the grid

        // Exporting some information
        NEON_INFO(grid.toString());
        grid.ioDomainToVtk("domain");

        return grid;
    }();


    // Step 3 -> Neon field: initializing a sphere through its signed distance function

    auto sphere = [dim, n, voxelEdge, &grid] {
        // Creating a scalar field over the grid.
        // Inactive cells will get associated with a default value of -100 */
        auto sphere = grid.newField<double>("sphere",  // <- Given name of the field.
                                            1,         // <- Number of field's component per grid point.
                                            -100);     // <- Default value for non active points.

        const double r = (n * voxelEdge / 2) * .5;

        // We initialize the field with the level set of a sphere.
        // We leverage the forEachActiveCell method to easily iterate over the active cells.
        sphere.forEachActiveCell([&](const Neon::index_3d& idx, int, double& value) {
            double sdf = sdfCenteredSphere(idx, dim, voxelEdge, r);
            value = sdf;
        });
        return sphere;
    }();

    // Exporting some information of the level set field on terminal and on a vtk file.
    NEON_INFO(sphere.toString());
    sphere.ioToVtk("sphere-levelSet", "levelSet");

    // Step 4 -> Neon map containers: expanding the sphere via a level set

    {  // loading the sphere to XPUs
        sphere.updateCompute(Neon::Backend::mainStreamIdx);
    }
    // Run a container that adds a value to the sdf sphere.
    // The result is a level set of an expanded sphere (not more a sdf).
    // We run the container asynchronously on the main stream
    expandLevelSet(sphere, 9.0).run(Neon::Backend::mainStreamIdx);

    {  // Moving asynchronously the values of the newly computed
        // level set values to the host
        sphere.updateIO(Neon::Backend::mainStreamIdx);
        // Waiting for the transfer to complete.
        backend.sync(Neon::Backend::mainStreamIdx);
        // Exporting once again the field to vtk
        sphere.ioToVtk("extended-sphere-levelSet", "levelSet");
    }

    // Step 5 -> Neon stencil containers: computing the grad of the level set field
    auto grad = grid.newField<double>("grad",  // <- Given name of the field.
                                      3,       // <- Number of field's component per grid point.
                                      0);      // <- Default value for non active points.

    Neon::set::HuOptions huOptions(Neon::set::TransferMode::get,
                                   true);
    sphere.haloUpdate(huOptions);

    // Execution of a container that computes the gradient of the sphere
    computeGrad(sphere, grad, voxelEdge).run(Neon::Backend::mainStreamIdx);

    {  // Moving the grad data onto the host and exporting it to vtk
        grad.updateIO(Neon::Backend::mainStreamIdx);
        backend.sync(Neon::Backend::mainStreamIdx);
        grad.ioToVtk("extended-sphere-grad", "grad");
    }

    return 0;
}