#include <sstream>

#include "Neon/Neon.h"
#include "Neon/domain/eGrid.h"
#include "Neon/skeleton/Skeleton.h"

#include "expandSphere.h"
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
    Neon::init();
    // auto runtime = Neon::Runtime::openmp;
    auto runtime = Neon::Runtime::stream;
    // We are overbooking GPU 0 three times
    std::vector<int> gpu_ids{0, 0, 0};
    Neon::Backend    backend(gpu_ids, runtime);
    // Printing some information
    NEON_INFO(backend.toString());

    // Step 2 -> Neon grid: setting up a dense cartesian domain
    const int32_t  n = 25;
    Neon::index_3d dim(n, n, n);     // Size of the domain
    const double   voxelEdge = 1.0;  // Size of a voxel edge

    using Grid = Neon::dGrid;  // Selecting one of the grid provided by Neon
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

    // Step 3 -> Neon field: initializing a sphere through its signed distance function

    // Creating a scalar field over the grid.
    // Inactive cells will get associated with a default value of -100 */
    auto sphereSdf = grid.newField<double>("sphereSdf",  // <- Given name of the field.
                                           1,            // <- Number of field's component per grid point.
                                           -100);        // <- Default value for non active points.

    const double r = (n * voxelEdge / 2) * .3;

    // Using the signed distance function of a sphere to initialize the field's values
    // We leverage the forEachActiveCell method to easily iterate over the active cells.
    sphereSdf.forEachActiveCell([&](const Neon::index_3d& idx, int, double& value) {
        double sdf = sdfCenteredSphere(idx, dim, voxelEdge, r);
        value = sdf;
    });

    // Exporting some information of the sdf on terminal and on a vtk file.
    NEON_INFO(sphereSdf.toString());
    sphereSdf.ioToVtk("sdf", "sdf");

    // Step 4 -> Neon map containers: expanding the sphere via a level set

    // loading the sphereSdf to device
    sphereSdf.updateDeviceData(Neon::Backend::mainStreamIdx);

    // Run a container that ads a value to the sphere sdf
    // The result is a level set of an expanded sphere (not more a sdf)
    // We run the container asynchronously on the main stream
    expandedLevelSet(sphereSdf, 5.0).run(Neon::Backend::mainStreamIdx);

    // Moving asynchronously the values of the newly computed level set back
    // to export the result to vtk.
    sphereSdf.updateHostData(Neon::Backend::mainStreamIdx);

    // Waiting for the transfer to complete.
    backend.sync(Neon::Backend::mainStreamIdx);

    // Exporting once again the fiel to vtk
    sphereSdf.ioToVtk("expandedLevelSet", "expandedLevelSet");

    // Step 5 -> Neon stencil containers: computing the grad of the level set field

    auto grad = grid.newField<double>("sphereSdf" , // <- Given name of the field.
                                      3, // <- Number of field's component per grid point.
                                      0); // <- Default value for non active points.

    sphereSdf.newHaloUpdate(Neon::set::StencilSemantic::standard,
                            Neon::set::TransferMode::get,
                            Neon::Execution::device).run(Neon::Backend::mainStreamIdx);

    computeGrad(sphereSdf, grad, voxelEdge).run(Neon::Backend::mainStreamIdx);
    grad.updateHostData(Neon::Backend::mainStreamIdx);
    backend.sync(Neon::Backend::mainStreamIdx);

    grad.ioToVtk("grad", "grad");

    return 0;
}