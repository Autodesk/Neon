![](img/04-layers-domain.png){ align=right  style="width:200px"}
# The Domain Level

Neon Domain level's goal is to provide users with simple mechanisms for some specific domains. 
Currently, Neon focus on those domains where a regular cartesian discretisations are leveraged. 
Using a simple example will look ad how the level mechanisms can be used. 

## Working with dense domains

Let's consider a simple example: a dense discrete domain where we implicitly want to represent a sphere through its signed distance function.
We'll be looking at how to implement the following in Neon:
Define a dense domain

- Allocate a field over the dense domain to store the sphere signed distance function (**Neon grid**)
- Export the signed distance field to vtk for visualization (**Neon field**)
- Compute the gradient of the sphere's signed distance function (**Neon Container over fields**)
- Export the gradient field to vtk for visualization

### Choosing a backend

```cpp linenums="2" 
int main(int, char**)
{
    Neon::init();
    
    // User define parameters for the problem
    const int32_t    n = 25;
    const double     voxelEdge = 1.0;
    Neon::index_3d   dim(n, n, n);
    const double     r = (n * voxelEdge / 2) * .8;
    
    // Defining the resources where to run the application
    std::vector<int> gpu_ids{0};
    auto runtime = Neon::Runtime::stream;
    // auto runtime = Neon::Runtime::openmp;
    Neon::Backend backend(gpu_ids, runtime);
```
### Grid allocation

```cpp linenums="17" 
    // Defining the grid
    using Grid = Neon::domain::eGrid;
    std::vector<Neon::index_3d> points{{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    Neon::domain::Stencil       gradStencil(points);

    Grid grid(
        backend /** Passing the target system for the computation */,
        dim /** Dimension of the regular grid used for the discretizasion */,
        [&](const Neon::index_3d&) -> bool {
            return true;
        } /** Implicit representation that identifies the interesting points in the grid */,
        gradStencil);

    /** Exporting information of the grid and the active points to a vtk file */
    grid.ioDomainToVtk("domain");
```
### Neon Fields
```cpp linenums="17" 
    /** Creating a scalar field over the grid.
     * Non active voxels will get be associated with a default value of -100 */
    auto sphereSdf = grid.newField<double>("sdf" /** Given name of the field */,
                                     1 /** Number of field's component per grid point */,
                                     -100 /** Default value for non active points */);

    /** Using the signed distance function of a sphere to initialize the field's values */
    sphereSdf.forEachActiveCell([&](const Neon::index_3d& idx, int, double& value) {
        double sdf = sdfCenteredSphere(idx, dim, voxelEdge, r);
        value = sdf;
    });
    
    sphereSdf.ioToVtk("sdf", "sdf");
```
## Neon Containers on Grids