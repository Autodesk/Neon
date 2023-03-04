
#include <map>

#include "Neon/core/core.h"
#include "Neon/core/tools/IO.h"

#include "Neon/domain/aGrid.h"
#include "Neon/domain/eGrid.h"

#include "Neon/domain/tools/IOGridVTK.h"

#include "gtest/gtest.h"

template <typename Grid>
auto containersTest(Neon::index_3d& dimension, Neon::Backend& bk) -> void
{
    // Dense data for a pressure field
    int  pCardinality = 1;
    auto pIO = Neon::IODense<int>::makeLinear(1, dimension, pCardinality);

    // Dense data for a velocity
    int  uCardinality = 3;
    auto uIO = Neon::IODense<int>::makeLinear(1, dimension, uCardinality);

    // Neon Grid section
    Grid grid;

    if constexpr (std::is_same_v<Grid, Neon::domain::aGrid>) {
        grid = Grid(bk, dimension);
    } else {
        grid = Grid(
            bk, dimension, [](const Neon::index_3d&) { return true; }, Neon::domain::Stencil::s6_Jacobi_t());
    }
    auto p = grid.template newField<int, 0>("pressure", pCardinality, 0);
    auto u = grid.template newField<int, 0>("velocity", uCardinality, 0);

    ASSERT_ANY_THROW(p.ioFromDense(uIO));
    ASSERT_ANY_THROW(u.ioFromDense(pIO));

    ASSERT_NO_THROW(p.ioFromDense(pIO));
    ASSERT_NO_THROW(u.ioFromDense(uIO));

    pIO.ioVtk("iovtk_Dense_p", "pressure");
    uIO.ioVtk("iovtk_Dense_u", "velocity");

    u.ioToVtk(std::string("iovtk_Field_u") + grid.getImplementationName(), "u");
    p.ioToVtk(std::string("iovtk_Field_p") + grid.getImplementationName(), "p");

    auto iovtk = Neon::domain::IOGridVTK(grid, "iovtk_Grid");
    iovtk.addField(p, "pressure");
    iovtk.addField(u, "pressure");
    iovtk.flushAndClear();
}

TEST(gUt_vtk, aGrid)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        std::vector<int> ids = {0, 0, 0};
        Neon::Backend    bk(ids,
                         Neon::Runtime::stream);

        NEON_INFO(bk.toString());
        Neon::index_3d dimension(10, 1, 1);
        containersTest<Neon::domain::aGrid>(dimension, bk);
    }
}

TEST(gUt_vtk, eGrid)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        std::vector<int> ids = {0, 0, 0};
        Neon::Backend    bk(ids,
                         Neon::Runtime::stream);

        NEON_INFO(bk.toString());
        Neon::index_3d dimension(10, 10, 20);
        containersTest<Neon::domain::details::eGrid::eGrid>(dimension, bk);
    }
}

TEST(gUt_vtk, iovtkCPU)
{
    std::vector<int> ids = {0};
    Neon::Backend    bk(ids,
                        Neon::Runtime::openmp);

    NEON_INFO(bk.toString());
    Neon::index_3d dimension(10, 1, 1);
    containersTest<Neon::domain::aGrid>(dimension, bk);
}
