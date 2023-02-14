
#include <map>

#include "Neon/core/core.h"
#include "Neon/core/tools/IO.h"

#include "Neon/domain/aGrid.h"
#include "Neon/domain/tools/IOGridVTK.h"

#include "gtest/gtest.h"

#include <iostream>


template <typename Field>
auto myHostSumContainer(const typename Field::Type& val,
                        Field&                      fC) -> Neon::set::Container
{
    const Neon::domain::aGrid& grid = fC.getGrid();
    auto                       container = grid.template getContainer(
        "MyHostKernel",
        [&, val](Neon::set::Loader& loader) {
            auto& lc = loader.load(fC);
            return [=] (const typename Field::Cell& e) mutable {
                for (int i = 0; i < lc.cardinality(); i++) {
                    lc(e, i) += (val +i);
                }
            }; });
    return container;
}

template <typename Grid>
auto containersTest([[maybe_unused]] std::string prefix, Neon::index_3d& dimension, Neon::Backend& bk) -> void
{
    // Dense data for a pressure field
    int  pCardinality = 1;
    auto pIO = Neon::IODense<int>::makeLinear(1, dimension, pCardinality);

    // Dense data for a velocity
    int  uCardinality = 3;
    auto uIO = Neon::IODense<int>::makeLinear(1, dimension, uCardinality);

    // Neon Grid section
    Grid grid(bk, dimension);
    auto p = grid.template newField<int, 0>("pressure", pCardinality, 0);
    NEON_INFO(p.toString());

    auto u = grid.template newField<int, 0>("velocity", uCardinality, 0);
    NEON_INFO(u.toString());

    {  // loading data
        p.ioFromDense(pIO);
        p.updateCompute(0);

        u.ioFromDense(uIO);
        u.updateCompute(0);

        bk.sync();
    }

    {  // Compute
        myAddContainer(21, p).run(0);
        myAddContainer(33, u).run(0);

        pIO.template forEach([](const Neon::index_3d&, int card, int& val) {
            val += 21 + card;
        });

        uIO.template forEach([](const Neon::index_3d&, int card, int& val) {
            val += 33 + card;
        });
    }

    {
        p.updateIO(0);
        u.updateIO(0);

        bk.sync();
    }

    auto pResIO = p.ioToDense();
    ASSERT_EQ(std::get<0>(pIO.maxDiff(pResIO, pIO)), 0);

    auto uResIO = u.ioToDense();
    ASSERT_EQ(std::get<0>(pIO.maxDiff(uResIO, uIO)), 0);

#if 0
    pIO.template ioToVti(prefix + "_Dense_p", "pressure");
    uIO.template ioToVti(prefix + "_Dense_u", "velocity");

    u.ioToVtk(prefix + "_Field_u" + grid.getGridImplementationName(), "u");
    p.ioToVtk(prefix + "_Field_p" + grid.getGridImplementationName(), "p");

    auto iovtk = Neon::domain::IOGridVTK(grid, prefix + "_Grid");
    iovtk.addField(p, "pressure");
    iovtk.addField(u, "velocity");
    iovtk.flushAndClear();
#endif
}


TEST(gUt, ContainerStream)
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        using Grid = Neon::domain::aGrid;
        std::vector<int> ids = {0, 0, 0, 0};
        Neon::Backend    bk(ids,
                            Neon::Runtime::stream);
        NEON_INFO(bk.toString());
        Neon::index_3d dimension(5013, 1, 1);
        containersTest<Grid>("gUt_ContainerStream_aGrid", dimension, bk);
    }
}

TEST(gUt, ContainerOpenmp_aGrid)
{
    std::vector<int> ids = {0, 0};
    Neon::Backend    bk(ids,
                        Neon::Runtime::openmp);

    NEON_INFO(bk.toString());
    Neon::index_3d dimension(100, 1, 1);
    containersTest<Neon::domain::aGrid>("gUt_ContainerOpenmp_aGrid", dimension, bk);
}