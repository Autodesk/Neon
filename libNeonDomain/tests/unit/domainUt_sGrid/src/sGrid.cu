#include "Neon/core/types/chrono.h"

#include "Neon/set/Containter.h"

#include "Neon/Neon.h"
#include "Neon/domain/aGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
//#include "Neon/domain/sGrid.h"
#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/Skeleton.h"

#include "Neon/domain/tools/Geometries.h"
#include "Neon/domain/tools/TestData.h"

#include <cctype>
#include <string>

#include "gtest/gtest.h"
#include "sGridRunHelper.h"

#include "Neon/domain/sGrid.h"
using namespace Neon::domain::tool::testing;
static const std::string testFilePrefix("domainUt_sGrid");
namespace help {

template <typename F, typename FS>
auto copyFromOuter(const F& Fouter,
                   FS&      FSGrid)
    -> Neon::set::Container
{
    return FSGrid.getGrid().getContainer(
        "SET_FROM_OUTER_CONTAINER",
        [&](Neon::set::Loader& loader) {
            const auto fouter = loader.load(Fouter);
            auto       fsGrid = loader.load(FSGrid);

            return [=] NEON_CUDA_HOST_DEVICE(const typename decltype(fsGrid)::Cell& cell) mutable {
                auto outerCell = fsGrid.mapToOuterGrid(cell);
                auto outerVal = fouter(outerCell, 0);
                fsGrid(cell, 0) = outerVal;
            };
        });
};

template <typename FS>
auto scale(typename FS::Type const& scaleVal,
           const FS&                Fin,
           FS&                      Fout)
    -> Neon::set::Container
{
    return Fin.getGrid().getContainer(
        "SET_FROM_OUTER_CONTAINER",
        [&](Neon::set::Loader& loader) {
            const auto fin = loader.load(Fin);
            auto       fout = loader.load(Fout);

            return [=] NEON_CUDA_HOST_DEVICE(const typename decltype(fin)::Cell& cell) mutable {
                fout(cell, 0) = scaleVal * fin(cell, 0);
            };
        });
};

template <typename F, typename FS>
auto copyToOuter(FS const& FSGrid,
                 F&        Fouter)
    -> Neon::set::Container
{
    return FSGrid.getGrid().getContainer(
        "SET_FROM_OUTER_CONTAINER",
        [&](Neon::set::Loader& loader) {
            auto       fouter = loader.load(Fouter);
            const auto fsGrid = loader.load(FSGrid);

            return [=] NEON_CUDA_HOST_DEVICE(const typename decltype(fsGrid)::Cell& cell) mutable {
                auto outerCell = fsGrid.mapToOuterGrid(cell);
                auto outerVal = fsGrid(cell, 0);
                fouter(outerCell, 0) = outerVal;
            };
        });
};
/**
 * Map-Stencil-Map test
 */
template <typename G, typename T, int C>
void sGridTestContainerRun(TestData<G, T, C>& data)
{
    /*
     * Grid structure:
     * - (x,y,z) in sGrid if X(x,y,z) %2 ==0
     * Computations on Neon
     * a. sX = X
     * b. sY = 2* sX
     * c. Y = sY
     *
     * Computation on Golden reference
     * - if X(x,y,z) %2, Y(x,y,z) = 2*X(x,y,z)
     * - else Y(x,y,z) =  Y_{t0}(x,y,z)
     *
     * Check
     * Y
     */
    using Type = typename TestData<G, T, C>::Type;
    auto& grid = data.getGrid();

    NEON_INFO(grid.toString());

    const std::string appName(testFilePrefix + "_" + grid.getImplementationName());
    data.resetValuesToLinear(1, 100);

    {  // NEON
        const index_3d        dim = grid.getDimension();
        std::vector<index_3d> elements;

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        dim.forEach([&](int x, int y, int z) {
#pragma omp critial
            {
                index_3d newE(x, y, z);
                if (X.isInsideDomain(newE)) {
                    if (X(newE, 0) % 2 == 0) {
                        elements.push_back(newE);
                    }
                }
            }
        });

        Neon::domain::sGrid<G> sGrid(grid, elements);
        auto                   sX = sGrid.template newField<int>("sX", 1, 11);
        auto                   sY = sGrid.template newField<int>("sY", 1, 11);

        copyFromOuter(X, sX).run(0);
        scale(2, sX, sY).run(0);
        copyToOuter(sY, Y).run(0);

        data.getBackend().sync(0);
        Y.updateIO(0);
    }


    //    {  // SKELETON
    //        auto& X = data.getField(FieldNames::X);
    //        auto& Y = data.getField(FieldNames::Y);
    //
    //        std::vector<Neon::set::Container> ops{
    //            UserTools::axpy(fR, Y, X),
    //            UserTools::laplace(X, Y),
    //            UserTools::axpy(fR, Y, Y)};
    //
    //        skl.sequence(ops, appName, opt);
    //
    //        skl.ioToDot(appName + "_" + opt.toString(opt.occ()));
    //
    //        timer.start();
    //        for (int i = 0; i < nIterations; i++) {
    //            skl.run();
    //        }
    //        data.getBackend().syncAll();
    //        timer.stop();
    //    }
    //
    {  // Golden data

        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);

        data.forEachActiveIODomain([&](const Neon::index_3d& idx,
                                       int                   cardinality,
                                       Type&                 a,
                                       Type&                 b) {
            if (a % 2 == 0) {
                b = 2 * a;
            }
        },
                                   X, Y);
    }
    // storage.ioToVti("After");
    {  // DEBUG
        data.getIODomain(FieldNames::Y).ioToVti("getIODomain_Y", "Y");
        data.getField(FieldNames::Y).ioToVtk("getField_Y", "Y");
    }


    bool isOk = data.compare(FieldNames::Y);
    isOk = isOk && data.compare(FieldNames::X);

    ASSERT_TRUE(isOk);
    //    std::cout<<"Done"<<std::endl;
}

template <typename G, typename T, int C>
void sGridTestSkeleton(TestData<G, T, C>& data)
{
    /*
     * Grid structure:
     * - (x,y,z) in sGrid if X(x,y,z) %2 ==0
     * Computations on Neon
     * a. sX = X
     * b. sY = 2* sX
     * c. Y = sY
     *
     * Computation on Golden reference
     * - if X(x,y,z) %2, Y(x,y,z) = 2*X(x,y,z)
     * - else Y(x,y,z) =  Y_{t0}(x,y,z)
     *
     * Check
     * Y
     */
    using Type = typename TestData<G, T, C>::Type;
    auto& grid = data.getGrid();

    NEON_INFO(grid.toString());

    const std::string appName(testFilePrefix + "_" + grid.getImplementationName());
    data.resetValuesToLinear(1, 100);

    {  // NEON
        const index_3d        dim = grid.getDimension();
        std::vector<index_3d> elements;

        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        dim.template forEach<Neon::computeMode_t::seq>([&](int x, int y, int z) {
#pragma omp critial
            {
                index_3d newE(x, y, z);
                if (X.isInsideDomain(newE)) {
                    if (X(newE, 0) % 2 == 0) {
                        elements.push_back(newE);
                    }
                }
            }
        });

        Neon::domain::sGrid<G> sGrid(grid, elements);
        auto                   sX = sGrid.template newField<int>("sX", 1, 11);
        auto                   sY = sGrid.template newField<int>("sY", 1, 11);

        //        std::vector<Neon::set::Container> ops{
        //            UserTools::axpy(fR, Y, X),
        //            UserTools::laplace(X, Y),
        //            UserTools::axpy(fR, Y, Y)};
        //
        //        skl.sequence(ops, appName, opt);

        Neon::skeleton::Skeleton skl(data.getBackend());
        Neon::skeleton::Options  opt(Neon::skeleton::Occ::none,
                                     Neon::set::TransferMode::get);


        std::vector<Neon::set::Container> ops{copyFromOuter(X, sX),
                                              scale(2, sX, sY),
                                              copyToOuter(sY, Y)};
        skl.sequence(ops, appName, opt);
        skl.run();

        data.getBackend().sync(0);
        Y.updateIO(0);
    }


    //    {  // SKELETON
    //        auto& X = data.getField(FieldNames::X);
    //        auto& Y = data.getField(FieldNames::Y);
    //
    //        std::vector<Neon::set::Container> ops{
    //            UserTools::axpy(fR, Y, X),
    //            UserTools::laplace(X, Y),
    //            UserTools::axpy(fR, Y, Y)};
    //
    //        skl.sequence(ops, appName, opt);
    //
    //        skl.ioToDot(appName + "_" + opt.toString(opt.occ()));
    //
    //        timer.start();
    //        for (int i = 0; i < nIterations; i++) {
    //            skl.run();
    //        }
    //        data.getBackend().syncAll();
    //        timer.stop();
    //    }
    //
    {  // Golden data

        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);

        data.forEachActiveIODomain([&](const Neon::index_3d& idx,
                                       int                   cardinality,
                                       Type&                 a,
                                       Type&                 b) {
            if (a % 2 == 0) {
                b = 2 * a;
            }
        },
                                   X, Y);
    }
    // storage.ioToVti("After");
    {  // DEBUG
        data.getIODomain(FieldNames::Y).ioToVti("getIODomain_Y", "Y");
        data.getField(FieldNames::Y).ioToVtk("getField_Y", "Y");
    }


    bool isOk = data.compare(FieldNames::Y);
    isOk = isOk && data.compare(FieldNames::X);

    ASSERT_TRUE(isOk);
    //    std::cout<<"Done"<<std::endl;
}
}  // namespace help

template <typename G, typename T, int C>
void sGridTest(TestData<G, T, C>& data)
{
    help::sGridTestContainerRun<G, T, C>(data);
}

namespace {
int getNGpus()
{
    int maxGPUs = Neon::set::DevSet::maxSet().setCardinality();
    if (maxGPUs > 1) {
        return maxGPUs;
    } else {
        return 3;
    }
}
}  // namespace

TEST(domainUnitTests, sGrid_eGrid)
{
    Neon::init();
    int nGpus = getNGpus();
    NEON_INFO("sGrid_eGrid");
    using Grid = Neon::domain::eGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("sGrid", help::sGridTestContainerRun<Grid, Type, 0>, nGpus, 1);
}

TEST(domainUnitTests, sGrid_eGrid_skeleton)
{
    Neon::init();
    int nGpus = getNGpus();
    NEON_INFO("sGrid_eGrid");
    using Grid = Neon::domain::eGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("sGrid", help::sGridTestSkeleton<Grid, Type, 0>, nGpus, 1);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Neon::init();
    return RUN_ALL_TESTS();
}
