#include "Neon/core/types/chrono.h"

#include "Neon/set/Containter.h"

#include "Neon/domain/aGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/domain/tools/Geometries.h"
#include "Neon/domain/tools/TestData.h"

#include "Neon/skeleton/Options.h"
#include "Neon/skeleton/Skeleton.h"

#include <cctype>
#include <string>

#include "gtest/gtest.h"
#include "sUt.runHelper.h"
#include "sUt_skeleton.onStream.kernels.h"

using namespace Neon::domain::tool::testing;

static const std::string testFilePrefix("sUt_skeleton_MapStencilDot");

template <typename G, typename T, int C>
void MapStencilDot(TestData<G, T, C>&      data,
                   Neon::skeleton::Occ     occ,
                   Neon::set::TransferMode transfer)
{
    using Type = typename TestData<G, T, C>::Type;

    auto occName = Neon::skeleton::OccUtils::toString(occ);
    occName[0] = toupper(occName[0]);
    const std::string appName(testFilePrefix + "_" + occName);

    const T   scalarVal = 2;
    const int nIterations = 2;

    Neon::skeleton::Skeleton skl(data.getBackend());
    Neon::skeleton::Options  opt(occ, transfer);
    auto                     fR = data.getGrid().template newPatternScalar<T>();

    fR() = scalarVal;

    data.getBackend().syncAll();
    data.resetValuesToRandom(10, 20);

    // skl.ioToDot(appName);

    Neon::Timer_sec timer;
    {  // SKELETON
        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        skl.sequence({UserTools::axpy(fR, Y, X),
                      UserTools::laplace(X, Y),
                      data.getGrid().dot("DotContainer", Y, Y, fR)},
                     appName, opt);

        skl.ioToDot(appName + "_" + Neon::skeleton::OccUtils::toString(opt.occ()));

        timer.start();
        for (int i = 0; i < nIterations; i++) {
            skl.run();
        }
        data.getBackend().syncAll();
        timer.stop();
    }

    Type dR = scalarVal;

    {  // GOLDEN
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);

        for (int i = 0; i < nIterations; i++) {
            data.axpy(&dR, Y, X);
            data.laplace(X, NEON_IO Y);
            data.dot(Y, Y, &dR);
        }
    }

    {  // DEBUG
        data.getIODomain(FieldNames::Y).ioToVti("__getIODomain_Y", "Y");
        data.getField(FieldNames::Y).ioToVtk("__getField_Y", "Y");
    }

    // storage.ioToVti("After");
    bool isOk = data.compare(FieldNames::Y, 0.000001);

    ASSERT_TRUE(isOk);
    ASSERT_NEAR(dR / fR(), 1.0, 0.000001) << "dR " << dR << " fR " << fR();
}

template <typename G, typename T, int C>
void MapStencilDotNoOcc(TestData<G, T, C>& data)
{
    MapStencilDot<G, T, C>(data,
                           Neon::skeleton::Occ::none,
                           Neon::set::TransferMode::get);
}

template <typename G, typename T, int C>
void MapStencilDotOcc(TestData<G, T, C>& data)
{

    MapStencilDot<G, T, C>(data, Neon::skeleton::Occ::standard, Neon::set::TransferMode::get);
}

template <typename G, typename T, int C>
void MapStencilDotExtendedOcc(TestData<G, T, C>& data)
{
    MapStencilDot<G, T, C>(data, Neon::skeleton::Occ::extended, Neon::set::TransferMode::get);
}

template <typename G, typename T, int C>
void MapStencilDotTwoWayExtendedOcc(TestData<G, T, C>& data)
{

    MapStencilDot<G, T, C>(data,
                           Neon::skeleton::Occ::twoWayExtended,
                           Neon::set::TransferMode::get);
}


namespace {
int getNGpus()
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int maxGPUs = Neon::set::DevSet::maxSet().setCardinality();
        if (maxGPUs > 1) {
            return maxGPUs;
        } else {
            return 2;
        }
    } else {
        return 0;
    }
}
}  // namespace

TEST(MapStencilDotNoOcc, bGrid)
{
    int nGpus = 1;
    using Grid = Neon::domain::bGrid;
    using Type = double;
    runAllTestConfiguration<Grid, Type, 0>("bGrid_t", MapStencilDotNoOcc<Grid, Type, 0>, nGpus, 1);
}

TEST(MapStencilDotNoOcc, dGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::domain::dGrid;
    using Type = double;
    runAllTestConfiguration<Grid, Type, 0>("dGrid_t", MapStencilDotNoOcc<Grid, Type, 0>, nGpus, 1);
}

TEST(MapStencilDotOcc, dGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::domain::dGrid;
    using Type = double;
    runAllTestConfiguration<Grid, Type, 0>("dGrid_t", MapStencilDotOcc<Grid, Type, 0>, nGpus, 1);
}

TEST(MapStencilDotExtendedOcc, dGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::domain::dGrid;
    using Type = double;
    runAllTestConfiguration<Grid, Type, 0>("dGrid_t", MapStencilDotExtendedOcc<Grid, Type, 0>, nGpus, 1);
}

TEST(MapStencilDotTwoWayExtendedOcc, dGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::domain::dGrid;
    using Type = double;
    runAllTestConfiguration<Grid, Type, 0>("dGrid_t", MapStencilDotTwoWayExtendedOcc<Grid, Type, 0>, nGpus, 2);
}
