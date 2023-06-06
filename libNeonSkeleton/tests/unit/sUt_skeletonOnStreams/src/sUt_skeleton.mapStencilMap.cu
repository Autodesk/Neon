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
static const std::string testFilePrefix("sUt_skeleton_MapStencilMap");

template <typename G, typename T, int C>
void MapStencilMap(TestData<G, T, C>&      data,
                   Neon::skeleton::Occ     occ,
                   Neon::set::TransferMode transfer)
{
    using Type = typename TestData<G, T, C>::Type;

    auto occName = Neon::skeleton::OccUtils::toString(occ);
    occName[0] = toupper(occName[0]);
    const std::string appName(testFilePrefix + "_" + occName);

    Neon::skeleton::Skeleton skl(data.getBackend());
    Neon::skeleton::Options  opt(occ, transfer);

    const Type scalarVal = 2;
    const int  nIterations = 10;

    auto fR = data.getGrid().template newPatternScalar<Type>();
    fR() = scalarVal;
    data.getBackend().syncAll();

    //data.resetValuesToRandom(1, 50);
    data.resetValuesToMasked(1,1,3);
    Neon::Timer_sec timer;

    {  // SKELETON
        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        std::vector<Neon::set::Container> ops{
            UserTools::axpy(fR, Y, X),
            UserTools::laplace(X, Y),
            UserTools::axpy(fR, Y, Y)};

        skl.sequence(ops, appName, opt);

        skl.ioToDot(appName + "_" + Neon::skeleton::OccUtils::toString(opt.occ()),
                    "",
                    true);

        timer.start();
        for (int i = 0; i < nIterations; i++) {
            skl.run();
            data.getBackend().syncAll();
        }
        data.getBackend().syncAll();
        timer.stop();
    }

    {  // Golden data
        auto time = timer.time();

        Type  dR = scalarVal;
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);

        for (int i = 0; i < nIterations; i++) {
            data.axpy(&dR, Y, X);
            data.laplace(X, Y);
            data.axpy(&dR, Y, Y);
        }
    }
    bool isOk = data.compare(FieldNames::X);
    isOk = isOk && data.compare(FieldNames::Y);

    {  // DEBUG
        data.getIODomain(FieldNames::X).ioToVti("IODomain_X", "X");
        data.getField(FieldNames::X).ioToVtk("Field_X", "X");

        data.getIODomain(FieldNames::Y).ioToVti("IODomain_Y", "Y");
        data.getField(FieldNames::Y).ioToVtk("Field_Y", "Y");
    }

    ASSERT_TRUE(isOk);
}

template <typename G, typename T, int C>
void MapStencilOCC(TestData<G, T, C>& data)
{
    MapStencilMap<G, T, C>(data, Neon::skeleton::Occ::standard, Neon::set::TransferMode::get);
}

template <typename G, typename T, int C>
void MapStencilExtendedOCC(TestData<G, T, C>& data)
{
    MapStencilMap<G, T, C>(data, Neon::skeleton::Occ::extended, Neon::set::TransferMode::get);
}

template <typename G, typename T, int C>
void MapStencilTwoWayExtendedOCC(TestData<G, T, C>& data)
{
    MapStencilMap<G, T, C>(data, Neon::skeleton::Occ::twoWayExtended, Neon::set::TransferMode::get);
}

template <typename G, typename T, int C>
void MapStencilNoOCC(TestData<G, T, C>& data)
{
    MapStencilMap<G, T, C>(data, Neon::skeleton::Occ::none, Neon::set::TransferMode::get);
}

namespace {
int getNGpus()
{
    if (Neon::sys::globalSpace::gpuSysObjStorage.numDevs() > 0) {
        int maxGPUs = Neon::set::DevSet::maxSet().setCardinality();
        if (maxGPUs > 1) {
            return maxGPUs;
        } else {
            return 3;
        }
    } else {
        return 0;
    }
}
}  // namespace


TEST(MapStencilMap_NoOCC, dGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::dGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("dGrid", MapStencilNoOCC<Grid, Type, 0>, nGpus, 2);
}

TEST(MapStencilMap_NoOCC, eGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::domain::details::eGrid::eGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("eGrid_t", MapStencilNoOCC<Grid, Type, 0>, nGpus, 1);
}
#if 0



TEST(MapStencilMap_OCC, eGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::domain::details::eGrid::eGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("eGrid_t", MapStencilOCC<Grid, Type, 0>, nGpus, 1);
}

TEST(MapStencilMap_OCC, dGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::dGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("dGrid", MapStencilOCC<Grid, Type, 0>, nGpus, 2);
}

TEST(MapStencilMap_ExtendedOCC, eGrid)
{
    int nGpus = getNGpus();
    NEON_INFO("MapStencilMap_ExtendedOCC");
    using Grid = Neon::domain::details::eGrid::eGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("eGrid_t", MapStencilExtendedOCC<Grid, Type, 0>, nGpus, 1);
}

TEST(MapStencilMap_ExtendedOCC, dGrid)
{
    int nGpus = getNGpus();
    NEON_INFO("MapStencilMap_ExtendedOCC");
    using Grid = Neon::dGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("dGrid", MapStencilExtendedOCC<Grid, Type, 0>, nGpus, 1);
}

TEST(MapStencilMap_TwoWayExtendedOCC, eGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::domain::details::eGrid::eGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("eGrid_t", MapStencilTwoWayExtendedOCC<Grid, Type, 0>, nGpus, 1);
}

TEST(MapStencilMap_TwoWayExtendedOCC, dGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::dGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("dGrid", MapStencilTwoWayExtendedOCC<Grid, Type, 0>, nGpus, 2);
}

TEST(MapStencilMap_NoOCC, bGrid)
{
    int nGpus = 1;
    using Grid = Neon::domain::details::bGrid::bGrid;
    using Type = int32_t;
    runAllTestConfiguration<Grid, Type, 0>("bGrid_t", MapStencilNoOCC<Grid, Type, 0>, nGpus, 1);
}

#endif