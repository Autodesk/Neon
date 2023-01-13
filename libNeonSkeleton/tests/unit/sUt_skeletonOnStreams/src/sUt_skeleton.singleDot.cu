#include "Neon/core/types/chrono.h"
#include "Neon/domain/aGrid.h"
#include "Neon/domain/bGrid.h"
#include "Neon/domain/dGrid.h"
#include "Neon/domain/eGrid.h"
#include "Neon/set/Containter.h"
#include "Neon/skeleton/Skeleton.h"
#include "gtest/gtest.h"
#include "sUt.runHelper.h"
#include "sUt_skeleton.onStream.kernels.h"

using namespace Neon::domain::tool::testing;
static const std::string testFilePrefix("sUt_skeleton_MapStencilMap");

namespace help {
template <typename G, typename T, int C>
void SingleDot(TestData<G, T, C>&      data,
               Neon::skeleton::Occ     occ,
               Neon::set::TransferMode transfer)
{
    using Type = typename TestData<G, T, C>::Type;

    auto occName = Neon::skeleton::OccUtils::toString(occ);
    occName[0] = toupper(occName[0]);
    const std::string appName(testFilePrefix + "_" + occName);

    Neon::skeleton::Skeleton skl(data.getBackend());
    Neon::skeleton::Options  opt(occ, transfer);

    const Type scalarVal = 0;

    auto fR = data.getGrid().template newPatternScalar<Type>();
    fR() = scalarVal;
    data.getBackend().syncAll();
    data.resetValuesToRandom(1, 50);


    {  // SKELETON
        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        std::vector<Neon::set::Container> sVec{
            X.getGrid().dot("DotContainer", X, X, fR)};

        sVec[0].run(0, Neon::DataView::STANDARD);
    }

    Type dR = scalarVal;
    {  // Golden data
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);

        data.dot(X, X, &dR);
    }

    data.getBackend().syncAll();

    data.getIODomain(FieldNames::Y).ioToVti("getIODomain_Y", "Y");
    data.getField(FieldNames::Y).ioToVtk("getField_Y", "Y");

    ASSERT_NEAR(dR / fR(), 1.0, 0.000001) << "No match between " << dR << " and " << fR();
}

template <typename G, typename T, int C>
void SingleStencilTestData(TestData<G, T, C>&      data,
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

    data.resetValuesToRandom(1, 50);
    Neon::Timer_sec timer;

    {  // SKELETON
        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);

        std::vector<Neon::set::Container> ops{UserTools::laplace(X, Y),
                                              UserTools::laplace(Y, X)};

        skl.sequence(ops, appName, opt);
        skl.ioToDot(appName + "_" + Neon::skeleton::OccUtils::toString(opt.occ()), "", true);

        timer.start();
        for (int i = 0; i < nIterations; i++) {
            skl.run();
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
            data.laplace(X, NEON_IO Y);
            data.laplace(Y, NEON_IO X);
        }
    }
#if 0
    {  // DEBUG
        data.getIODomain(FieldNames::Y).ioToVti("getIODomain_Y", "Y");
        data.getField(FieldNames::Y).ioToVtk("getField_Y", "Y");
    }
#endif

    bool isOk = data.compare(FieldNames::Y);
    isOk = isOk && data.compare(FieldNames::X);

    ASSERT_TRUE(isOk);
}
}  // namespace help

template <typename G, typename T, int C>
void runSingleDot(TestData<G, T, C>& data)
{
    help::SingleDot<G, T, C>(data, Neon::skeleton::Occ::none, Neon::set::TransferMode::get);
}

template <typename G, typename T, int C>
void runSingleStencilTestData(TestData<G, T, C>& data)
{
    std::cout << "Executing Occ::none" << std::endl;
    help::SingleStencilTestData<G, T, C>(data, Neon::skeleton::Occ::none, Neon::set::TransferMode::get);
    std::cout << "Executing Occ::standard" << std::endl;
    help::SingleStencilTestData<G, T, C>(data, Neon::skeleton::Occ::standard, Neon::set::TransferMode::get);
}

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

TEST(SingleDot, dGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::domain::dGrid;
    using Type = double;
    runAllTestConfiguration<Grid, Type, 0>("dGrid", runSingleDot<Grid, Type, 0>, nGpus, 1);
}

TEST(SingleStencilTestData, dGrid)
{
    int nGpus = getNGpus();
    using Grid = Neon::domain::dGrid;
    using Type = double;
    runAllTestConfiguration<Grid, Type, 0>("dGrid", runSingleStencilTestData<Grid, Type, 0>, nGpus, 1);
}


TEST(SingleDot, bGrid)
{
    int nGpus = 1;
    using Grid = Neon::domain::bGrid;
    using Type = double;
    runAllTestConfiguration<Grid, Type, 0>("bGrid", runSingleDot<Grid, Type, 0>, nGpus, 1);
}

TEST(SingleStencilTestData, bGrid)
{
    int nGpus = 1;
    using Grid = Neon::domain::bGrid;
    using Type = double;
    runAllTestConfiguration<Grid, Type, 0>("bGrid", runSingleStencilTestData<Grid, Type, 0>, nGpus, 1);
}