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
#include "setUt_containerGraph_kernels.h"
#include "setUt_containerGraph_runHelper.h"

#include "Neon/set/container/Graph.h"

using namespace Neon::domain::tool::testing;
static const std::string testFilePrefix("setUt_containerGraph");

template <typename G, typename T, int C>
void ThreeIndependentMapsTest(TestData<G, T, C>& data)
{
    using Type = typename TestData<G, T, C>::Type;

    const std::string appName(testFilePrefix);

    const Type scalarVal = 2;
    const int  nIterations = 10;

    auto fR = data.getGrid().template newPatternScalar<Type>();
    fR() = scalarVal;
    data.getBackend().syncAll();

    //data.resetValuesToRandom(1, 50);
    data.resetValuesToConst(1, 50);
    Neon::Timer_sec timer;

    {  // NEON
        auto& X = data.getField(FieldNames::X);
        auto& Y = data.getField(FieldNames::Y);
        auto& Z = data.getField(FieldNames::Z);
        auto& W = data.getField(FieldNames::W);

        Neon::set::container::Graph graph(data.getBackend());

        graph.addNode(UserTools::axpy(fR, W, X, "nodeA"));
        graph.addNode(UserTools::axpy(fR, W, Y, "nodeB"));
        graph.addNode(UserTools::axpy(fR, W, Z, "nodeC"));

        graph.ioToDot(appName, "UserGraph", false);
        graph.ioToDot(appName + "-debug", "UserGraph", true);

        Neon::set::Container exec = Neon::set::Container::factoryGraph("Test", graph, [&](Neon::SetIdx,
                                                                                          Neon::set::Loader& loader) {
            loader.load(X);
            loader.load(Y);
            loader.load(Z);
        });

        timer.start();
        for (int i = 0; i < nIterations; i++) {
            exec.run(0);
        }
        data.getBackend().syncAll();
        timer.stop();
    }

    {  // Golden data

        Type  dR = scalarVal;
        auto& X = data.getIODomain(FieldNames::X);
        auto& Y = data.getIODomain(FieldNames::Y);
        auto& Z = data.getIODomain(FieldNames::Z);
        auto& W = data.getIODomain(FieldNames::W);

        for (int i = 0; i < nIterations; i++) {
            data.axpy(&dR, W, X);
            data.axpy(&dR, W, Y);
            data.axpy(&dR, W, Z);
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
void ThreeIndependentMaps(TestData<G, T, C>& data)
{
    ThreeIndependentMapsTest<G, T, C>(data);
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

TEST(ThreeIndependentMaps, eGrid)
{
    using Grid = Neon::domain::internal::eGrid::eGrid;
    using Type = int32_t;
    runOneTestConfiguration<Grid, Type, 0>("eGrid_t", ThreeIndependentMaps<Grid, Type, 0>, 1, 1);
}
