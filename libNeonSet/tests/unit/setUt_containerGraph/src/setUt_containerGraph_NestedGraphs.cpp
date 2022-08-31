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
static const std::string testFilePrefix("setUt_containerGraph_nestedGraph");

template <typename G, typename T, int C>
void NestedGraphsTest(TestData<G, T, C>& data)
{
    using Type = typename TestData<G, T, C>::Type;

    const std::string appName(testFilePrefix);

    const Type scalarVal = 2;
    const int  nIterations = 10;

    auto fR = data.getGrid().template newPatternScalar<Type>();
    fR() = scalarVal;
    data.getBackend().syncAll();

    data.resetValuesToRandom(1, 50);
    Neon::Timer_sec timer;

    {  // NEON
        auto& X = data.getField(FieldNames::X);

        auto& Y = data.getField(FieldNames::Y);
        auto& Z = data.getField(FieldNames::Z);

        auto& W = data.getField(FieldNames::W);

        auto generateInnerGraph = [&](std::string name) -> Neon::set::Container {
            Neon::set::container::Graph graph;

            auto nodeA = graph.addNode(UserTools::axpy(fR, W, X, name+"-StageA"));
            auto nodeC = graph.addNode(UserTools::axpy(fR, W, Y, name+"-StageC"));
            graph.addNodeInBetween(nodeA, UserTools::axpy(fR, W, Z, name+"-StageB"), nodeC);

//            graph.ioToDot(appName, "UserGraph", false);
//            graph.ioToDot(appName + "-debug", "UserGraph", true);

            auto container = Neon::set::Container::factoryGraph(name, graph, [](Neon::SetIdx, Neon::set::Loader&) {});
            return container;
        };

        Neon::set::container::Graph graph(data.getBackend());
        auto                        nodeA = generateInnerGraph("GraphK");
        auto                        nodeB = generateInnerGraph("GraphL");
        auto                        nodeC = generateInnerGraph("GraphM");

        graph.addNode(nodeA);
        graph.addNode(nodeB);
        graph.addNode(nodeC);

        {
            auto fname  = appName + "_withSubGraph";
            graph.runtimePreSet(0);
            graph.ioToDot(fname, "UserGraph", false);
            graph.ioToDot(fname + "-debug", "UserGraph", true);
        }

        {
            auto fname  = appName + "_expandSubGraphs";

            graph.expandSubGraphs();
            graph.runtimePreSet(0);
            graph.ioToDot(fname, "UserGraph", false);
            graph.ioToDot(fname + "-debug", "UserGraph", true);
        }

        //        timer.start();
        //        for (int i = 0; i < nIterations; i++) {
        //            skl.run();
        //        }
        //        data.getBackend().syncAll();
        //        timer.stop();
    }

    {  // Golden data
        //auto time = timer.time();

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

    /*{  // DEBUG
        data.getIODomain(FieldNames::X).ioToVti("IODomain_X", "X");
        data.getField(FieldNames::X).ioToVtk("Field_X", "X");

        data.getIODomain(FieldNames::Y).ioToVti("IODomain_Y", "Y");
        data.getField(FieldNames::Y).ioToVtk("Field_Y", "Y");
    }*/

    ASSERT_TRUE(isOk);
}

template <typename G, typename T, int C>
void NestedGraphs(TestData<G, T, C>& data)
{
    NestedGraphsTest<G, T, C>(data);
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

TEST(NestedGraphs, eGrid)
{
    using Grid = Neon::domain::internal::eGrid::eGrid;
    using Type = int32_t;
    runOneTestConfiguration<Grid, Type, 0>("eGrid_t", NestedGraphs<Grid, Type, 0>, 1, 1);
}
