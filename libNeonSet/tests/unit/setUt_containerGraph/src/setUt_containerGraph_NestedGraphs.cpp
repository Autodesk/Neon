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
#define DEBUG_MODE 0

using namespace Neon::domain::tool::testing;
static const std::string testFilePrefix("setUt_containerGraph_nestedGraph");

template <typename G, typename T, int C>
void NestedGraphsTest(TestData<G, T, C>& data)
{
    using Type = typename TestData<G, T, C>::Type;

    const std::string appName(testFilePrefix);

    const Type scalarVal = 2;

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
            Neon::set::container::Graph graph(data.getBackend());

            auto nodeA = graph.addNode(UserTools::axpy(fR, W, X, name + "-StageA"));
            auto nodeC = graph.addNode(UserTools::axpy(fR, W, Y, name + "-StageC"));
            graph.addNodeInBetween(nodeA, UserTools::axpy(fR, W, Z, name + "-StageB"), nodeC);

#if (DEBUG_MODE == 1)
            graph.ioToDot(appName, "UserGraph", false);
            graph.ioToDot(appName + "-debug", "UserGraph", true);
#endif

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
            auto fname = appName + "_withSubGraph";
            graph.runtimePreSet(0);
            graph.ioToDot(fname, "UserGraph", false);
            graph.ioToDot(fname + "-debug", "UserGraph", true);
        }

        {
            auto fname = appName + "_expandSubGraphs";

            graph.expandSubGraphs();
            graph.runtimePreSet(0);
            graph.ioToDot(fname, "UserGraph", false);
            graph.ioToDot(fname + "-debug", "UserGraph", true);
        }

        ASSERT_EQ(graph.getNumberOfNodes(), 9);
    }
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
    using Grid = Neon::domain::details::eGrid::eGrid;
    using Type = int32_t;
    runOneTestConfiguration<Grid, Type, 0>("eGrid_t", NestedGraphs<Grid, Type, 0>, 1, 1);
}

#undef DEBUG_MODE