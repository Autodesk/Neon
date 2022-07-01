
#include <Neon/core/types/digraph.h>

#include <cstring>
#include <iostream>

#include "Neon/core/core.h"
#include "gtest/gtest.h"

// Custom vertex property
struct MyVertexProp
{
    std::string name = "";
};
// Custom edge property
struct MyEdgeProp
{
    std::string name = "";
};

TEST(DiGraph, AddRemove)
{
    Neon::DiGraph G;
    G.addVertex(0);
    G.addVertex(1);
    ASSERT_EQ(G.numVertices(), 2);
    G.addEdge(0, 1);
    ASSERT_EQ(G.numEdges(), 1);
    G.removeVertex(1);
    ASSERT_EQ(G.numVertices(), 1);
    ASSERT_EQ(G.numEdges(), 0);
    ASSERT_TRUE(G.hasVertex(0));
    ASSERT_FALSE(G.hasVertex(1));
    G.clear();
    ASSERT_EQ(G.numVertices(), 0);
}

TEST(DiGraph, Properties)
{
    Neon::DiGraph<MyVertexProp, MyEdgeProp> G;

    // Check if vertex property was constructed properly
    G.addVertex(0, {"FirstVertex"});
    G.addVertex(1, {"SecondVertex"});
    ASSERT_EQ(G.getVertexProperty(0).name, "FirstVertex");
    ASSERT_EQ(G.getVertexProperty(1).name, "SecondVertex");

    // Check if vertex property can be changed
    G.getVertexProperty(0).name = "VertexOne";
    G.getVertexProperty(1).name = "VertexTwo";
    ASSERT_EQ(G.getVertexProperty(0).name, "VertexOne");
    ASSERT_EQ(G.getVertexProperty(1).name, "VertexTwo");

    // Check if vertex property can be set
    G.setVertexProperty(0, {"FirstVertex"});
    G.setVertexProperty(1, {"SecondVertex"});
    ASSERT_EQ(G.getVertexProperty(0).name, "FirstVertex");
    ASSERT_EQ(G.getVertexProperty(1).name, "SecondVertex");

    // Check if edge property was constructed properly
    G.addEdge(0, 1, {"FirstEdge"});
    ASSERT_EQ(G.getEdgeProperty({0, 1}).name, "FirstEdge");

    // Check if edge property can be changed
    G.getEdgeProperty({0, 1}).name = "EdgeOne";
    ASSERT_EQ(G.getEdgeProperty({0, 1}).name, "EdgeOne");

    // Check if edge property can be set
    G.setEdgeProperty({0, 1}, {"FirstEdge"});
    ASSERT_EQ(G.getEdgeProperty({0, 1}).name, "FirstEdge");
}

TEST(DiGraph, DanglingVertices)
{
    Neon::DiGraph G;
    G.addVertex(0);
    G.addVertex(1);
    ASSERT_EQ(G.numVertices(), 2);
    G.removeDanglingVertices();
    ASSERT_EQ(G.numVertices(), 0);
}

TEST(DiGraph, ContractEdge)
{
    Neon::DiGraph<MyVertexProp, MyEdgeProp> G;
    G.addVertex(0, {"a"});
    G.addVertex(1, {"b"});
    G.addVertex(2, {"c"});
    G.addVertex(3, {"d"});
    G.addEdge(0, 2, {"a->c"});
    G.addEdge(1, 2, {"b->c"});
    G.addEdge(2, 3, {"c->d"});
    ASSERT_EQ(G.numVertices(), 4);
    ASSERT_EQ(G.numEdges(), 3);
    /*
    a--->c--->d
         ^
         |
         b
    */
    G.contractEdge({2, 3});
    /*
    a--->d
         ^
         |
         b
    */
    ASSERT_EQ(G.numVertices(), 3);                      // One vertex should have been removed
    ASSERT_EQ(G.numEdges(), 2);                         // One edge should have been removed
    ASSERT_FALSE(G.hasVertex(2));                       // Vertex 2 ("c") must have beeen deleted
    ASSERT_TRUE(G.hasEdge({0, 3}));                     // a->d must now exist
    ASSERT_TRUE(G.hasEdge({1, 3}));                     // b->d must now exist
    ASSERT_EQ(G.getEdgeProperty({0, 3}).name, "c->d");  // a->d should have name "c->d"
    ASSERT_EQ(G.getEdgeProperty({1, 3}).name, "c->d");  // b->d should have name "c->d"
}

TEST(DiGraph, IncomingOutgoing)
{
    Neon::DiGraph G;
    G.addVertex(0);
    G.addVertex(1);
    G.addVertex(2);
    G.addVertex(3);
    G.addVertex(4);
    G.addEdge(0, 2);
    G.addEdge(1, 2);
    G.addEdge(2, 3);
    G.addEdge(2, 4);
    /*
    0     1
     \   /
      v v
       2 
     /   \
    v     v
    3     4
    */
    auto inEdges = G.inEdges(2);
    ASSERT_TRUE(inEdges.find({0, 2}) != inEdges.end());
    ASSERT_TRUE(inEdges.find({1, 2}) != inEdges.end());
    auto outEdges = G.outEdges(2);
    ASSERT_TRUE(outEdges.find({2, 3}) != outEdges.end());
    ASSERT_TRUE(outEdges.find({2, 4}) != outEdges.end());
}

TEST(DiGraph, ExportDotFile)
{
    Neon::DiGraph<MyVertexProp, MyEdgeProp> G;
    G.addVertex(0, {"Zero"});
    G.addVertex(1, {"One"});
    G.addVertex(2, {"Two"});
    G.addVertex(3, {"Three"});
    G.addVertex(4, {"Four"});
    G.addEdge(0, 2, {"0->2"});
    G.addEdge(1, 2, {"1->2"});
    G.addEdge(2, 3, {"2->3"});
    G.addEdge(2, 4, {"2->4"});

    G.exportDotFile("graph_nolabels.dot", "X");
    auto vertexLabel = [&](size_t v) {
        return G.getVertexProperty(v).name;
    };
    auto edgeLabel = [&](const std::pair<size_t, size_t>& edge) {
        return G.getEdgeProperty(edge).name;
    };
    G.exportDotFile("graph_labels.dot", "X", vertexLabel, edgeLabel);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
