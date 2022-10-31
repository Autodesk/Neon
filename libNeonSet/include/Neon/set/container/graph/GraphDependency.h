#pragma once

#include "Neon/set/container/graph/GraphNode.h"
#include "Neon/set/dependency/Token.h"

#include "GraphDependencyType.h"

namespace Neon::skeleton::internal {
struct MultiXpuGraph;
}

namespace Neon::set::container {

struct GraphNode;
struct Graph;

struct GraphDependency
{
    using Tokens = std::vector<Neon::set::dataDependency::Token>;
    using Token = Neon::set::dataDependency::Token;
    using RawGraph = DiGraph<Neon::set::container::GraphNode, GraphDependency>;

    friend Neon::skeleton::internal::MultiXpuGraph;
    friend Graph;

   public:
    GraphDependency();

    explicit GraphDependency(GraphDependencyType   type,
                             const RawGraph::Edge& edge);

    explicit GraphDependency(const Token&          type,
                             const RawGraph::Edge& edge);

    explicit GraphDependency(const Tokens&         type,
                             const RawGraph::Edge& edge);

    auto setType(GraphDependencyType   type,
                 const RawGraph::Edge& edge)
        -> void;

    auto getType()
        const -> GraphDependencyType;

    auto getLabel() const
        -> std::string;

    auto addToken(const Neon::set::dataDependency::Token& token)
        -> void;

    auto getSourceNode(const Graph& graph)
        const -> const GraphNode&;

    auto getDestinationNode(const Graph& graph)
        const -> const GraphNode&;
    /**
     * Returns the tokens generating this dependency
     */
    auto getTokens()
        const -> const Tokens&;

    auto hasStencilDependency()
        const -> bool;

   private:
    auto getRawEdge()
        const -> RawGraph::Edge;

    GraphDependencyType mType;
    Tokens              mTokens /**< Tokens creating this dependency. */;
    RawGraph::Edge      mEdge;
    // TODO - add information for data and Scheduling dependency
};

}  // namespace Neon::set::container
