#include "Neon/set/container/graph/GraphDependency.h"
#include "Neon/set/container/Graph.h"

#include <string>

namespace Neon::set::container {

GraphDependency::
    GraphDependency()
{
}

auto GraphDependency::
    setType(GraphDependencyType   type,
            const RawGraph::Edge& edge)
        -> void
{
    mType = type;
    mEdge = edge;
}

auto GraphDependency::
    getType()
        const -> GraphDependencyType
{
    return mType;
}
GraphDependency::
    GraphDependency(GraphDependencyType   type,
                    const RawGraph::Edge& edge)
{
    setType(type, edge);
}

GraphDependency::
    GraphDependency(const dataDependency::Token& token,
                    const RawGraph::Edge&        edge)
{
    setType(GraphDependencyType::data, edge);
    mTokens.push_back(token);
}

GraphDependency::
    GraphDependency(const std::vector<Neon::set::dataDependency::Token>& tokens,
                    const RawGraph::Edge&                                edge)
{
    setType(GraphDependencyType::data, edge);
    mTokens = tokens;
}

auto GraphDependency::
    getLabel()
        const -> std::string
{
    return GraphDependencyTypeUtil::toString(getType());
}

auto GraphDependency::
    addToken(const Token& token)
        -> void
{
    mTokens.push_back(token);
}


auto GraphDependency::getTokens()
    const -> const Tokens&
{
    return mTokens;
}

auto GraphDependency::hasStencilDependency()
    const -> bool
{
    bool isStencil = std::any_of(mTokens.begin(),
                                 mTokens.end(),
                                 [](const auto& token) {
                                     return token.compute() == Neon::Pattern::STENCIL;
                                 });
    return isStencil;
}

auto GraphDependency::getRawEdge()
    const -> RawGraph::Edge
{
    return mEdge;
}

auto GraphDependency::getSourceNode(const Graph& graph)
    const -> const GraphNode&
{
    return graph.helpGetGraphNode(mEdge.first);
}

auto GraphDependency::getDestinationNode(const Graph& graph)
    const -> const GraphNode&
{
    return graph.helpGetGraphNode(mEdge.second);
}


}  // namespace Neon::set::container
