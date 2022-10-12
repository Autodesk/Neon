#pragma once

#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "Neon/core/types/Exceptions.h"

namespace {

// Default function that returns an empty string for vertices used in exporting to dot files
std::function<std::string(size_t)> defaultVertexLabelFn = [](size_t) {
    return "";
};

// Default function that returns an empty string for edges used in exporting to dot files
std::function<std::string(const std::pair<size_t, size_t>&)> defaultEdgeLabelFn = [](const std::pair<size_t, size_t>&) {
    return "";
};

}  // namespace

namespace Neon {

// Empty placeholder for vertex and edge properties
struct Empty
{
};

/**
 * DiGraph is a directed graph datastructure implemented using an adjacency list.
 *
 * Self loops can be represented.
 * Parallel loops cannot be represented.
 *
 * @tparam VertexProp Type representing the properties to be stored per-vertex
 * @tparam EdgeProp Type representing the properties to be stored per-edge
 */
template <typename VertexProp = Empty, typename EdgeProp = Empty>
class DiGraph
{
   public:
    using Edge = std::pair<size_t, size_t>;

   private:
    std::map<size_t, std::set<size_t>> mAdj;    // Adjacency list
    std::map<size_t, VertexProp>       mVprop;  // Per-vertex properties
    std::map<Edge, EdgeProp>           mEprop;  // Per-edge properties

    // Helper function that throws an exception when vertex is invalid
    void validateVertex(size_t v) const
    {
        if (hasVertex(v)) {
            return;
        }
        NeonException exception("DiGraph_t::validateVertex");
        exception << "Vertex " + std::to_string(v) + " does not exist.";
        NEON_THROW(exception);
    }

    // Helper function that throws an exception when edge is invalid
    void validateEdge(size_t vi, size_t vj) const
    {
        if (hasEdge(vi, vj)) {
            return;
        }
        NeonException exception("DiGraph_t::validateEdge");
        exception << "Edge (" + std::to_string(vi) + ", " + std::to_string(vj) +
                         ") does not exist.";
        NEON_THROW(exception);
    }

   public:
    /** Virtual destructor */
    ~DiGraph() = default;

    /**
     * Checks whether the vertex is present in the graph
     *
     * @param v Vertex to check
     * @return true If vertex exists
     * @return false If vertex does not exist
     */
    bool hasVertex(size_t v) const
    {
        return mAdj.find(v) != mAdj.end();
    }

    /**
     * Checks whether the edge is present in the graph
     *
     * @param v1 Edge's start vertex
     * @param v2 Edge's end vertex
     * @return true If edge exists
     * @return false If edge does not exist
     */
    bool hasEdge(size_t v1, size_t v2) const
    {
        if (!hasVertex(v1) || !hasVertex(v2)) {
            return false;
        }
        return std::find(mAdj.at(v1).begin(), mAdj.at(v1).end(), v2) != mAdj.at(v1).end();
    }

    /**
     * Checks whether the edge is present in the graph
     *
     * @param edge Edge
     * @return true If edge exists
     * @return false If edge does not exist
     */
    bool hasEdge(const Edge& edge) const
    {
        return hasEdge(edge.first, edge.second);
    }

    /**
     * Adds the vertex to the graph
     *
     * @param v Vertex to add
     * @param prop Vertex property to use for new vertex
     * @return true Successfully added
     * @return false Not added
     */
    bool addVertex(size_t v, VertexProp prop = VertexProp())
    {
        if (hasVertex(v)) {
            return false;
        }
        mAdj.insert({v, std::set<size_t>()});
        mVprop.insert({v, prop});
        return true;
    }

    /**
     * Adds an edge between the two given vertices
     *
     * @param v1 One of the edge's endpoint vertices
     * @param v2 One of the edge's endpoint vertices
     * @param prop Edge property to use for new edge
     * @return true Successfully added
     * @return false Not added
     *
     * NOTE: Throws if vertices do not exist. Doesn't do anything if edge
     * already exists.
     */
    bool addEdge(size_t v1, size_t v2, EdgeProp prop = EdgeProp())
    {
        validateVertex(v1);
        validateVertex(v2);
        if (hasEdge(v1, v2)) {
            return false;
        }
        mAdj[v1].insert(v2);
        setEdgeProperty({v1, v2}, prop);
        return true;
    }

    /**
     * Returns a copy of vertices in the graph
     *
     * @return Copy of all vertices
     */
    std::vector<size_t> vertices() const
    {
        std::vector<size_t> tmp;
        tmp.reserve(numVertices());
        std::transform(mAdj.cbegin(),
                       mAdj.cend(),
                       std::back_inserter(tmp),
                       [](const auto& kv) { return kv.first; });
        return tmp;
    }

    /**
     * Run a function on each vertex in the graph
     *
     * @param fn Function that takes in a vertex and does something
     */
    void forEachVertex(std::function<void(size_t)> fn)
    {
        for (auto& kv : mAdj) {
            fn(kv.first);
        }
    }

    /**
     * Run a function on each vertex in the graph
     *
     * @param fn Function that takes in a vertex and does something
     */
    void forEachVertex(std::function<void(size_t)> fn) const
    {
        for (auto& kv : mAdj) {
            fn(kv.first);
        }
    }

    /**
     * Returns a copy of incoming edges from a vertex as pairs of vertex ids
     *
     * @return Copy of incoming edges from given vertex
     */
    std::set<Edge> inEdges(size_t vertex)
    {
        std::set<Edge> ins;
        forEachInEdge(vertex, [&](const Edge& inEdge) { ins.insert(inEdge); });
        return ins;
    }

    /**
     * Run a function on each incoming edge of a vertex
     *
     * @param vertex Vertex whose incoming edges are considered
     * @param fn Function that takes in an edge and does something
     */
    void forEachInEdge(size_t vertex, std::function<void(const Edge&)> fn)
    {
        for (auto& kv : mAdj) {
            const auto& adjList = kv.second;
            if (adjList.find(vertex) != adjList.end()) {
                fn({kv.first, vertex});
            }
        }
    }

    /**
     * Run a function on each incoming edge of a vertex
     *
     * @param vertex Vertex whose incoming edges are considered
     * @param fn Function that takes in an edge and does something
     */
    void forEachInEdge(size_t vertex, std::function<void(const Edge&)> fn) const
    {
        for (auto& kv : mAdj) {
            const auto& adjList = kv.second;
            if (adjList.find(vertex) != adjList.end()) {
                fn({kv.first, vertex});
            }
        }
    }

    /**
     * Returns a copy of outgoing edges from a vertex as pairs of vertex ids
     *
     * @return Copy of outgoing edges from given vertex
     */
    std::set<Edge> outEdges(size_t vertex)
    {
        std::set<Edge> outs;
        forEachOutEdge(vertex, [&](const Edge& outEdge) { outs.insert(outEdge); });
        return outs;
    }

    /**
     * Returns the number of out edges
     *
     * @return Copy of outgoing edges from given vertex
     */
    auto outEdgesCount(size_t vertex) -> size_t
    {
        size_t count = 0;
        forEachOutEdge(vertex, [&](const Edge&) { count++; });
        return count;
    }

    /**
     * Returns the number of in edges
     *
     * @return Copy of outgoing edges from given vertex
     */
    auto inEdgesCount(size_t vertex) -> size_t
    {
        size_t count = 0;
        forEachInEdge(vertex, [&](const Edge&) { count++; });
        return count;
    }


    /**
     * Run a function on each outgoing edge of a vertex
     *
     * @param vertex Vertex whose outgoing edges are considered
     * @param fn Function that takes in an edge and does something
     */
    void forEachOutEdge(size_t vertex, std::function<void(const Edge&)> fn)
    {
        for (size_t target : mAdj.at(vertex)) {
            fn({vertex, target});
        }
    }

    /**
     * Returns a copy of edges in the graph as pairs of vertex ids
     *
     * @return Copy of all edges
     */
    std::set<Edge> edges() const
    {
        std::set<Edge> edges;
        for (auto i : mAdj) {
            size_t src = i.first;
            for (size_t tgt : i.second) {
                edges.insert({src, tgt});
            }
        }
        return edges;
    }

    /**
     * Run a function on each edge
     *
     * @param fn Function that takes in an edge and does something
     */
    void forEachEdge(std::function<void(const Edge&)> fn)
    {
        for (auto i : mAdj) {
            size_t src = i.first;
            for (size_t tgt : i.second) {
                fn({src, tgt});
            }
        }
    }

    /**
     * Run a function on each edge
     *
     * @param fn Function that takes in an edge and does something
     */
    void forEachEdge(std::function<void(const Edge&)> fn) const
    {
        for (auto i : mAdj) {
            size_t src = i.first;
            for (size_t tgt : i.second) {
                fn({src, tgt});
            }
        }
    }

    /**
     * Get a const-ref to the vertex property
     *
     * @param v Vertex
     * @return Property associated to v
     */
    const VertexProp& getVertexProperty(size_t v) const
    {
        validateVertex(v);
        return mVprop.at(v);
    }

    /**
     * Get a mutable ref to the vertex property
     *
     * @param v Vertex
     * @return Property associated to v
     */
    VertexProp& getVertexProperty(size_t v)
    {
        validateVertex(v);
        return mVprop.at(v);
    }


    /**
     * Set the vertex property
     *
     * @param v Vertex
     * @param Vertex property
     */
    void setVertexProperty(size_t v, const VertexProp& prop)
    {
        validateVertex(v);
        mVprop[v] = prop;
    }

    /**
     * Get the property for the given edge
     *
     * @param edge Edge
     * @return Edge property
     *
     * Throws if edge does not exist
     */
    const EdgeProp& getEdgeProperty(const Edge& edge) const
    {
        return mEprop.at(edge);
    }

    /**
     * Get a mutable ref to the property for the given edge
     *
     * @param edge Edge
     * @return Edge property
     *
     * Throws if edge does not exist
     */
    EdgeProp& getEdgeProperty(const Edge& edge)
    {
        return mEprop.at(edge);
    }

    /**
     * Set the property for the given edge
     *
     * @param edge Edge
     * @param prop Property
     *
     * Throws if edge does not exist
     */
    void setEdgeProperty(const Edge& edge, const EdgeProp& prop)
    {
        validateEdge(edge.first, edge.second);
        mEprop[edge] = prop;
    }

    /**
     * Number of edges in the graph
     *
     * @return number of edges
     */
    size_t numEdges() const
    {
        size_t count = 0;
        for (const auto& kv : mAdj) {
            count += kv.second.size();
        }
        return count;
    }

    /**
     * Number of vertices in the graph
     *
     * @return number of vertices
     */
    size_t numVertices() const
    {
        return mAdj.size();
    }

    /**
     * Contract an edge by deleting the source vertex and connecting all its neighbors
     * with the target vertex. The edge property of the removed vertex is set to all the
     * new edges.
     *
     * a--->c--->d
     *      ^
     *      |
     *      b
     * contractEdge(c, d);
     * a--->d
     *      ^
     *      |
     *      b
     *
     * @param from Source vertex
     * @param to Target vertex
     * @return Vertex property of the removed vertex
     */
    VertexProp contractEdge(size_t from, size_t to)
    {
        validateEdge(from, to);
        VertexProp vprop = getVertexProperty(from);
        EdgeProp   eprop = getEdgeProperty({from, to});
        /*for (size_t endpoint : mAdj[from]) {
            addEdge(endpoint, to, eprop);
        }*/
        forEachInEdge(from, [&](const Edge& inEdge) { addEdge(inEdge.first, to, eprop); });

        removeSelfLoop(to);
        removeVertex(from);
        return vprop;
    }

    /**
     * Contract an edge by deleting the source vertex and connecting all its neighbors
     * with the target vertex The edge property of the removed vertex is set to all the
     * new edges.
     *
     * a--->c--->d
     *      ^
     *      |
     *      b
     * contractEdge(c, d);
     * a--->d
     *      ^
     *      |
     *      b
     *
     * @param edge Edge to contract
     * @return Vertex property of the removed vertex
     */
    VertexProp contractEdge(Edge edge)
    {
        return contractEdge(edge.first, edge.second);
    }

    /**
     * Removes floating vertices that are not connected to any others
     */
    void removeDanglingVertices()
    {
        std::vector<size_t> to_remove;
        forEachVertex([&](size_t v) {
            if (mAdj[v].size() == 0) {
                to_remove.push_back(v);
            }
        });
        for (size_t v : to_remove) {
            removeVertex(v);
        }
    }

    /**
     * Returns a const ref of the set of neighbors of the given vertex
     *
     * @param v Vertex
     * @return Neighboring vertices of v
     */
    const std::set<size_t>& neighbors(size_t v) const
    {
        validateVertex(v);
        return mAdj.at(v);
    }

    /**
     * Returns a set of neighbours from incoming edges
     *
     * @param v Vertex
     * @return Neighboring vertices of v
     */
    auto inNeighbors(size_t v) const
        -> const std::set<size_t>
    {
        std::set<size_t> in;
        forEachInEdge(v, [&](const Edge& e) {
            in.insert(e.first);
        });
        return in;
    }

    /**
     * Returns a set of neighbours from incoming edges
     *
     * @param v Vertex
     * @return Neighboring vertices of v
     */
    auto outNeighbors(size_t v) const -> std::set<size_t>
    {
        std::set<size_t> out;
        forEachOutEdge(v, [&](const Edge& e) {
            out.insert(e.second);
        });
        return out;
    }

    /**
     * Returns a set of neighbours from incoming edges
     *
     * @param v Vertex
     * @return Neighboring vertices of v
     */
    auto outNeighbors(size_t v) -> std::set<size_t>
    {
        std::set<size_t> out;
        forEachOutEdge(v, [&](const Edge& e) {
            out.insert(e.second);
        });
        return out;
    }

    /**
     * Returns a const ref to the internal adjacency list used by the graph
     *
     * @return Adjacency list
     */
    const std::map<size_t, std::set<size_t>>& adj() const
    {
        return mAdj;
    }

    /**
     * Remove the edge
     *
     * @param u Source vertex
     * @param v Target vertex
     *
     * Throws exception if the edge does not exist
     */
    void removeEdge(size_t u, size_t v)
    {
        validateEdge(u, v);
        mAdj[u].erase(v);
        mEprop.erase({u, v});
    }

    /**
     * Remove the edge
     *
     * @param edge Edge to remove
     *
     * Throws exception if the edge does not exist
     */
    void removeEdge(const Edge& edge)
    {
        removeEdge(edge.first, edge.second);
    }

    /**
     * Remove the vertex and connected edges
     *
     * @param v Vertex to remove
     */
    void removeVertex(size_t v)
    {
        validateVertex(v);
        // Remove all outgoing edges
        for (const Edge& outEdge : outEdges(v)) {
            removeEdge(outEdge);
        }
        // Remove all incoming edges
        for (const Edge& inEdge : inEdges(v)) {
            removeEdge(inEdge);
        }
        // Remove vertex
        mAdj.erase(v);
        // Remove vertex property
        mVprop.erase(v);
    }

    /**
     * Removes self edge-loops, if any, from the given vertex
     *
     * @param v Vertex to remove self loops from
     */
    void removeSelfLoop(size_t v)
    {
        validateVertex(v);
        mAdj[v].erase(v);
        mEprop.erase({v, v});
    }

    /**
     * Removes self edge-loops, if any, from all vertices
     */
    void removeSelfLoops()
    {
        for (const auto& kv : mAdj) {
            removeSelfLoop(kv.first);
        }
    }

    /**
     * Clears the graph by deleting all vertices, edges and properties
     */
    void clear()
    {
        mAdj.clear();
        mEprop.clear();
        mVprop.clear();
    }

    /**
     * Exports the graph as a dot file for visualization
     *
     * @param filename Dot file name
     * @param graphName Name of the graph
     * @param vertexLabel Function that takes in a vertex id and returns a label for it
     * @param edgeLabel Function that takes in an edge and returns a label for it
     */
    virtual void exportDotFile(const std::string& filename, std::string graphName = "", std::function<std::string(size_t)> vertexLabel = ::defaultVertexLabelFn, std::function<std::string(const Edge&)> edgeLabel = ::defaultEdgeLabelFn, std::function<std::string(size_t)> vertexLabelProperty = ::defaultVertexLabelFn, std::function<std::string(const Edge&)> edgeLabelProerty = ::defaultEdgeLabelFn)
    {
        std::ofstream out(filename);
        // Start digraph structure
        out << "digraph " << graphName << " {" << std::endl;
        out << "rankdir=LR;" << std::endl;
        forEachVertex([&](size_t v) {
            std::string label = vertexLabel(v);
            std::string property = vertexLabelProperty(v);
            out << v;

            auto h_createLabel = [](std::string l) {
                if (l.empty())
                    return l;
                return std::string("label=\"") + l + "\"";
            };

            auto h_createProperty = [](std::string p, std::string l) {
                if (p.empty())
                    return p;
                if (l.empty())
                    return p;
                return std::string(", ") + p;
            };

            if (!label.empty() || !property.empty()) {
                const auto& labelInfo = h_createLabel(label);
                const auto& propertyInfo = h_createProperty(property, label);
                out << " [ " << labelInfo << propertyInfo << " ]";
            }

            if (!label.empty()) {
                out << " [label=\"" << label << "\"]";
            }
            out << ";" << std::endl;
        });
        forEachEdge([&](const Edge& edge) {
            std::string label = edgeLabel(edge);
            std::string property = edgeLabelProerty(edge);
            out << edge.first << " -> " << edge.second;

            auto h_createLabel = [](std::string l) {
                if (l.empty())
                    return l;
                return std::string("label=\"") + l + "\"";
            };

            auto h_createProperty = [](std::string p, std::string l) {
                if (p.empty())
                    return p;
                if (l.empty())
                    return p;
                return std::string(", ") + p;
            };

            if (!label.empty() || !property.empty()) {
                const auto& labelInfo = h_createLabel(label);
                const auto& propertyInfo = h_createProperty(property, label);
                out << " [ " << labelInfo << propertyInfo << " ]";
            }
            out << ";" << std::endl;
        });
        // End digraph structure
        out << "}" << std::endl;
    }
};

}  // namespace Neon