#include "Neon/skeleton/internal/multiGpuGraph/Edge.h"

#include <memory>
namespace Neon {
namespace skeleton {
namespace internal {


std::atomic<uint64_t> edgeCounter_g{0};

Edge::Edge(size_t             edgeId,
               const DataToken_t& dataToken,
               Dependencies_e     dependency,
               bool               haloUp)
    : m_edgeId(edgeId)
{
    m_dependencies.emplace_back(dataToken, dependency, haloUp);
}

Edge::Edge(size_t edgeId,
               bool   isSchedulingEdge)
    : m_edgeId(edgeId), m_isSchedulingEdge(isSchedulingEdge)
{
    if(isSchedulingEdge == false){
        NEON_THROW_UNSUPPORTED_OPTION("");
    }
}

auto Edge::factory(const DataToken_t& dataToken,
                     Dependencies_e     type,
                     bool               haloUp) -> Edge
{
    size_t id = (edgeCounter_g++);
    Edge   edge(id, dataToken, type, haloUp);
    return edge;
}

auto Edge::factorySchedulingEdge() -> Edge
{
    size_t id = (edgeCounter_g++);
    Edge   edge(id, true);
    return edge;
}

auto Edge::edgeId() const -> size_t
{
    return m_edgeId;
}
auto Edge::nDependencies() const -> size_t
{
    return m_dependencies.size();
}

auto Edge::info() const -> const std::vector<Info>&
{
    return m_dependencies;
}
auto Edge::infoMutable() -> std::vector<Info>&
{
    return m_dependencies;
}

auto Edge::append(const DataToken_t& dataToken,
                    Dependencies_e     dType) -> void
{
    m_dependencies.push_back({dataToken, dType});
}
auto Edge::toString() const -> std::string
{
    std::string intro = "Dependencies: \\l";  // + std::to_string(nDependencies());
    for (auto& i : info()) {
        intro = intro + i.toString();
    }
    return intro;
}
auto Edge::clone() const -> Edge
{
    size_t id = (edgeCounter_g++);
    Edge   clone(*this);
    clone.m_edgeId = id;
    return clone;
}

}  // namespace internal
}  // namespace skeleton
}  // namespace Neon