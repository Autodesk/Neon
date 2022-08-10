#pragma once

#include "Neon/set/ContainerTools/Bfs.h"
#include "Neon/set/ContainerTools/Graph.h"

namespace Neon::set::container {

template <typename Fun>
auto Bfs::forEachLevel(Fun fun) -> void
{
    int levelIdx = 0;
    for (auto& level : data) {
        fun(level, levelIdx);
        levelIdx++;
    }
}

template <typename Fun>
auto Bfs::forEachNodeAtLevel(int levelIdx, const Graph& graph, Fun fun) -> void
{
    for (auto& nodeIdx : data.at(levelIdx)) {
        const auto& node = graph.helpGetGraphNode(nodeIdx);
        fun(node);
    }
}

auto Bfs::clear() -> void
{
    data.clear();
}
template <typename Fun>
auto Bfs::forEachNodeByLevel(const Graph& graph, Fun fun) -> void
{
    forEachLevel([&graph, &fun](const Level& level, int levelIdx) {
        for (const auto& nodeIdx : level) {
            const auto& node = graph.helpGetGraphNode(nodeIdx);
            fun(node, levelIdx);
        }
    });
}


}  // namespace Neon::set::container
