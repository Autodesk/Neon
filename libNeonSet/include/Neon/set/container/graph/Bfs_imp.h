#pragma once

#include "Neon/set/container/Graph.h"

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
auto Bfs::forEachNodeAtLevel(int levelIdx, Neon::set::container::Graph& graph, Fun fun) -> void
{
    for (auto& nodeIdx : data.at(levelIdx)) {
        auto& node = graph.helpGetGraphNode(nodeIdx);
        fun(node);
    }
}


template <typename Fun>
auto Bfs::forEachNodeByLevel(Graph& graph, Fun fun) -> void
{
    forEachLevel([&graph, &fun](const Level& level, int levelIdx) {
        for (const auto& nodeIdx : level) {
            auto& node = graph.helpGetGraphNode(nodeIdx);
            fun(node, levelIdx);
        }
    });
}


}  // namespace Neon::set::container
