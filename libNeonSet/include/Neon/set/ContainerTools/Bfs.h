#pragma once

#include "Neon/set/ContainerTools/graph/GraphData.h"

namespace Neon::set::container {

struct Graph;

struct Bfs
{
    using Level = std::vector<GraphData::Uid>;
    /**
     * Returns the number of levels
     */
    auto getNumberOfLevels() -> int;

    /**
     * Returns a reference to a level and it's level index
     */
    auto getNewLevel() -> std::pair<Level&, int>;

    /**
     * Returns a reference to a specific level
     */
    auto getLevel(int levelId) const -> const Level&;

    /**
     * Returns a reference to a specific level
     */
    auto getLevel(int levelId) -> Level&;

    /**
     * Returns max level width
     */
    auto getMaxLevelWidth() const -> int;

    /**
     * Returns max level width
     */
    auto getLevelWidth(int levelIdx) const -> int;

    /**
     * For Each iterator (read-only)
     */
    template <typename Fun>
    auto forEachLevel(Fun fun) -> void;

    /**
     * For Each iterator (read-only)
     */
    template <typename Fun>
    auto forEachNodeAtLevel(int levelIdx, const Graph& graph, Fun fun) -> void;

    /**
     * For Each iterator (read-only)
     */
    template <typename Fun>
    auto forEachNodeByLevel(const Graph& graph, Fun fun) -> void;

    /**
     * Clear the BFS status
     */
    auto clear() -> void;

   private:
    std::vector<Level> data;
};

}  // namespace Neon::set::container

#include "Neon/set/ContainerTools/Bfs_imp.h"
