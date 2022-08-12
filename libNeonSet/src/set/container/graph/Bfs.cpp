#include "Neon/set/container/graph/Bfs.h"

namespace Neon::set::container {


auto Bfs::getNumberOfLevels() -> int
{
    return int(data.size());
}

/**
 * Returns a reference to a level and it's level index
 */
auto Bfs::getNewLevel() -> std::pair<std::vector<GraphData::Uid>&, int>
{
    data.push_back(std::vector<GraphData::Uid>());
    int idx = int(data.size()) - 1;
    return {data[idx], idx};
}

/**
 * Returns a reference to a specific level
 */
auto Bfs::getLevel(int levelId) const
    -> const std::vector<GraphData::Uid>&
{
    return data[levelId];
}

/**
 * Returns a reference to a specific level
 */
auto Bfs::getLevel(int levelId) -> std::vector<GraphData::Uid>&
{
    return data[levelId];
}

/**
 * Returns max level width
 */
auto Bfs::getMaxLevelWidth() const -> int
{
    int maxWidth = 0;
    for (const auto& level : data) {
        maxWidth = std::max(maxWidth, int(level.size()));
    }
    return maxWidth;
}

/**
 * Returns max level width
 */
auto Bfs::getLevelWidth(int levelIdx) const
    -> int
{
    return int(data.at(levelIdx).size());
}


}  // namespace Neon::set::container
