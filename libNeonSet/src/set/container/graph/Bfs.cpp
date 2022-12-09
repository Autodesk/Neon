#include "Neon/set/container/graph/Bfs.h"
#include <sstream>

namespace Neon::set::container {


auto Bfs::getNumberOfLevels() -> int
{
    return int(data.size());
}

auto Bfs::getNewLevel() -> std::pair<std::vector<GraphData::Uid>&, int>
{
    data.push_back(std::vector<GraphData::Uid>());
    int idx = int(data.size()) - 1;
    return {data[idx], idx};
}

auto Bfs::getLevel(int levelId) const
    -> const std::vector<GraphData::Uid>&
{
    return data[levelId];
}

auto Bfs::getLevel(int levelId) -> std::vector<GraphData::Uid>&
{
    return data[levelId];
}

auto Bfs::getMaxLevelWidth() const -> int
{
    int maxWidth = 0;
    for (const auto& level : data) {
        maxWidth = std::max(maxWidth, int(level.size()));
    }
    return maxWidth;
}

auto Bfs::getLevelWidth(int levelIdx) const
    -> int
{
    return int(data.at(levelIdx).size());
}

auto Bfs::clear() -> void
{
    data.clear();
}

auto Bfs::toString() -> std::string
{
    std::stringstream s;
    s << "Num levels "<<data.size();
    int i=0;
    for(const auto& level : data){
        s <<"\t\tLevel "<<i<<" (";
        for(const auto& uid : level){
            s << uid << " ";
        }
        s<<")"<<std::endl;
        i++;
    }
    return s.str();
}

}  // namespace Neon::set::container
