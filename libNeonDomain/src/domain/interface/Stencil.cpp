#include "Neon/domain/interface/Stencil.h"
#include <unordered_set>


namespace Neon::domain {

Stencil::Stencil(std::vector<Neon::index_3d> const& points,
                 bool                               filterCenterOut)
    : m_points(points)
{
    p_updateNeighbourList(filterCenterOut);
}

int Stencil::nPoints() const
{
    return int(m_points.size());
}

auto Stencil::find(const Neon::index_3d& direction) const -> int
{
    auto it = std::find(m_neighbour.begin(), m_neighbour.end(), direction);
    if (it == m_neighbour.end()) {
        return -1;
    }
    int index = static_cast<int>(std::distance(m_neighbour.begin(), it));
    return index;
}

Stencil Stencil::s19_t(bool filterCenterOut)
{
    Stencil res;
    res.m_points = std::vector<Neon::index_3d>({{0, 0, 0},
                                                {0, 0, -1},
                                                {0, 0, 1},
                                                {0, -1, 0},
                                                {0, -1, -1},
                                                {0, -1, 1},
                                                {0, 1, 0},
                                                {0, 1, -1},
                                                {0, 1, 1},
                                                {-1, 0, 0},
                                                {-1, 0, -1},
                                                {-1, 0, 1},
                                                {-1, -1, 0},
                                                {-1, 1, 0},
                                                {1, 0, 0},
                                                {1, 0, -1},
                                                {1, 0, 1},
                                                {1, -1, 0},
                                                {1, 1, 0}});
    res.p_updateNeighbourList(filterCenterOut);
    return res;
}

Stencil Stencil::s7_Laplace_t(bool filterCenterOut)
{
    Stencil res;
    res.m_points = std::vector<Neon::index_3d>({{0, 0, 0},
                                                {0, 0, -1},
                                                {0, 0, 1},
                                                {0, -1, 0},
                                                {0, 1, 0},
                                                {-1, 0, 0},
                                                {1, 0, 0}});
    res.p_updateNeighbourList(filterCenterOut);
    return res;
}

Stencil Stencil::s27_t(bool filterCenterOut)
{
    Stencil res;
    res.m_points = std::vector<Neon::index_3d>({
        {-1, -1, -1},
        {-1, -1, 0},
        {-1, -1, 1},

        {-1, 0, -1},
        {-1, 0, 0},
        {-1, 0, 1},

        {-1, 1, -1},
        {-1, 1, 0},
        {-1, 1, 1},


        {0, -1, -1},
        {0, -1, 0},
        {0, -1, 1},

        {0, 0, -1},
        {0, 0, 0},
        {0, 0, 1},

        {0, 1, -1},
        {0, 1, 0},
        {0, 1, 1},


        {1, -1, -1},
        {1, -1, 0},
        {1, -1, 1},

        {1, 0, -1},
        {1, 0, 0},
        {1, 0, 1},

        {1, 1, -1},
        {1, 1, 0},
        {1, 1, 1},


    });
    res.p_updateNeighbourList(filterCenterOut);
    return res;
}

Stencil Stencil::s6_Jacobi_t()
{
    Stencil res;
    res.m_points = std::vector<Neon::index_3d>({{0, 0, -1},
                                                {0, 0, 1},
                                                {0, -1, 0},
                                                {0, 1, 0},
                                                {-1, 0, 0},
                                                {1, 0, 0}});
    res.p_updateNeighbourList(false);
    return res;
}


void Stencil::p_updateNeighbourList(bool filterCenterOut)
{
    if (!filterCenterOut) {
        m_neighbour = m_points;
    } else {
        m_neighbour = std::vector<Neon::index_3d>();
        for (const auto& p : m_points) {
            if (p.x != 0 || p.y != 0 || p.z != 0) {
                m_neighbour.push_back(p);
            }
        }
    }
}

auto Stencil::points() const
    -> const std::vector<Neon::index_3d>&
{
    return m_points;
}

auto Stencil::neighbours() const
    -> const std::vector<Neon::index_3d>&
{
    return m_neighbour;
}

auto Stencil::nNeighbours() const
    -> int32_t
{
    return int(m_neighbour.size());
}
auto Stencil::getUnion(const std::vector<Stencil>& vec) -> Stencil
{
    if (vec.empty()) {
        return {};
    }
    Stencil output(vec[0].neighbours(), false);

    for (size_t i = 1; i < vec.size(); i++) {
       for(const auto& point: vec[i].neighbours()){
            output.addPoint(point);
       }
    }
    return output;
}

auto Stencil::addPoint(const index_3d& newPoint) -> void
{
    int position = this->find(newPoint);
    if (position == -1) {
        m_neighbour.push_back(newPoint);
        m_points.push_back(newPoint);
    }
}

}  // namespace Neon::domain
