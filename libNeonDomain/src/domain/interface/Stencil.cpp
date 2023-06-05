#include "Neon/domain/interface/Stencil.h"
#include <unordered_set>


namespace Neon::domain {

Stencil::Stencil(std::vector<Neon::index_3d> const& points,
                 bool                               filterCenterOut)
    : mPoints(points)
{
    p_updateNeighbourList(filterCenterOut);
}

int Stencil::nPoints() const
{
    return int(mPoints.size());
}

auto Stencil::find(const Neon::index_3d& direction) const -> int
{
    auto it = std::find(mNeighbours.begin(), mNeighbours.end(), direction);
    if (it == mNeighbours.end()) {
        return -1;
    }
    int index = static_cast<int>(std::distance(mNeighbours.begin(), it));
    return index;
}

Stencil Stencil::s19_t(bool filterCenterOut)
{
    Stencil res;
    res.mPoints = std::vector<Neon::index_3d>({{0, 0, 0},
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
    res.mPoints = std::vector<Neon::index_3d>({{0, 0, 0},
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
    res.mPoints = std::vector<Neon::index_3d>({
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
    res.mPoints = std::vector<Neon::index_3d>({{0, 0, -1},
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
        mNeighbours = mPoints;
    } else {
        mNeighbours = std::vector<Neon::index_3d>();
        for (const auto& p : mPoints) {
            if (p.x != 0 || p.y != 0 || p.z != 0) {
                mNeighbours.push_back(p);
            }
        }
    }
}

auto Stencil::getRadius() const  -> int32_t
{
    int32_t radius = 0;
    for (const auto& p : this->neighbours()) {
        radius = std::max(radius, std::abs(p.x));
        radius = std::max(radius, std::abs(p.y));
        radius = std::max(radius, std::abs(p.z));
    }
    return radius;
}

auto Stencil::points() const
    -> const std::vector<Neon::index_3d>&
{
    return mPoints;
}

auto Stencil::neighbours() const
    -> const std::vector<Neon::index_3d>&
{
    return mNeighbours;
}

auto Stencil::nNeighbours() const
    -> int32_t
{
    return int(mNeighbours.size());
}
auto Stencil::getUnion(const std::vector<Stencil>& vec) -> Stencil
{
    if (vec.empty()) {
        return {};
    }
    Stencil output(vec[0].neighbours(), false);

    for (size_t i = 1; i < vec.size(); i++) {
        for (const auto& point : vec[i].neighbours()) {
            output.addPoint(point);
        }
    }
    return output;
}

auto Stencil::addPoint(const index_3d& newPoint) -> void
{
    int position = this->find(newPoint);
    if (position == -1) {
        mNeighbours.push_back(newPoint);
        mPoints.push_back(newPoint);
    }
}

}  // namespace Neon::domain
