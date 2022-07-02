
#include "./sPt_geometry.h"
#include <map>
#include "gtest/gtest.h"

#include "Neon/core/core.h"
#include "Neon/core/tools/io/ioToVti.h"

using namespace Neon;
using namespace Neon::domain;

namespace eGrid = Neon::domain::internal::eGrid;
using eGrid_t = Neon::domain::internal::eGrid::eGrid;


geometry_t::geometry_t(topologies_e topo, const Neon::index_3d& domain_size)
{
    m_topo = topo;
    m_domainSize = domain_size.newType<int64_t>();
    m_center = domain_size.newType<double>() / 2;
    setImplicit(topo);
}

geometry_t::geometry_t(topologies_e topo, const Neon::index64_3d& domain_size)
{
    m_domainSize = domain_size;
    m_center = domain_size.newType<double>() / 2;
    setImplicit(topo);
}

auto geometry_t::setImplicit(topologies_e topo) -> void
{
    m_topo = topo;
}

auto geometry_t::operator()(const Neon::index_3d& target) -> bool
{
    switch (m_topo) {
        case FullDomain: {
            return full(target);
        }
        case LowerLeft: {
            return lowerLeft(target);
            ;
        }
        case Cube: {
            return cube(target);
        }
        case HollowCube: {
            return hollowCube(target);
            ;
            ;
        }
        case Sphere: {
            return sphere(target);
            ;
            break;
        }
        case HollowSphere: {
            return hollowShere(target);
            ;
        }
        default: {
            NEON_THROW_UNSUPPORTED_OPTION();
        }
    }
}
auto geometry_t::full(const Neon::index_3d& /*target*/) -> bool
{
    return true;
};

auto geometry_t::lowerLeft(const Neon::index_3d& target) -> bool
{
    auto c = m_center.rSum();
    auto t = target.rSum();
    return t < c;
};

auto geometry_t::cube(const Neon::index_3d& target) -> bool
{
    auto c = m_center.newType<double>();
    auto t = target.newType<double>();
    bool xOK = std::abs(c.x - t.x) < c.x - 1;
    bool yOK = std::abs(c.y - t.y) < c.y - 1;
    bool zOK = std::abs(c.z - t.z) < c.z - 1;
    return xOK && yOK && zOK;
    // return (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin() - 1, 2);
};


auto geometry_t::hollowCube(const Neon::index_3d& target) -> bool
{
    auto c = m_center.newType<double>();
    auto t = target.newType<double>();
    bool testA;
    {
        bool xOK = std::abs(c.x - t.x) < c.x - 1;
        bool yOK = std::abs(c.y - t.y) < c.y - 1;
        bool zOK = std::abs(c.z - t.z) < c.z - 1;
        testA = xOK && yOK && zOK;
    }
    bool testB;
    {
        bool xOK = std::abs(c.x - t.x) < (c.x - 1) / 3;
        bool yOK = std::abs(c.y - t.y) < (c.x - 1) / 3;
        bool zOK = std::abs(c.z - t.z) < (c.x - 1) / 3;
        testB = !(xOK && yOK && zOK);
    }
    return testA && testB;
    // return (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin() - 1, 2);
};

auto geometry_t::sphere(const Neon::index_3d& target) -> bool
{
    auto c = m_center.newType<double>();
    auto t = target.newType<double>();
    return (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin(), 2);
    // return (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin() - 1, 2);
};

auto geometry_t::hollowShere(const Neon::index_3d& target) -> bool
{
    auto c = m_center.newType<double>();
    auto t = target.newType<double>();
    bool inSphereExt = (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin(), 2);
    bool outSphereIn = (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) > std::pow(c.rMin() - 4, 2);
    return inSphereExt && outSphereIn;
    // return (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin() - 1, 2);
};
