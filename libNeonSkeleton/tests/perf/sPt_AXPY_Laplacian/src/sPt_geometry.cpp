
#include "./sPt_geometry.h"
#include <map>
#include "gtest/gtest.h"

#include "Neon/core/core.h"
#include "Neon/core/tools/io/ioToVti.h"

using namespace Neon;
using namespace Neon::domain;

namespace eGrid = Neon::domain::details::eGrid;
using eGrid_t = Neon::domain::details::eGrid::eGrid;


Geometry::Geometry(topologies_e topo, const Neon::index_3d& domain_size)
{
    mTopo = topo;
    mDomainSize = domain_size.newType<int64_t>();
    mCenter = domain_size.newType<double>() / 2;
    setImplicit(topo);
}

Geometry::Geometry(topologies_e topo, const Neon::index64_3d& domain_size)
{
    mDomainSize = domain_size;
    mCenter = domain_size.newType<double>() / 2;
    setImplicit(topo);
}

auto Geometry::setImplicit(topologies_e topo) -> void
{
    mTopo = topo;
}

auto Geometry::operator()(const Neon::index_3d& target) const-> bool
{
    switch (mTopo) {
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
auto Geometry::full(const Neon::index_3d& /*target*/) const -> bool
{
    return true;
};

auto Geometry::lowerLeft(const Neon::index_3d& target) const -> bool
{
    auto c = mCenter.rSum();
    auto t = target.rSum();
    return t < c;
};

auto Geometry::cube(const Neon::index_3d& target) const -> bool
{
    auto c = mCenter.newType<double>();
    auto t = target.newType<double>();
    bool xOK = std::abs(c.x - t.x) < c.x - 1;
    bool yOK = std::abs(c.y - t.y) < c.y - 1;
    bool zOK = std::abs(c.z - t.z) < c.z - 1;
    return xOK && yOK && zOK;
    // return (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin() - 1, 2);
};


auto Geometry::hollowCube(const Neon::index_3d& target) const -> bool
{
    auto c = mCenter.newType<double>();
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

auto Geometry::sphere(const Neon::index_3d& target) const -> bool
{
    auto c = mCenter.newType<double>();
    auto t = target.newType<double>();
    return (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin(), 2);
    // return (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin() - 1, 2);
};

auto Geometry::hollowShere(const Neon::index_3d& target) const -> bool
{
    auto c = mCenter.newType<double>();
    auto t = target.newType<double>();
    bool inSphereExt = (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin(), 2);
    bool outSphereIn = (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) > std::pow(c.rMin() - 4, 2);
    return inSphereExt && outSphereIn;
    // return (std::pow(t.x - c.x, 2) + std::pow(t.y - c.y, 2) + std::pow(t.z - c.z, 2)) < std::pow(c.rMin() - 1, 2);
};
