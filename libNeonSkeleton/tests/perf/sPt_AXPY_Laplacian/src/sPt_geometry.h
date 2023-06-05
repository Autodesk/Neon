#pragma once
#include <map>

#include "Neon/core/core.h"
#include "Neon/core/tools/io/ioToVti.h"
#include "Neon/domain/eGrid.h"
#include "gtest/gtest.h"
using namespace Neon;
using namespace Neon::domain;


enum topologies_e
{
    FullDomain,
    LowerLeft,
    Cube,
    HollowCube,
    Sphere,
    HollowSphere
};
namespace {
inline std::string topologiesToString(topologies_e e)
{
    switch (e) {
        case topologies_e::FullDomain: {
            return "FullDomain";
        }
        case topologies_e::LowerLeft: {
            return "LowerLeft";
        }
        case topologies_e::Cube: {
            return "Cube";
        }
        case topologies_e::HollowCube: {
            return "HollowCube";
        }
        case topologies_e::Sphere: {
            return "Sphere";
        }
        case topologies_e::HollowSphere: {
            return "HollowSphere";
        }
        default: {
            return "Error";
        }
    }
}
}  // namespace
struct Geometry
{
   private:
    // INPUT
    topologies_e     mTopo;
    Neon::index64_3d mDomainSize;

    // COMPUTED
    Neon::double_3d mCenter;

   public:
    Geometry() = default;
    Geometry(topologies_e topo, const Neon::index64_3d& domain_size);
    Geometry(topologies_e topo, const Neon::index_3d& domain_size);

   private:
    auto setImplicit(topologies_e topo) -> void;

   public:
    auto operator()(const Neon::index_3d& target) const -> bool;

   private:
    auto full(const Neon::index_3d& target) const -> bool;
    auto lowerLeft(const Neon::index_3d& target) const -> bool;
    auto cube(const Neon::index_3d& target) const -> bool;
    auto hollowCube(const Neon::index_3d& target) const -> bool;
    auto sphere(const Neon::index_3d& target) const -> bool;
    auto hollowShere(const Neon::index_3d& target) const -> bool;
};
