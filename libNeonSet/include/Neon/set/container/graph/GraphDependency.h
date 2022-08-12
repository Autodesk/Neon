#pragma once

#include "GraphDependencyType.h"

namespace Neon::set::container {

struct GraphDependency
{
   public:
    GraphDependency();
    GraphDependency(GraphDependencyType type);

    auto setType(GraphDependencyType type) -> void;
    auto getType() const-> GraphDependencyType;

   private:
    GraphDependencyType mType;
    // TODO - add information for data and Scheduling dependency
};

}  // namespace Neon::set::container
