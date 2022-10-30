#pragma once

#include "Neon/core/core.h"

#include "Neon/set/container/ContainerAPI.h"
#include "Neon/set/container/Graph.h"
#include "Neon/set/container/GraphContainer.h"
#include "Neon/set/container/Loader.h"
#include "Neon/set/container/types/SynchronizationContainerType.h"

namespace Neon::set {
struct Container;
}

namespace Neon::set::internal {

template <typename MultiXpuDataT>
struct HaloUpdateContainer
    : public GraphContainer
{

   public:
    ~HaloUpdateContainer() override = default;

    HaloUpdateContainer(const Neon::Backend&        bk,
                        const Neon::set::Container& dataTransferContainer,
                        const Neon::set::Container& syncContainer);

   private:
};

}  // namespace Neon::set::internal

#include "Neon/set/container/HaloUpdateContainer_imp.h"