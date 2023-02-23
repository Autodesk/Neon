#pragma once
#include "Neon/core/core.h"
#include "Neon/set/MemoryOptions.h"

namespace Neon::domain {

template <typename SP>
concept SparsityPattern = requires(SP sparsityPattern) {
    {
        sparsityPattern(Neon::index_3d(0,0,0))
    } -> std::same_as<bool>;
};

template <typename G>
concept Grid = requires(G grid) {
    {
        grid.template newField<int>("fieldName",
                                    int(0),
                                    int(0),
                                    Neon::DataUse::IO_COMPUTE,
                                    Neon::MemoryOptions())
    } -> std::same_as<typename G::template Field<int, 0>>;
};

}  // namespace Neon::domain
