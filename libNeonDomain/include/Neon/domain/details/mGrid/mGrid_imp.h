#include "Neon/domain/details/mGrid/mGrid.h"

namespace Neon::domain::details::mGrid {

template <typename T, int C>
auto mGrid::newField(const std::string          name,
                     int                        cardinality,
                     T                          inactiveValue,
                     Neon::DataUse              dataUse,
                     const Neon::MemoryOptions& memoryOptions) const -> Field<T, C>
{
    mField<T, C> field(name, *this, cardinality, inactiveValue, dataUse, memoryOptions);

    return field;
}


template <typename LoadingLambda>
auto mGrid::newContainer(const std::string& name,
                         int                level,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda) const -> Neon::set::Container
{


    Neon::set::Container kContainer = mData->grids[level].newContainer(name, blockSize, sharedMem, lambda);

    return kContainer;
}

template <typename LoadingLambda>
auto mGrid::newContainer(const std::string& name,
                         int                level,
                         LoadingLambda      lambda) const -> Neon::set::Container
{
    Neon::set::Container kContainer = mData->grids[level].newContainer(name, lambda);

    return kContainer;
}
}  // namespace Neon::domain::details::mGrid
