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


template <Neon::Execution execution, typename LoadingLambda>
auto mGrid::newContainer(const std::string& name,
                         int                level,
                         index_3d           blockSize,
                         size_t             sharedMem,
                         LoadingLambda      lambda) const -> Neon::set::Container
{


    Neon::set::Container kContainer = mData->grids[level].template newContainer<execution>(name, blockSize, sharedMem, lambda);

    return kContainer;
}

template <Neon::Execution execution, typename LoadingLambda>
auto mGrid::newContainer(const std::string& name,
                         int                level,
                         LoadingLambda      lambda) const -> Neon::set::Container
{
    Neon::set::Container kContainer = mData->grids[level].template newContainer<execution>(name, lambda);

    return kContainer;
}

template <Neon::Execution execution, typename LoadingLambda>
auto mGrid::newContainer(const std::string& name,
                         int                level,
                         bool               isAlpha,
                         LoadingLambda      lambda) const -> Neon::set::Container
{
    if (isAlpha) {
        return newAlphaContainer<execution>(name, level, lambda);
    } else {
        return newBetaContainer<execution>(name, level, lambda);
    }
}

template <Neon::Execution execution, typename LoadingLambda>
auto mGrid::newAlphaContainer(const std::string& name,
                              int                level,
                              LoadingLambda      lambda) const -> Neon::set::Container
{
    Neon::set::Container kContainer = mData->grids[level].template newAlphaContainer<execution>(name, lambda);

    return kContainer;
}

template <Neon::Execution execution, typename LoadingLambda>
auto mGrid::newBetaContainer(const std::string& name,
                             int                level,
                             LoadingLambda      lambda) const -> Neon::set::Container
{
    Neon::set::Container kContainer = mData->grids[level].template newBetaContainer<execution>(name, lambda);

    return kContainer;
}

template <Neon::Execution execution, typename LoadingLambdaAlpha, typename LoadingLambdaBeta>
auto mGrid::newAlphaBetaContainer(const std::string& name,
                                  int                level,
                                  LoadingLambdaAlpha lambdaAlpha,
                                  LoadingLambdaBeta  lambdaBeta) const -> Neon::set::Container
{
    Neon::set::Container kContainer = mData->grids[level].template newAlphaBetaContainer<execution>(name, lambdaAlpha, lambdaBeta);

    return kContainer;
}


}  // namespace Neon::domain::details::mGrid
