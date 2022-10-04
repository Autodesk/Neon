#include "Neon/domain/dGrid.h"
#include "Neon/domain/internal/experimantal/staggeredGrid/StaggeredGrid.h"

template <typename StaggeredGrid, typename T>
struct Containers
{
    using Self = Containers<StaggeredGrid, T>;

    using Type = T;
    using NodeField = typename StaggeredGrid::template NodeField<T, 0>;
    using VoxelField = typename StaggeredGrid::template VoxelField<T, 0>;

    static auto mapOnNodes(NodeField&  input_field,
                           NodeField&  output_field,
                           const Type& alpha)
        -> Neon::set::Container;
};


extern template struct Containers<Neon::domain::internal::experimental::staggeredGrid::StaggeredGrid<Neon::domain::dGrid>, double>;
